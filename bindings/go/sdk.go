package mlsr

import (
	"context"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
)

type sdkOwned struct {
	handle atomic.Uint64
	once   sync.Once
	closed chan struct{}
	err    error
}

func newSDKOwned(handle uint64) *sdkOwned {
	r := &sdkOwned{closed: make(chan struct{})}
	r.handle.Store(handle)
	return r
}
func (r *sdkOwned) get() (uint64, error) {
	h := r.handle.Load()
	if h == 0 {
		return 0, &SDKError{Code: "handle_closed", Message: "SDK handle is closed"}
	}
	return h, nil
}
func (r *sdkOwned) close(forget bool) error {
	r.once.Do(func() { h := r.handle.Swap(0); close(r.closed); r.err = sdkRelease(h, forget) })
	return r.err
}
func validContext(ctx context.Context) error {
	if ctx == nil {
		return &SDKError{Code: "invalid_argument", Message: "nil context"}
	}
	return ctx.Err()
}

// requestLife is separate from the public handle so a context watcher cannot
// keep that handle reachable and defeat its GC fallback. It never consumes PCM.
type requestLife struct {
	resource           *sdkOwned
	speech, monitored  bool
	mu                 sync.Mutex
	detached, finished bool
	monitorDone        chan struct{}
}

func cancelRequest(handle uint64, speech bool) error {
	if speech {
		return sdkCommand(handle, 0, 0)
	}
	return sdkSynthesisCommand(handle, false)
}
func newRequestLife(ctx context.Context, handle uint64, speech bool) *requestLife {
	life := &requestLife{resource: newSDKOwned(handle), speech: speech, monitored: ctx.Done() != nil, monitorDone: make(chan struct{})}
	if !life.monitored {
		close(life.monitorDone)
		return life
	}
	stop := context.AfterFunc(ctx, func() { _ = cancelRequest(handle, speech) })
	go func() {
		defer close(life.monitorDone)
		defer stop()
		for {
			select {
			case <-life.resource.closed:
				return
			default:
			}
			result, err := sdkWait(handle, 50)
			if result != nil || err != nil {
				life.mu.Lock()
				life.finished = true
				retire := life.detached
				life.mu.Unlock()
				if retire {
					_ = life.resource.close(true)
				}
				return
			}
		}
	}()
	return life
}
func (l *requestLife) abandon() {
	// A discarded Speech still obeys a live cancellation context. The bounded
	// completion watcher owns its C registry ID until terminal, then forgets it.
	// A background-context Speech can be forgotten immediately: Rust owns play.
	l.mu.Lock()
	keep := l.speech && l.monitored && !l.finished
	if keep {
		l.detached = true
	}
	l.mu.Unlock()
	if !keep {
		_ = l.resource.close(true)
	}
}
func (l *requestLife) close() error {
	err := l.resource.close(false)
	<-l.monitorDone
	return err
}

// Voices discovers declared metadata without loading models or opening output.
func Voices() ([]Voice, error) { return sdkVoices(0) }
func ResolveVoice(locale, voice string, fallback VoiceFallback) (ResolvedVoice, error) {
	return sdkResolve(0, locale, voice, fallback)
}

// TTS owns the shared inference lane, output controllers and retention budget.
// Do not copy handles. Explicit Close is the deterministic cleanup boundary.
type TTS struct{ resource *sdkOwned }

// OpenTTS verifies local assets and loads the persistent engine. If ctx cancels
// during an uncancellable native load, the result is closed before returning.
// Use DefaultTTSOptions for normal budgets; it does not open an output device.
func OpenTTS(ctx context.Context, manifest, bundle string, engine EngineOptions, options TTSOptions) (*TTS, error) {
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	h, err := sdkOpen(manifest, bundle, engine, options)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		_ = sdkRelease(h, false)
		return nil, err
	}
	return newTTS(h), nil
}
func newTTS(h uint64) *TTS {
	t := &TTS{resource: newSDKOwned(h)}
	runtime.SetFinalizer(t, func(t *TTS) { _ = t.Close() })
	return t
}
func (t *TTS) Close() error { defer runtime.KeepAlive(t); return t.resource.close(false) }
func (t *TTS) Voices() ([]Voice, error) {
	defer runtime.KeepAlive(t)
	h, err := t.resource.get()
	if err != nil {
		return nil, err
	}
	return sdkVoices(h)
}
func (t *TTS) ResolveVoice(locale, voice string, fallback VoiceFallback) (ResolvedVoice, error) {
	defer runtime.KeepAlive(t)
	h, err := t.resource.get()
	if err != nil {
		return ResolvedVoice{}, err
	}
	return sdkResolve(h, locale, voice, fallback)
}
func (t *TTS) Snapshot() (TTSState, error) {
	defer runtime.KeepAlive(t)
	var s TTSState
	h, err := t.resource.get()
	if err != nil {
		return s, err
	}
	err = sdkSnapshot(h, &s)
	return s, err
}

// Controller returns an owned handle to the default controller (independent=false)
// or creates a new replace/enqueue scope. Closing a default handle closes that
// default controller permanently; it does not create a second default on reuse.
func (t *TTS) Controller(independent bool) (*SpeechController, error) {
	defer runtime.KeepAlive(t)
	h, err := t.resource.get()
	if err != nil {
		return nil, err
	}
	handle, err := sdkController(h, independent)
	if err != nil {
		return nil, err
	}
	c := &SpeechController{resource: newSDKOwned(handle), owner: t}
	runtime.SetFinalizer(c, func(c *SpeechController) { _ = c.Close() })
	return c, nil
}

// Speak accepts and returns immediately; the context cancels native playback
// even if no state or audio is consumed. Native speech sends no PCM through Go.
func (t *TTS) Speak(ctx context.Context, request SpeechRequest, options SpeechOptions, mode SpeakMode) (*Speech, error) {
	defer runtime.KeepAlive(t)
	return speak(ctx, t.resource, t, request, options, mode)
}

// Say is the convenient normal-rate, retained-audio, replace helper.
func (t *TTS) Say(ctx context.Context, text string) (*Speech, error) {
	return t.Speak(ctx, DefaultSpeechRequest(text), DefaultSpeechOptions(), Replace)
}
func speak(ctx context.Context, resource *sdkOwned, owner any, request SpeechRequest, options SpeechOptions, mode SpeakMode) (*Speech, error) {
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	h, err := resource.get()
	if err != nil {
		return nil, err
	}
	handle, id, err := sdkSpeak(h, request, options, mode)
	if err != nil {
		return nil, err
	}
	return newSpeech(ctx, owner, handle, id), nil
}

// Synthesize creates lazy work. State, plan and completion reads never start it.
func (t *TTS) Synthesize(ctx context.Context, request SpeechRequest) (*SynthesisJob, error) {
	defer runtime.KeepAlive(t)
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	h, err := t.resource.get()
	if err != nil {
		return nil, err
	}
	handle, id, err := sdkSynthesize(h, request)
	if err != nil {
		return nil, err
	}
	j := &SynthesisJob{requestHandle: requestHandle{ID: id, life: newRequestLife(ctx, handle, false), owner: t}}
	runtime.SetFinalizer(j, func(j *SynthesisJob) { j.life.abandon() })
	return j, nil
}

// Collect claims the job's sole consumer. Zero maxBytes selects 32 MiB. Context
// cancellation reaches Rust independently of the blocking collection call.
func (t *TTS) Collect(ctx context.Context, job *SynthesisJob, maxBytes uint64) (AudioClip, error) {
	defer runtime.KeepAlive(t)
	defer runtime.KeepAlive(job)
	if err := validContext(ctx); err != nil {
		return AudioClip{}, err
	}
	owner, err := t.resource.get()
	if err != nil {
		return AudioClip{}, err
	}
	h, err := job.life.resource.get()
	if err != nil {
		return AudioClip{}, err
	}
	stop := context.AfterFunc(ctx, func() { _ = cancelRequest(h, false) })
	defer stop()
	clip, err := sdkCollect(owner, h, maxBytes)
	if ctx.Err() != nil {
		return AudioClip{}, ctx.Err()
	}
	return clip, err
}
func (t *TTS) Render(ctx context.Context, request SpeechRequest, maxBytes uint64) (AudioClip, error) {
	job, err := t.Synthesize(ctx, request)
	if err != nil {
		return AudioClip{}, err
	}
	defer job.Close()
	return t.Collect(ctx, job, maxBytes)
}

// WriteJob incrementally exports a lazy/started job to a new WAV. Zero maxFrames
// selects the 30-minute default; cancellation cleans only its owned partial file.
func (t *TTS) WriteJob(ctx context.Context, job *SynthesisJob, path string, maxFrames uint64) (WavFile, error) {
	defer runtime.KeepAlive(t)
	defer runtime.KeepAlive(job)
	if err := validContext(ctx); err != nil {
		return WavFile{}, err
	}
	owner, err := t.resource.get()
	if err != nil {
		return WavFile{}, err
	}
	h, err := job.life.resource.get()
	if err != nil {
		return WavFile{}, err
	}
	stop := context.AfterFunc(ctx, func() { _ = cancelRequest(h, false) })
	defer stop()
	file, err := sdkWrite(owner, h, path, maxFrames, false)
	if ctx.Err() != nil {
		return WavFile{}, ctx.Err()
	}
	return file, err
}
func (t *TTS) RenderToFile(ctx context.Context, request SpeechRequest, path string, maxFrames uint64) (WavFile, error) {
	job, err := t.Synthesize(ctx, request)
	if err != nil {
		return WavFile{}, err
	}
	defer job.Close()
	return t.WriteJob(ctx, job, path, maxFrames)
}

// WriteSpeech exports completed retained PCM without inference. This bounded
// synchronous file operation is joined by TTS.Close; it has no per-export cancel.
func (t *TTS) WriteSpeech(speech *Speech, path string, maxFrames uint64) (WavFile, error) {
	defer runtime.KeepAlive(t)
	defer runtime.KeepAlive(speech)
	owner, err := t.resource.get()
	if err != nil {
		return WavFile{}, err
	}
	h, err := speech.life.resource.get()
	if err != nil {
		return WavFile{}, err
	}
	return sdkWrite(owner, h, path, maxFrames, true)
}

type SpeechController struct {
	resource *sdkOwned
	owner    *TTS
}

func (c *SpeechController) Close() error { defer runtime.KeepAlive(c); return c.resource.close(false) }
func (c *SpeechController) Capabilities() (OutputCapabilities, error) {
	defer runtime.KeepAlive(c)
	var caps OutputCapabilities
	h, err := c.resource.get()
	if err != nil {
		return caps, err
	}
	err = sdkSnapshot(h, &caps)
	return caps, err
}
func (c *SpeechController) Speak(ctx context.Context, request SpeechRequest, options SpeechOptions, mode SpeakMode) (*Speech, error) {
	defer runtime.KeepAlive(c)
	return speak(ctx, c.resource, c, request, options, mode)
}
func (c *SpeechController) Say(ctx context.Context, text string) (*Speech, error) {
	return c.Speak(ctx, DefaultSpeechRequest(text), DefaultSpeechOptions(), Replace)
}

// requestHandle's exported methods are shared by Speech and SynthesisJob.
type requestHandle struct {
	ID    uint64
	life  *requestLife
	owner any
}

func (r *requestHandle) Close() error { defer runtime.KeepAlive(r); return r.life.close() }
func (r *requestHandle) Cancel() error {
	defer runtime.KeepAlive(r)
	h, err := r.life.resource.get()
	if err != nil {
		return err
	}
	return cancelRequest(h, r.life.speech)
}
func (r *requestHandle) Plan() (*TextPlan, error) {
	defer runtime.KeepAlive(r)
	h, err := r.life.resource.get()
	if err != nil {
		return nil, err
	}
	return sdkPlan(h)
}
func (r *requestHandle) Request() (RequestMetadata, error) {
	defer runtime.KeepAlive(r)
	h, err := r.life.resource.get()
	if err != nil {
		return RequestMetadata{}, err
	}
	return sdkRequestMetadata(h)
}

// Completion is a nonblocking observation. Nil means pending, including a lazy job.
func (r *requestHandle) Completion() (*Completion, error) {
	defer runtime.KeepAlive(r)
	h, err := r.life.resource.get()
	if err != nil {
		return nil, err
	}
	return sdkWait(h, 0)
}

// Wait observes completion without consuming PCM. Context cancellation requests
// cancellation and joins native quiescence, then returns ctx.Err. To observe with
// a deadline without cancelling, use Completion or a separate state subscription.
func (r *requestHandle) Wait(ctx context.Context) (Completion, error) {
	defer runtime.KeepAlive(r)
	if ctx == nil {
		return Completion{}, &SDKError{Code: "invalid_argument", Message: "nil context"}
	}
	h, err := r.life.resource.get()
	if err != nil {
		return Completion{}, err
	}
	stop := context.AfterFunc(ctx, func() { _ = cancelRequest(h, r.life.speech) })
	defer stop()
	for {
		result, err := sdkWait(h, 50)
		if err != nil {
			return Completion{}, err
		}
		if result != nil {
			if ctx.Err() != nil {
				return *result, ctx.Err()
			}
			return *result, result.Result()
		}
	}
}

type SynthesisJob struct{ requestHandle }

func (j *SynthesisJob) Start() error {
	defer runtime.KeepAlive(j)
	h, err := j.life.resource.get()
	if err != nil {
		return err
	}
	return sdkSynthesisCommand(h, true)
}
func (j *SynthesisJob) Snapshot() (SynthesisState, error) {
	defer runtime.KeepAlive(j)
	var s SynthesisState
	h, err := j.life.resource.get()
	if err != nil {
		return s, err
	}
	err = sdkSnapshot(h, &s)
	return s, err
}
func (j *SynthesisJob) Audio() (*AudioStream, error) {
	defer runtime.KeepAlive(j)
	h, err := j.life.resource.get()
	if err != nil {
		return nil, err
	}
	handle, err := sdkAudio(h)
	if err != nil {
		return nil, err
	}
	a := &AudioStream{resource: newSDKOwned(handle), owner: j}
	runtime.SetFinalizer(a, func(a *AudioStream) { _ = a.Close() })
	return a, nil
}
func (j *SynthesisJob) States(ctx context.Context) (*StateStream[SynthesisState], error) {
	defer runtime.KeepAlive(j)
	return newStates[SynthesisState](ctx, j, j.life.resource, func(s SynthesisState) bool { return s.Completion != nil })
}

type Speech struct{ requestHandle }

func newSpeech(ctx context.Context, owner any, handle, id uint64) *Speech {
	s := &Speech{requestHandle: requestHandle{ID: id, life: newRequestLife(ctx, handle, true), owner: owner}}
	runtime.SetFinalizer(s, func(s *Speech) { s.life.abandon() })
	return s
}
func (s *Speech) Snapshot() (SpeechState, error) {
	defer runtime.KeepAlive(s)
	var state SpeechState
	h, err := s.life.resource.get()
	if err != nil {
		return state, err
	}
	err = sdkSnapshot(h, &state)
	return state, err
}
func (s *Speech) States(ctx context.Context) (*StateStream[SpeechState], error) {
	defer runtime.KeepAlive(s)
	return newStates[SpeechState](ctx, s, s.life.resource, func(s SpeechState) bool { return s.Completion != nil })
}
func (s *Speech) command(command uint32, value float32) error {
	defer runtime.KeepAlive(s)
	h, err := s.life.resource.get()
	if err != nil {
		return err
	}
	return sdkCommand(h, command, value)
}
func (s *Speech) Pause() error                        { return s.command(1, 0) }
func (s *Speech) Resume() error                       { return s.command(2, 0) }
func (s *Speech) SetVolume(value float32) error       { return s.command(3, value) }
func (s *Speech) SetPlaybackRate(value float32) error { return s.command(4, value) }
func (s *Speech) Seek(frame uint64) error {
	defer runtime.KeepAlive(s)
	h, err := s.life.resource.get()
	if err != nil {
		return err
	}
	return sdkSeek(h, frame)
}

// ReplayFrom gives a completed recording a new request identity, sharing PCM.
func (s *Speech) ReplayFrom(ctx context.Context, frame uint64, options SpeechOptions, mode SpeakMode) (*Speech, error) {
	defer runtime.KeepAlive(s)
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	h, err := s.life.resource.get()
	if err != nil {
		return nil, err
	}
	handle, id, err := sdkReplay(h, frame, options, mode)
	if err != nil {
		return nil, err
	}
	return newSpeech(ctx, s.owner, handle, id), nil
}
func (s *Speech) navigate(ctx context.Context, target int64) (Navigation, error) {
	defer runtime.KeepAlive(s)
	if err := validContext(ctx); err != nil {
		return Navigation{}, err
	}
	h, err := s.life.resource.get()
	if err != nil {
		return Navigation{}, err
	}
	kind, frame, handle, id, err := sdkNavigate(h, target)
	if err != nil {
		return Navigation{}, err
	}
	value := Navigation{Kind: kind, Frame: frame}
	if handle != 0 {
		value.Speech = newSpeech(ctx, s.owner, handle, id)
	}
	return value, nil
}
func (s *Speech) PreviousSentence(ctx context.Context) (Navigation, error) {
	return s.navigate(ctx, -1)
}
func (s *Speech) NextSentence(ctx context.Context) (Navigation, error) { return s.navigate(ctx, -2) }
func (s *Speech) NavigateToSentence(ctx context.Context, id uint32) (Navigation, error) {
	return s.navigate(ctx, int64(id))
}

// AudioStream has one demand-driven consumer; no Go PCM queue or producer.
// Close on early exit. It cancels consumer_dropped and joins independently of Next.
type AudioStream struct {
	resource   *sdkOwned
	owner      *SynthesisJob
	next       sync.Mutex
	ended      bool
	completion atomic.Pointer[Completion]
}

func (a *AudioStream) Close() error { defer runtime.KeepAlive(a); return a.resource.close(false) }
func (a *AudioStream) Completion() *Completion {
	c := a.completion.Load()
	if c == nil {
		return nil
	}
	copy := *c
	return &copy
}
func (a *AudioStream) Next(ctx context.Context) (AudioChunk, error) {
	defer runtime.KeepAlive(a)
	if ctx == nil {
		return AudioChunk{}, &SDKError{Code: "invalid_argument", Message: "nil context"}
	}
	if !a.next.TryLock() {
		return AudioChunk{}, &SDKError{Code: "consumer_busy", Message: "audio already has an active reader"}
	}
	defer a.next.Unlock()
	for {
		if ctx.Err() != nil {
			_ = a.Close()
			return AudioChunk{}, ctx.Err()
		}
		h := a.resource.handle.Load()
		if h == 0 || a.ended {
			return AudioChunk{}, io.EOF
		}
		chunk, completion, status, err := sdkAudioPoll(h, 50)
		if a.resource.handle.Load() == 0 {
			return AudioChunk{}, io.EOF
		}
		if err != nil {
			return AudioChunk{}, err
		}
		if status == 1 {
			continue
		}
		if completion != nil {
			a.ended = true
			a.completion.Store(completion)
			if err := completion.Result(); err != nil {
				return AudioChunk{}, err
			}
			return AudioChunk{}, io.EOF
		}
		if status == 2 {
			a.ended = true
			return AudioChunk{}, io.EOF
		}
		return chunk, nil
	}
}

type stateControl struct {
	stop chan struct{}
	done chan struct{}
	once sync.Once
	mu   sync.Mutex
	err  error
}

// StateStream has a single latest-value slot. Slow readers skip intermediate
// revisions; terminal replaces that slot before C closes. Keep the subscription
// and defer Close. Its cancellation/closure never cancels the observed request.
type StateStream[T any] struct {
	C       <-chan T
	control *stateControl
	owner   any
}

func (s *StateStream[T]) Close() error {
	defer runtime.KeepAlive(s)
	s.control.once.Do(func() { close(s.control.stop) })
	<-s.control.done
	return nil
}
func (s *StateStream[T]) Err() error {
	s.control.mu.Lock()
	defer s.control.mu.Unlock()
	return s.control.err
}
func newStates[T any](ctx context.Context, owner any, resource *sdkOwned, terminal func(T) bool) (*StateStream[T], error) {
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	h, err := resource.get()
	if err != nil {
		return nil, err
	}
	watch, err := sdkWatch(h)
	if err != nil {
		return nil, err
	}
	ch := make(chan T, 1)
	control := &stateControl{stop: make(chan struct{}), done: make(chan struct{})}
	s := &StateStream[T]{C: ch, control: control, owner: owner}
	runtime.SetFinalizer(s, func(s *StateStream[T]) { _ = s.Close() })
	go func() {
		defer close(control.done)
		defer close(ch)
		defer sdkRelease(watch, false)
		fail := func(err error) { control.mu.Lock(); control.err = err; control.mu.Unlock() }
		for {
			select {
			case <-control.stop:
				return
			case <-ctx.Done():
				fail(ctx.Err())
				return
			default:
			}
			var value T
			status, err := sdkStatePoll(watch, 50, &value)
			if err != nil {
				fail(err)
				return
			}
			if status == 2 {
				return
			}
			if status == 1 {
				continue
			}
			// A reader can race the replacement, so drain only if necessary;
			// this goroutine is the sole sender and always has at most one slot.
			select {
			case ch <- value:
			default:
				select {
				case <-ch:
				default:
				}
				ch <- value
			}
			if terminal(value) {
				return
			}
		}
	}()
	return s, nil
}
