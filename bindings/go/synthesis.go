package mlsr

import (
	"context"
	"fmt"
	"io"
	"runtime"
	"sync"
	"sync/atomic"
)

// Error is a native failure with a stable machine-readable code.
type Error struct{ Code, Message string }

func (e *Error) Error() string { return fmt.Sprintf("mlsr %s: %s", e.Code, e.Message) }

// EngineOptions selects permitted devices and output-buffer reuse.
type EngineOptions struct {
	CPUOnly            bool
	DisableOutputReuse bool
	ExecutionVariant   uint32
}

// Request contains owned Go strings, copied by Rust before Start returns.
// Use DefaultRequest to select the standard inference and queue limits.
type Request struct {
	Text, Language, Voice                      string
	Steps                                      uint32
	Speed                                      float32
	Seed                                       uint64
	QueueFrames, ChunkFrames, MaxSegmentFrames uint32
}

// DefaultRequest uses English/M1, eight steps, speed 1.05 and bounded PCM.
func DefaultRequest(text string) Request {
	return Request{Text: text, Language: "en", Voice: "M1", Steps: 8, Speed: 1.05, QueueFrames: 44100, ChunkFrames: 4096, MaxSegmentFrames: 1572864}
}

// Event is a prediction, owned audio chunk, or terminal outcome. TextStart and
// TextEnd are half-open UTF-8 byte offsets in Request.Text (Go string indices).
type Event struct {
	Kind                                    string
	JobID                                   uint64
	SegmentID                               uint32
	TextStart, TextEnd                      uint64
	StartFrame, Frames, PredictedFrames     uint64
	SampleRate, Channels, CompletedSegments uint32
	Samples                                 []float32
	Failure                                 *Error
}

// Terminal reports synthesis completion, which is separate from playback.
func (e Event) Terminal() bool {
	return e.Kind == "completed" || e.Kind == "cancelled" || e.Kind == "failed"
}

// Engine retains models across jobs. Close cancels and joins native work.
type Engine struct{ handle atomic.Uint64 }

// Open verifies local assets and loads Core ML. It performs no downloads.
func Open(manifest, bundle string, options EngineOptions) (*Engine, error) {
	if _, err := Version(); err != nil {
		return nil, err
	}
	handle, err := nativeOpen(manifest, bundle, options)
	if err != nil {
		return nil, err
	}
	e := &Engine{}
	e.handle.Store(handle)
	runtime.SetFinalizer(e, func(e *Engine) { _ = e.Close() })
	return e, nil
}

// Start accepts one active job. A context watcher cancels Rust even if the
// consumer stops calling Next; it never reads or buffers audio itself.
func (e *Engine) Start(ctx context.Context, request Request) (*Job, error) {
	if ctx == nil {
		return nil, &Error{Code: "invalid_argument", Message: "nil context"}
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	handle := e.handle.Load()
	if handle == 0 {
		return nil, &Error{Code: "engine_closed", Message: "engine is closed"}
	}
	jobHandle, id, err := nativeStart(handle, request)
	runtime.KeepAlive(e)
	if err != nil {
		return nil, err
	}
	j := &Job{ID: id, done: make(chan struct{})}
	j.handle.Store(jobHandle)
	// Capture only the native ID and done channel, so the watcher does not keep
	// an otherwise unreachable Job alive and defeat its cleanup fallback.
	if ctx.Done() != nil {
		go func(handle uint64, done <-chan struct{}) {
			select {
			case <-ctx.Done():
				_ = nativeCancel(handle)
			case <-done:
			}
		}(jobHandle, j.done)
	}
	runtime.SetFinalizer(j, func(j *Job) { _ = j.Close() })
	return j, nil
}

// Close is idempotent and safe concurrently with submissions. Existing jobs
// remain valid to observe cancellation and release their own handles.
func (e *Engine) Close() error {
	handle := e.handle.Swap(0)
	if handle == 0 {
		return nil
	}
	return nativeReleaseEngine(handle)
}

// Job has one pull consumer. Cancel/Close may run concurrently with Next.
// There is no Go audio queue; callers own returned slices.
type Job struct {
	ID     uint64
	handle atomic.Uint64
	next   sync.Mutex
	ended  bool
	done   chan struct{}
	stop   sync.Once
}

func (j *Job) stopWatcher() { j.stop.Do(func() { close(j.done) }) }

// Next waits for one event. It delivers one terminal event, then io.EOF.
func (j *Job) Next() (Event, error) {
	j.next.Lock()
	defer j.next.Unlock()
	for {
		handle := j.handle.Load()
		if handle == 0 || j.ended {
			return Event{}, io.EOF
		}
		event, status, err := nativePoll(handle, 100)
		runtime.KeepAlive(j)
		if j.handle.Load() == 0 {
			return Event{}, io.EOF
		}
		if err != nil {
			return Event{}, err
		}
		if status == 1 {
			continue
		}
		if status == 2 {
			j.ended = true
			j.stopWatcher()
			return Event{}, io.EOF
		}
		if event.Terminal() {
			j.ended = true
			j.stopWatcher()
		}
		return event, nil
	}
}

// Cancel bypasses the inference queue and does not wait for a native call.
func (j *Job) Cancel() error {
	handle := j.handle.Load()
	if handle == 0 {
		return nil
	}
	err := nativeCancel(handle)
	runtime.KeepAlive(j)
	return err
}

// Wait observes quiescence without consuming events. A stalled producer needs
// cancellation or a consumer. timeoutMS is a millisecond deadline.
func (j *Job) Wait(timeoutMS uint32) (bool, error) {
	handle := j.handle.Load()
	if handle == 0 {
		return true, nil
	}
	complete, err := nativeWait(handle, timeoutMS)
	runtime.KeepAlive(j)
	return complete, err
}

// Close cancels and waits. It wakes both a stalled producer and a timed Next.
func (j *Job) Close() error {
	handle := j.handle.Swap(0)
	j.stopWatcher()
	if handle == 0 {
		return nil
	}
	return nativeReleaseJob(handle)
}
