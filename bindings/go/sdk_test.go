package mlsr

import (
	"bytes"
	"context"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestSDKCatalogAndOwnedWAV(t *testing.T) {
	before := nativeHandleCount()
	voices, err := Voices()
	if err != nil {
		t.Fatal(err)
	}
	if len(voices) != 2 || voices[0].ID != "supertonic3:M1" || voices[0].Readiness != "catalog_only" || voices[0].Capabilities.WordAlignment {
		t.Fatalf("unexpected catalog: %+v", voices)
	}
	if _, err := ResolveVoice("en-US", "", StrictLocale); err == nil {
		t.Fatal("implicit locale fallback")
	}
	resolved, err := ResolveVoice("en-US", "supertonic3:F1", SameLanguage)
	if err != nil || !resolved.UsedFallback || resolved.Language != "en" {
		t.Fatalf("resolution: %+v %v", resolved, err)
	}
	clip := AudioClip{Samples: []float32{0, -0.5, 0.75}, Format: AudioFormat{SampleRate: 44100, Channels: 1}}
	path := filepath.Join(t.TempDir(), "clip.wav")
	file, err := clip.WriteWAV(path)
	if err != nil {
		t.Fatal(err)
	}
	if file.Frames != 3 || file.Format != clip.Format {
		t.Fatalf("metadata: %+v", file)
	}
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := clip.WriteWAV(path); err == nil {
		t.Fatal("clobbered existing file")
	}
	again, _ := os.ReadFile(path)
	if !bytes.Equal(content, again) {
		t.Fatal("existing WAV changed")
	}
	invalid := filepath.Join(filepath.Dir(path), "invalid.wav")
	clip.Samples[0] = float32(math.NaN())
	if _, err := clip.WriteWAV(invalid); err == nil {
		t.Fatal("accepted invalid PCM")
	}
	entries, _ := os.ReadDir(filepath.Dir(path))
	if len(entries) != 1 {
		t.Fatal("partial file leaked")
	}
	if nativeHandleCount() != before {
		t.Fatal("metadata/error handle leak")
	}
}

func openSDK(t *testing.T, output bool) *TTS {
	t.Helper()
	if os.Getenv("MLSR_TEST_BUNDLE") == "" {
		t.Skip("real model assets not requested")
	}
	options := DefaultTTSOptions()
	if !output {
		options.Output = HeadlessOutput
	}
	owner, err := OpenTTS(context.Background(), os.Getenv("MLSR_TEST_MANIFEST"), os.Getenv("MLSR_TEST_BUNDLE"), EngineOptions{}, options)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := owner.Close(); err != nil {
			t.Error(err)
		}
	})
	return owner
}
func sdkRequestForTest() SpeechRequest {
	req := DefaultSpeechRequest("Hello from MLSR.\nA second sentence.")
	req.Backend.Seed = 1729
	return req
}

func TestRealSDKLazyStateAndPCM(t *testing.T) {
	before := nativeHandleCount()
	owner := openSDK(t, false)
	ctx := context.Background()
	request := sdkRequestForTest()
	job, err := owner.Synthesize(ctx, request)
	if err != nil {
		t.Fatal(err)
	}
	defer job.Close()
	copied, err := job.Request()
	if err != nil || copied.Text != request.Text || copied.RequestID != job.ID || copied.SpeakingRate != 1 || copied.EffectiveSynthesis.Speed != float32(1.05) || copied.EffectiveSynthesis.Seed != 1729 {
		t.Fatalf("accepted request: %+v %v", copied, err)
	}
	initial, err := job.Snapshot()
	if err != nil || initial.Phase != "created" {
		t.Fatalf("lazy state %+v %v", initial, err)
	}
	if done, err := job.Completion(); err != nil || done != nil {
		t.Fatal("completion started lazy work")
	}
	if plan, err := job.Plan(); err != nil || plan != nil {
		t.Fatal("plan observation started lazy work")
	}
	states, err := job.States(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer states.Close()
	if cap(states.C) != 1 || (<-states.C).Phase != "created" {
		t.Fatal("unbounded/incorrect initial state")
	}
	audio, err := job.Audio()
	if err != nil {
		t.Fatal(err)
	}
	defer audio.Close()
	if second, err := job.Audio(); err == nil {
		second.Close()
		t.Fatal("second PCM consumer accepted")
	}
	var samples []float32
	for {
		chunk, err := audio.Next(ctx)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			t.Fatal(err)
		}
		if chunk.RequestID != job.ID || chunk.StartFrame != uint64(len(samples)) {
			t.Fatal("PCM metadata discontinuity")
		}
		text, err := chunk.TextRange.Text(request.Text)
		if err != nil || text == "" {
			t.Fatal("invalid original range")
		}
		samples = append(samples, chunk.Audio.Samples...)
	}
	done, err := job.Wait(ctx)
	if err != nil || done.Outcome.Kind != "completed" || done.Frames != uint64(len(samples)) {
		t.Fatalf("completion %+v %v", done, err)
	}
	if audio.Completion() == nil {
		t.Fatal("audio lost terminal completion")
	}
	var latest SynthesisState
	for state := range states.C {
		latest = state
	}
	if latest.Completion == nil || states.Err() != nil {
		t.Fatalf("lost coalesced terminal: %+v %v", latest, states.Err())
	}
	states.Close()
	audio.Close()
	job.Close()
	clip, err := owner.Render(ctx, request, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(clip.Samples) != len(samples) {
		t.Fatal("render length mismatch")
	}
	for i, sample := range samples {
		if math.Float32bits(sample) != math.Float32bits(clip.Samples[i]) {
			t.Fatalf("PCM mismatch at %d", i)
		}
	}
	root := t.TempDir()
	a, b := filepath.Join(root, "render.wav"), filepath.Join(root, "clip.wav")
	if _, err := owner.RenderToFile(ctx, request, a, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := clip.WriteWAV(b); err != nil {
		t.Fatal(err)
	}
	first, _ := os.ReadFile(a)
	second, _ := os.ReadFile(b)
	if !bytes.Equal(first, second) {
		t.Fatal("shared WAV output differs")
	}
	_, err = owner.Render(ctx, request, 4)
	var failure *SDKError
	var old *Error
	if !errors.As(err, &failure) || failure.Failure == nil || failure.Failure.Stage != "rendering" || !errors.As(err, &old) {
		t.Fatalf("structured error: %v", err)
	}
	owner.Close()
	if nativeHandleCount() != before {
		t.Fatal("SDK handle leak")
	}
}

func TestRealSDKCancellationAndObservation(t *testing.T) {
	before := nativeHandleCount()
	owner := openSDK(t, false)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	job, err := owner.Synthesize(ctx, sdkRequestForTest())
	if err != nil {
		t.Fatal(err)
	}
	defer job.Close()
	observerCtx, stopObserver := context.WithCancel(context.Background())
	states, err := job.States(observerCtx)
	if err != nil {
		t.Fatal(err)
	}
	<-states.C
	stopObserver()
	states.Close()
	state, err := job.Snapshot()
	if err != nil || state.Phase != "created" || state.CancellationRequested {
		t.Fatal("observer cancellation changed job")
	}
	cancel()
	done, err := job.Wait(context.Background())
	if err != nil || done.Outcome.Reason != "requested" {
		t.Fatalf("context cancellation: %+v %v", done, err)
	}
	job.Close()
	consumerJob, err := owner.Synthesize(context.Background(), sdkRequestForTest())
	if err != nil {
		t.Fatal(err)
	}
	defer consumerJob.Close()
	audio, err := consumerJob.Audio()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := audio.Next(context.Background()); err != nil {
		t.Fatal(err)
	}
	// Close and Cancel may race a consumer without a Go pointer crossing C.
	finished := make(chan struct{})
	go func() { defer close(finished); _, _ = audio.Next(context.Background()) }()
	audio.Close()
	<-finished
	done, err = consumerJob.Wait(context.Background())
	if err != nil || done.Outcome.Reason != "consumer_dropped" {
		t.Fatalf("consumer cancellation: %+v %v", done, err)
	}
	consumerJob.Close()
	owner.Close()
	if nativeHandleCount() != before {
		t.Fatal("cancelled handle leak")
	}
}

func waitSpeechState(t *testing.T, speech *Speech, predicate func(SpeechState) bool) SpeechState {
	t.Helper()
	until := time.Now().Add(10 * time.Second)
	for {
		state, err := speech.Snapshot()
		if err != nil {
			t.Fatal(err)
		}
		if predicate(state) {
			return state
		}
		if time.Now().After(until) {
			t.Fatalf("speech state timed out: %+v", state)
		}
		time.Sleep(5 * time.Millisecond)
	}
}
func TestRealSDKNativeReader(t *testing.T) {
	if os.Getenv("MLSR_TEST_NATIVE_OUTPUT") == "" {
		t.Skip("explicit native playback not requested")
	}
	before := nativeHandleCount()
	owner := openSDK(t, true)
	ctx := context.Background()
	options := DefaultSpeechOptions()
	options.Volume = 0
	options.Paused = true
	request := DefaultSpeechRequest("Hello from MLSR.")
	request.Backend.Seed = 1729
	speech, err := owner.Speak(ctx, request, options, Replace)
	if err != nil {
		t.Fatal(err)
	}
	defer speech.Close()
	initial := waitSpeechState(t, speech, func(s SpeechState) bool { return s.DurationFrames != nil })
	if initial.PositionFrame != 0 || initial.Completion != nil {
		t.Fatal("paused speech advanced/completed")
	}
	queued, err := owner.Speak(ctx, request, options, Enqueue)
	if err != nil {
		t.Fatal(err)
	}
	if err := queued.Cancel(); err != nil {
		t.Fatal(err)
	}
	done, err := queued.Wait(ctx)
	if err != nil || done.Outcome.Reason != "requested" {
		t.Fatal("queued cancellation")
	}
	queued.Close()
	if err := speech.SetPlaybackRate(2); err != nil {
		t.Fatal(err)
	}
	if err := speech.SetVolume(0); err != nil {
		t.Fatal(err)
	}
	if err := speech.Seek(0); err != nil {
		t.Fatal(err)
	}
	if err := speech.Resume(); err != nil {
		t.Fatal(err)
	}
	done, err = speech.Wait(ctx)
	if err != nil || done.Outcome.Kind != "completed" {
		t.Fatalf("playback completion %+v %v", done, err)
	}
	root := t.TempDir()
	file, err := owner.WriteSpeech(speech, filepath.Join(root, "retained.wav"), 0)
	if err != nil || file.Frames != done.Frames {
		t.Fatalf("retained export: %+v %v", file, err)
	}
	replay, err := speech.ReplayFrom(ctx, done.Frames/2, options, Replace)
	if err != nil {
		t.Fatal(err)
	}
	state := waitSpeechState(t, replay, func(s SpeechState) bool { return s.Playback == "paused" })
	if replay.ID == speech.ID || state.PositionFrame != done.Frames/2 {
		t.Fatal("replay identity/position")
	}
	replay.Cancel()
	replay.Wait(ctx)
	replay.Close()
	original, err := speech.Completion()
	if err != nil || original.Outcome.Kind != "completed" {
		t.Fatal("replay changed original outcome")
	}
	speech.Close()
	snapshot, err := owner.Snapshot()
	if err != nil || snapshot.Retention.UsedBytes != 0 {
		t.Fatalf("retention cleanup %+v %v", snapshot, err)
	}

	// Prove that a discarded public handle does not cancel, and its context can
	// still cancel after finalization. The life object holds no Speech pointer.
	detachedCtx, cancel := context.WithCancel(ctx)
	life := func() *requestLife {
		dropped, err := owner.Speak(detachedCtx, request, options, Replace)
		if err != nil {
			t.Fatal(err)
		}
		return dropped.life
	}()
	until := time.Now().Add(3 * time.Second)
	for {
		runtime.GC()
		life.mu.Lock()
		detached := life.detached
		life.mu.Unlock()
		if detached {
			break
		}
		if time.Now().After(until) {
			t.Fatal("public Speech retained by context watcher")
		}
		time.Sleep(10 * time.Millisecond)
	}
	h, err := life.resource.get()
	if err != nil {
		t.Fatal(err)
	}
	if result, err := sdkWait(h, 0); err != nil || result != nil {
		t.Fatal("GC cancelled fire-and-forget speech")
	}
	cancel()
	select {
	case <-life.monitorDone:
	case <-time.After(3 * time.Second):
		t.Fatal("context did not retire discarded speech")
	}
	if life.resource.handle.Load() != 0 {
		t.Fatal("discarded speech registry ID leaked")
	}
	owner.Close()
	if nativeHandleCount() != before {
		t.Fatal("native speech handle leak")
	}
}

func TestSDKVerifiedModelReadiness(t *testing.T) {
	before := nativeHandleCount()
	if _, err := OpenModel(context.Background(), "/missing", ModelDescriptor{CatalogYAML: "not a catalog", ID: "model", Platform: "macos-arm64"}); err == nil {
		t.Fatal("accepted invalid catalog")
	}
	if nativeHandleCount() != before {
		t.Fatal("failed verification leaked an owner")
	}
	catalogPath := os.Getenv("MLSR_TEST_CATALOG")
	if catalogPath == "" {
		t.Skip("verified model catalog not requested")
	}
	catalog, err := os.ReadFile(catalogPath)
	if err != nil {
		t.Fatal(err)
	}
	model, err := OpenModel(context.Background(), os.Getenv("MLSR_TEST_BUNDLE"), ModelDescriptor{CatalogYAML: string(catalog), ID: "supertonic3-coreml-int4-v1", Platform: "macos-arm64"})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	voices, err := model.Voices()
	if err != nil || voices[0].Readiness != "assets_verified" {
		t.Fatalf("verified readiness: %v %v", voices, err)
	}
	resolved, err := model.ResolveVoice("en", "", StrictLocale)
	if err != nil || resolved.Voice.Readiness != "assets_verified" {
		t.Fatalf("verified resolution: %+v %v", resolved, err)
	}
	options := DefaultTTSOptions()
	options.Output = HeadlessOutput
	tts, err := TTSFromModel(context.Background(), model, EngineOptions{}, options)
	if err != nil {
		t.Fatal(err)
	}
	defer tts.Close()
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	voices, err = tts.Voices()
	if err != nil || voices[0].Readiness != "engine_ready" {
		t.Fatalf("loaded readiness: %v %v", voices, err)
	}
	if _, err := model.Voices(); err == nil {
		t.Fatal("closed model remained usable")
	}
	if err := tts.Close(); err != nil {
		t.Fatal(err)
	}
	if nativeHandleCount() != before {
		t.Fatal("model/facade handle leak")
	}
}
