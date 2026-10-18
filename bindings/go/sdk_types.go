package mlsr

import (
	"fmt"
	"time"
)

// Failure preserves native request identity and the stage/category of failure.
type Failure struct {
	RequestID uint64  `json:"request_id"`
	Category  string  `json:"category"`
	Stage     string  `json:"stage"`
	Backend   *string `json:"backend"`
	Message   string  `json:"message"`
}

// SDKError is the developer API error. Unwrap retains errors.As compatibility
// with the original low-level *Error without changing that public struct.
type SDKError struct {
	Code    string   `json:"code"`
	Message string   `json:"message"`
	Failure *Failure `json:"failure"`
}

func (e *SDKError) Error() string { return fmt.Sprintf("mlsr %s: %s", e.Code, e.Message) }
func (e *SDKError) Unwrap() error { return &Error{Code: e.Code, Message: e.Message} }

type Outcome struct {
	Kind    string   `json:"kind"`
	Reason  string   `json:"reason,omitempty"`
	Failure *Failure `json:"error,omitempty"`
}
type Completion struct {
	RequestID uint64  `json:"request_id"`
	Frames    uint64  `json:"frames"`
	Outcome   Outcome `json:"outcome"`
}

// Result returns typed failures. Cancellation remains an ordinary outcome.
func (c Completion) Result() error {
	if f := c.Outcome.Failure; f != nil {
		return &SDKError{Code: f.Category, Message: f.Message, Failure: f}
	}
	return nil
}

// Span is half-open; the containing field specifies frames, UTF-8 or UTF-16.
type Span struct {
	Start uint64 `json:"start"`
	End   uint64 `json:"end"`
}
type TextRange struct {
	SegmentID  uint32 `json:"segment_id"`
	SentenceID uint32 `json:"sentence_id"`
	UTF8       Span   `json:"utf8"`
	UTF16      Span   `json:"utf16"`
}

func (r TextRange) Text(original string) (string, error) {
	if r.UTF8.Start > r.UTF8.End || r.UTF8.End > uint64(len(original)) {
		return "", &SDKError{Code: "invalid_range", Message: "text range is outside the original input"}
	}
	return original[r.UTF8.Start:r.UTF8.End], nil
}

type AudioFormat struct {
	SampleRate uint32 `json:"sample_rate"`
	Channels   uint32 `json:"channels"`
}

func (f AudioFormat) Duration(frames uint64) time.Duration {
	if f.SampleRate == 0 {
		return 0
	}
	return time.Duration(float64(frames) / float64(f.SampleRate) * float64(time.Second))
}

// AudioClip owns interleaved Float32 PCM in Go. Do not modify Samples while a
// synchronous WriteWAV call borrows them. Closing native owners preserves it.
type AudioClip struct {
	Samples []float32
	Format  AudioFormat
}

func (a AudioClip) Frames() uint64 {
	if a.Format.Channels == 0 {
		return 0
	}
	return uint64(len(a.Samples)) / uint64(a.Format.Channels)
}
func (a AudioClip) Duration() time.Duration               { return a.Format.Duration(a.Frames()) }
func (a AudioClip) WriteWAV(path string) (WavFile, error) { return sdkWritePCM(a, path) }

type AudioChunk struct {
	RequestID  uint64
	StartFrame uint64
	Audio      AudioClip
	TextRange  TextRange
}
type WavFile struct {
	Path   string      `json:"path"`
	Frames uint64      `json:"frames"`
	Format AudioFormat `json:"format"`
}

type VoiceCapabilities struct {
	PCM                bool `json:"pcm"`
	DurationPrediction bool `json:"duration_prediction"`
	SegmentAlignment   bool `json:"segment_alignment"`
	WordAlignment      bool `json:"word_alignment"`
}
type Voice struct {
	ID                 string            `json:"id"`
	ModelVoiceID       string            `json:"model_voice_id"`
	DisplayName        string            `json:"display_name"`
	Model              string            `json:"model"`
	BundleID           string            `json:"bundle_id"`
	Revision           string            `json:"revision"`
	ManifestSHA256     string            `json:"manifest_sha256"`
	Backend            string            `json:"backend"`
	Languages          []string          `json:"languages"`
	QualifiedLanguages []string          `json:"qualified_languages"`
	RequiresNetwork    bool              `json:"requires_network"`
	Readiness          string            `json:"readiness"`
	Capabilities       VoiceCapabilities `json:"capabilities"`
}
type ResolvedVoice struct {
	Voice           Voice  `json:"voice"`
	RequestedLocale string `json:"requested_locale"`
	Language        string `json:"language"`
	UsedFallback    bool   `json:"used_fallback"`
}
type VoiceFallback uint32

const (
	StrictLocale VoiceFallback = iota
	SameLanguage
)

type OutputMode uint32

const (
	DefaultOutput OutputMode = iota
	HeadlessOutput
	HostManagedOutput
)

type SpeakMode uint32

const (
	Replace SpeakMode = iota
	Enqueue
)

type SupertonicOptions struct {
	Steps uint32
	Seed  uint64
}

// RequestMetadata is the immutable accepted input and effective model settings.
type RequestMetadata struct {
	RequestID    uint64        `json:"request_id"`
	Text         string        `json:"text"`
	Voice        ResolvedVoice `json:"voice"`
	SpeakingRate float32       `json:"speaking_rate"`
	Backend      struct {
		Model string `json:"model"`
		SupertonicOptions
	} `json:"backend"`
	EffectiveSynthesis struct {
		Steps uint32  `json:"steps"`
		Speed float32 `json:"speed"`
		Seed  uint64  `json:"seed"`
	} `json:"effective_synthesis"`
	StartSegment uint32 `json:"start_segment"`
}

// SpeechRequest is copied by Rust on admission. SpeakingRate 1.0 is normal;
// the original low-level Request.Speed keeps its raw-model-speed meaning.
type SpeechRequest struct {
	Text, Locale, Voice string
	Fallback            VoiceFallback
	SpeakingRate        float32
	Backend             SupertonicOptions
}

func DefaultSpeechRequest(text string) SpeechRequest {
	return SpeechRequest{Text: text, Locale: "en", SpeakingRate: 1, Backend: SupertonicOptions{Steps: 8}}
}

type TTSOptions struct {
	Output                                                                OutputMode
	MaxRequests, MaxTextBytes, QueueFrames, ChunkFrames, MaxSegmentFrames uint32
	MaxRetainedStores                                                     uint32
	MaxRetainedBytes                                                      uint64
}

func DefaultTTSOptions() TTSOptions {
	return TTSOptions{MaxRequests: 16, MaxTextBytes: 256 * 1024, QueueFrames: 44100,
		ChunkFrames: 4096, MaxSegmentFrames: 512 * 3072, MaxRetainedStores: 128, MaxRetainedBytes: 128 * 1024 * 1024}
}

type SpeechOptions struct {
	PlaybackRate, Volume float32
	Paused, RetainAudio  bool
	LookaheadFrames      uint32
}

func DefaultSpeechOptions() SpeechOptions {
	return SpeechOptions{PlaybackRate: 1, Volume: 1, RetainAudio: true, LookaheadFrames: 88200}
}

type OutputCapabilities struct {
	MaxChunkFrames     uint32      `json:"max_chunk_frames"`
	PauseResume        bool        `json:"pause_resume"`
	Volume             bool        `json:"volume"`
	RateRange          *[2]float32 `json:"rate_range"`
	Rebase             bool        `json:"rebase"`
	HostManagedPolicy  bool        `json:"host_managed_policy"`
	RouteEvents        bool        `json:"route_events"`
	InterruptionEvents bool        `json:"interruption_events"`
}
type RetentionStats struct {
	UsedBytes  uint64 `json:"used_bytes"`
	PeakBytes  uint64 `json:"peak_bytes"`
	LiveStores uint32 `json:"live_stores"`
	PeakStores uint32 `json:"peak_stores"`
}
type TTSState struct {
	OutputCapabilities *OutputCapabilities `json:"output_capabilities"`
	Retention          RetentionStats      `json:"retention"`
}
type SynthesisState struct {
	Revision              uint64      `json:"revision"`
	RequestID             uint64      `json:"request_id"`
	Phase                 string      `json:"phase"`
	CancellationRequested bool        `json:"cancellation_requested"`
	PlanReady             bool        `json:"plan_ready"`
	GeneratedFrames       uint64      `json:"generated_frames"`
	ConsumedFrames        uint64      `json:"consumed_frames"`
	QueuedFrames          uint64      `json:"queued_frames"`
	PeakQueuedFrames      uint64      `json:"peak_queued_frames"`
	Format                AudioFormat `json:"format"`
	EstimatedFrames       *uint64     `json:"estimated_frames"`
	DurationFrames        *uint64     `json:"duration_frames"`
	Completion            *Completion `json:"completion"`
}
type SpeechState struct {
	Revision             uint64      `json:"revision"`
	RequestID            uint64      `json:"request_id"`
	Synthesis            string      `json:"synthesis"`
	Playback             string      `json:"playback"`
	Format               AudioFormat `json:"format"`
	PositionFrame        uint64      `json:"position_frame"`
	BufferedUntilFrame   uint64      `json:"buffered_until_frame"`
	SeekableRanges       []Span      `json:"seekable_ranges"`
	EstimatedFrames      *uint64     `json:"estimated_frames"`
	DurationFrames       *uint64     `json:"duration_frames"`
	PlaybackRate         float32     `json:"playback_rate"`
	Volume               float32     `json:"volume"`
	ActiveTextRange      *TextRange  `json:"active_text_range"`
	OutputLatencySeconds *float64    `json:"output_latency_seconds"`
	Cancellation         *string     `json:"cancellation"`
	Completion           *Completion `json:"completion"`
}

func (s SpeechState) Position() time.Duration { return s.Format.Duration(s.PositionFrame) }

type TextSegment struct {
	ID         uint32 `json:"id"`
	SentenceID uint32 `json:"sentence_id"`
	UTF8       Span   `json:"utf8"`
	UTF16      Span   `json:"utf16"`
	Original   string `json:"original"`
	Normalized string `json:"normalized"`
}
type TextSentence struct {
	ID       uint32 `json:"id"`
	UTF8     Span   `json:"utf8"`
	UTF16    Span   `json:"utf16"`
	Segments Span   `json:"segments"`
}
type TextPlan struct {
	Original       string         `json:"original"`
	Language       string         `json:"language"`
	ManifestSHA256 string         `json:"manifest_sha256"`
	Segments       []TextSegment  `json:"segments"`
	Sentences      []TextSentence `json:"sentences"`
}
type Navigation struct {
	Kind   string
	Frame  uint64
	Speech *Speech
}
