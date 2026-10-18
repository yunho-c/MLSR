package mlsr

/*
#include "mlsr.h"
// Build the borrowed view on the C stack. Go passes only a pointer to Float32
// storage (which contains no Go pointers), never a Go struct containing it.
static MlsrStatus mlsr_go_write_pcm(const float *samples, size_t count,
    uint32_t rate, uint32_t channels, uint64_t frames, MlsrString path,
    MlsrHandle *output, MlsrHandle *error) {
    MlsrPcmView view = {rate, channels, frames, samples, count};
    return mlsr_pcm_write_wav(&view, path, output, error);
}
*/
import "C"

import (
	"encoding/json"
	"runtime"
	"unsafe"
)

func sdkCheck() error {
	if _, err := Version(); err != nil {
		return err
	}
	if uint32(C.mlsr_features())&8 == 0 {
		return &SDKError{Code: "unsupported_abi", Message: "developer SDK schema v1 is unavailable"}
	}
	return nil
}

type sdkEnvelope struct {
	Version uint32          `json:"schema_version"`
	Type    string          `json:"type"`
	Data    json.RawMessage `json:"data"`
}

func sdkJSON(handle C.MlsrHandle, target any) error {
	defer C.mlsr_sdk_release(handle)
	var view C.MlsrString
	var failure C.MlsrHandle
	// Use the original diagnostic decoder here to avoid recursion while reading
	// an error-details JSON handle.
	status := C.mlsr_json_read(handle, &view, &failure)
	if err := nativeError(status, failure); err != nil {
		return err
	}
	var envelope sdkEnvelope
	if err := json.Unmarshal([]byte(gostring(view)), &envelope); err != nil {
		return err
	}
	if envelope.Version != 1 || envelope.Type == "" {
		return &SDKError{Code: "unsupported_schema", Message: "unexpected native metadata"}
	}
	return json.Unmarshal(envelope.Data, target)
}
func sdkError(status C.MlsrStatus, handle C.MlsrHandle) error {
	if status >= 0 {
		return nil
	}
	defer C.mlsr_error_release(handle)
	result := &SDKError{Code: "native_error", Message: "native call failed without a diagnostic"}
	var info C.MlsrErrorInfo
	if handle != 0 && C.mlsr_error_read(handle, &info, C.sizeof_MlsrErrorInfo) == 0 {
		result.Code, result.Message = gostring(info.code), gostring(info.message)
		var output, failure C.MlsrHandle
		if C.mlsr_error_details(handle, &output, &failure) == 0 {
			var detailed SDKError
			if sdkJSON(output, &detailed) == nil {
				result = &detailed
			}
		}
		C.mlsr_error_release(failure)
	}
	return result
}
func sdkRelease(handle uint64, forget bool) error {
	var status C.MlsrStatus
	if forget {
		status = C.mlsr_sdk_forget(C.MlsrHandle(handle))
	} else {
		status = C.mlsr_sdk_release(C.MlsrHandle(handle))
	}
	if status != 0 {
		return &SDKError{Code: "release_failed", Message: "SDK handle release failed"}
	}
	return nil
}
func sdkOpen(manifest, bundle string, engine EngineOptions, options TTSOptions) (uint64, error) {
	if err := sdkCheck(); err != nil {
		return 0, err
	}
	m, freeM := cstring(manifest)
	defer freeM()
	b, freeB := cstring(bundle)
	defer freeB()
	var e C.MlsrEngineOptions
	C.mlsr_engine_options_default(&e)
	if engine.CPUOnly {
		e.compute_policy = 0
	}
	if engine.DisableOutputReuse {
		e.reuse_output_buffers = 0
	}
	e.execution_variant = C.uint32_t(engine.ExecutionVariant)
	var s C.MlsrTtsOptions
	C.mlsr_tts_options_default(&s)
	s.output_mode = C.uint32_t(options.Output)
	s.max_requests = C.uint32_t(options.MaxRequests)
	s.max_text_bytes = C.uint32_t(options.MaxTextBytes)
	s.queue_frames = C.uint32_t(options.QueueFrames)
	s.chunk_frames = C.uint32_t(options.ChunkFrames)
	s.max_segment_frames = C.uint32_t(options.MaxSegmentFrames)
	s.max_retained_stores = C.uint32_t(options.MaxRetainedStores)
	s.max_retained_bytes = C.uint64_t(options.MaxRetainedBytes)
	var output, failure C.MlsrHandle
	status := C.mlsr_tts_open(m, b, &e, &s, &output, &failure)
	return uint64(output), sdkError(status, failure)
}
func sdkVoices(handle uint64) ([]Voice, error) {
	if err := sdkCheck(); err != nil {
		return nil, err
	}
	var output, failure C.MlsrHandle
	status := C.mlsr_voice_catalog(C.MlsrHandle(handle), &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return nil, err
	}
	var catalog struct {
		Voices []Voice `json:"voices"`
	}
	err := sdkJSON(output, &catalog)
	return catalog.Voices, err
}
func sdkResolve(handle uint64, locale, voice string, fallback VoiceFallback) (ResolvedVoice, error) {
	if err := sdkCheck(); err != nil {
		return ResolvedVoice{}, err
	}
	l, freeL := cstring(locale)
	defer freeL()
	v, freeV := cstring(voice)
	defer freeV()
	var output, failure C.MlsrHandle
	status := C.mlsr_voice_resolve(C.MlsrHandle(handle), l, v, C.uint32_t(fallback), &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return ResolvedVoice{}, err
	}
	var value ResolvedVoice
	err := sdkJSON(output, &value)
	return value, err
}
func sdkRequest(request SpeechRequest) (C.MlsrRequest, func()) {
	t, freeT := cstring(request.Text)
	l, freeL := cstring(request.Locale)
	v, freeV := cstring(request.Voice)
	var settings C.MlsrRequest
	C.mlsr_request_default(&settings)
	settings.text, settings.locale, settings.voice = t, l, v
	settings.locale_fallback = C.uint32_t(request.Fallback)
	settings.speaking_rate = C.float(request.SpeakingRate)
	settings.supertonic_steps = C.uint32_t(request.Backend.Steps)
	settings.seed = C.uint64_t(request.Backend.Seed)
	return settings, func() { freeT(); freeL(); freeV() }
}
func sdkSpeechOptions(options SpeechOptions, mode SpeakMode) C.MlsrSpeechOptions {
	var s C.MlsrSpeechOptions
	C.mlsr_speech_options_default(&s)
	s.mode = C.uint32_t(mode)
	s.playback_rate = C.float(options.PlaybackRate)
	s.volume = C.float(options.Volume)
	s.paused = 0
	if options.Paused {
		s.paused = 1
	}
	s.retain_audio = 0
	if options.RetainAudio {
		s.retain_audio = 1
	}
	s.lookahead_frames = C.uint32_t(options.LookaheadFrames)
	return s
}
func sdkSynthesize(owner uint64, request SpeechRequest) (uint64, uint64, error) {
	s, free := sdkRequest(request)
	defer free()
	var output, failure C.MlsrHandle
	status := C.mlsr_tts_synthesize(C.MlsrHandle(owner), &s, &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return 0, 0, err
	}
	return uint64(output), uint64(C.mlsr_sdk_id(output)), nil
}
func sdkSpeak(owner uint64, request SpeechRequest, options SpeechOptions, mode SpeakMode) (uint64, uint64, error) {
	r, free := sdkRequest(request)
	defer free()
	s := sdkSpeechOptions(options, mode)
	var output, failure C.MlsrHandle
	status := C.mlsr_speak(C.MlsrHandle(owner), &r, &s, &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return 0, 0, err
	}
	return uint64(output), uint64(C.mlsr_sdk_id(output)), nil
}
func sdkController(owner uint64, independent bool) (uint64, error) {
	var output, failure C.MlsrHandle
	var flag C.uint32_t
	if independent {
		flag = 1
	}
	status := C.mlsr_tts_controller(C.MlsrHandle(owner), flag, &output, &failure)
	return uint64(output), sdkError(status, failure)
}
func sdkSnapshot(handle uint64, target any) error {
	var output, failure C.MlsrHandle
	status := C.mlsr_sdk_snapshot(C.MlsrHandle(handle), &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return err
	}
	return sdkJSON(output, target)
}
func sdkRequestMetadata(handle uint64) (RequestMetadata, error) {
	var output, failure C.MlsrHandle
	status := C.mlsr_sdk_request(C.MlsrHandle(handle), &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return RequestMetadata{}, err
	}
	var data RequestMetadata
	err := sdkJSON(output, &data)
	return data, err
}
func sdkPlan(handle uint64) (*TextPlan, error) {
	var output, failure C.MlsrHandle
	status := C.mlsr_text_plan(C.MlsrHandle(handle), &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return nil, err
	}
	if status == 1 {
		return nil, nil
	}
	var plan TextPlan
	err := sdkJSON(output, &plan)
	return &plan, err
}
func sdkWait(handle uint64, timeout uint32) (*Completion, error) {
	var output, failure C.MlsrHandle
	status := C.mlsr_sdk_wait(C.MlsrHandle(handle), C.uint32_t(timeout), &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return nil, err
	}
	if status == 1 {
		return nil, nil
	}
	var result Completion
	err := sdkJSON(output, &result)
	return &result, err
}
func sdkCommand(handle uint64, command uint32, value float32) error {
	var failure C.MlsrHandle
	status := C.mlsr_speech_command(C.MlsrHandle(handle), C.uint32_t(command), C.float(value), &failure)
	return sdkError(status, failure)
}
func sdkSynthesisCommand(handle uint64, start bool) error {
	var failure C.MlsrHandle
	var status C.MlsrStatus
	if start {
		status = C.mlsr_synthesis_start(C.MlsrHandle(handle), &failure)
	} else {
		status = C.mlsr_synthesis_cancel(C.MlsrHandle(handle), 0, &failure)
	}
	return sdkError(status, failure)
}
func sdkSeek(handle, frame uint64) error {
	var failure C.MlsrHandle
	status := C.mlsr_speech_seek(C.MlsrHandle(handle), C.uint64_t(frame), &failure)
	return sdkError(status, failure)
}
func sdkReplay(handle, frame uint64, options SpeechOptions, mode SpeakMode) (uint64, uint64, error) {
	s := sdkSpeechOptions(options, mode)
	var output, failure C.MlsrHandle
	status := C.mlsr_speech_replay(C.MlsrHandle(handle), C.uint64_t(frame), &s, &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return 0, 0, err
	}
	return uint64(output), uint64(C.mlsr_sdk_id(output)), nil
}
func sdkNavigate(handle uint64, target int64) (string, uint64, uint64, uint64, error) {
	var speech, output, failure C.MlsrHandle
	status := C.mlsr_speech_navigate(C.MlsrHandle(handle), C.int64_t(target), &speech, &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return "", 0, 0, 0, err
	}
	var result struct {
		Kind  string `json:"kind"`
		Frame uint64 `json:"frame"`
		ID    uint64 `json:"request_id"`
	}
	if err := sdkJSON(output, &result); err != nil {
		C.mlsr_sdk_release(speech)
		return "", 0, 0, 0, err
	}
	return result.Kind, result.Frame, uint64(speech), result.ID, nil
}
func sdkWatch(handle uint64) (uint64, error) {
	var output, failure C.MlsrHandle
	status := C.mlsr_sdk_states(C.MlsrHandle(handle), &output, &failure)
	return uint64(output), sdkError(status, failure)
}
func sdkStatePoll(handle uint64, timeout uint32, target any) (int, error) {
	var output, failure C.MlsrHandle
	status := C.mlsr_state_poll(C.MlsrHandle(handle), C.uint32_t(timeout), &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return int(status), err
	}
	if status == 0 {
		return 0, sdkJSON(output, target)
	}
	return int(status), nil
}
func sdkAudio(handle uint64) (uint64, error) {
	var output, failure C.MlsrHandle
	status := C.mlsr_synthesis_audio(C.MlsrHandle(handle), &output, &failure)
	return uint64(output), sdkError(status, failure)
}
func sdkReadPCM(handle C.MlsrHandle) (AudioClip, error) {
	var info C.MlsrPcmView
	var failure C.MlsrHandle
	status := C.mlsr_audio_read(handle, &info, C.sizeof_MlsrPcmView, &failure)
	if err := sdkError(status, failure); err != nil {
		return AudioClip{}, err
	}
	if info.sample_count > 1024*1024*1024/4 || uint64(info.frames)*uint64(info.channels) != uint64(info.sample_count) {
		return AudioClip{}, &SDKError{Code: "invalid_audio", Message: "unexpected PCM layout"}
	}
	samples := append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(info.samples)), int(info.sample_count))...)
	return AudioClip{Samples: samples, Format: AudioFormat{SampleRate: uint32(info.sample_rate), Channels: uint32(info.channels)}}, nil
}
func sdkAudioPoll(handle uint64, timeout uint32) (AudioChunk, *Completion, int, error) {
	var audio, completion, failure C.MlsrHandle
	status := C.mlsr_audio_stream_poll(C.MlsrHandle(handle), C.uint32_t(timeout), &audio, &completion, &failure)
	if err := sdkError(status, failure); err != nil {
		return AudioChunk{}, nil, int(status), err
	}
	if status != 0 {
		return AudioChunk{}, nil, int(status), nil
	}
	if completion != 0 {
		var c Completion
		err := sdkJSON(completion, &c)
		return AudioChunk{}, &c, 0, err
	}
	defer C.mlsr_sdk_release(audio)
	clip, err := sdkReadPCM(audio)
	if err != nil {
		return AudioChunk{}, nil, 0, err
	}
	var data struct {
		RequestID  uint64 `json:"request_id"`
		StartFrame uint64 `json:"start_frame"`
		TextRange
	}
	if err := sdkSnapshot(uint64(audio), &data); err != nil {
		return AudioChunk{}, nil, 0, err
	}
	return AudioChunk{RequestID: data.RequestID, StartFrame: data.StartFrame, Audio: clip, TextRange: data.TextRange}, nil, 0, nil
}
func sdkCollect(owner, job, maxBytes uint64) (AudioClip, error) {
	var output, failure C.MlsrHandle
	status := C.mlsr_tts_collect(C.MlsrHandle(owner), C.MlsrHandle(job), C.uint64_t(maxBytes), &output, &failure)
	if err := sdkError(status, failure); err != nil {
		return AudioClip{}, err
	}
	defer C.mlsr_sdk_release(output)
	return sdkReadPCM(output)
}
func sdkWrite(owner, source uint64, path string, maxFrames uint64, speech bool) (WavFile, error) {
	p, free := cstring(path)
	defer free()
	var output, failure C.MlsrHandle
	var status C.MlsrStatus
	if speech {
		status = C.mlsr_tts_write_speech(C.MlsrHandle(owner), C.MlsrHandle(source), p, C.uint64_t(maxFrames), &output, &failure)
	} else {
		status = C.mlsr_tts_write_job(C.MlsrHandle(owner), C.MlsrHandle(source), p, C.uint64_t(maxFrames), &output, &failure)
	}
	if err := sdkError(status, failure); err != nil {
		return WavFile{}, err
	}
	var result WavFile
	err := sdkJSON(output, &result)
	return result, err
}
func sdkWritePCM(clip AudioClip, path string) (WavFile, error) {
	if err := sdkCheck(); err != nil {
		return WavFile{}, err
	}
	if clip.Format.Channels == 0 || len(clip.Samples)%int(clip.Format.Channels) != 0 {
		return WavFile{}, &SDKError{Code: "invalid_audio", Message: "PCM must contain complete frames"}
	}
	p, free := cstring(path)
	defer free()
	var output, failure C.MlsrHandle
	status := C.mlsr_go_write_pcm((*C.float)(unsafe.Pointer(unsafe.SliceData(clip.Samples))), C.size_t(len(clip.Samples)),
		C.uint32_t(clip.Format.SampleRate), C.uint32_t(clip.Format.Channels), C.uint64_t(clip.Frames()), p, &output, &failure)
	runtime.KeepAlive(clip.Samples)
	if err := sdkError(status, failure); err != nil {
		return WavFile{}, err
	}
	var result WavFile
	err := sdkJSON(output, &result)
	return result, err
}
