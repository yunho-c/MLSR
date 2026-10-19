package mlsr

/*
#include "mlsr.h"
*/
import "C"
import "unsafe"

func cstring(value string) (C.MlsrString, func()) {
	pointer := C.CBytes([]byte(value))
	return C.MlsrString{data: (*C.uint8_t)(pointer), len: C.uintptr_t(len(value))}, func() { C.free(pointer) }
}
func gostring(value C.MlsrString) string {
	if value.len == 0 {
		return ""
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(value.data)), int(value.len)))
}
func nativeError(status C.MlsrStatus, handle C.MlsrHandle) error {
	if status >= 0 {
		return nil
	}
	defer C.mlsr_error_release(handle)
	var info C.MlsrErrorInfo
	if handle != 0 && C.mlsr_error_read(handle, &info, C.sizeof_MlsrErrorInfo) == 0 {
		return &Error{Code: gostring(info.code), Message: gostring(info.message)}
	}
	return &Error{Code: "native_error", Message: "native call failed without a diagnostic"}
}
func nativeOpen(manifest, bundle string, options EngineOptions) (uint64, error) {
	m, freeM := cstring(manifest)
	defer freeM()
	b, freeB := cstring(bundle)
	defer freeB()
	var settings C.MlsrEngineOptions
	C.mlsr_engine_options_default(&settings)
	if options.CPUOnly {
		settings.compute_policy = 0
	}
	if options.QNNHTP {
		settings.compute_policy = 2
	}
	if options.DisableOutputReuse {
		settings.reuse_output_buffers = 0
	}
	settings.execution_variant = C.uint32_t(options.ExecutionVariant)
	var handle, failure C.MlsrHandle
	status := C.mlsr_engine_open(m, b, &settings, &handle, &failure)
	return uint64(handle), nativeError(status, failure)
}
func nativeStart(engine uint64, request Request) (uint64, uint64, error) {
	t, freeT := cstring(request.Text)
	defer freeT()
	l, freeL := cstring(request.Language)
	defer freeL()
	v, freeV := cstring(request.Voice)
	defer freeV()
	var settings C.MlsrJobOptions
	C.mlsr_job_options_default(&settings)
	settings.steps = C.uint32_t(request.Steps)
	settings.speed = C.float(request.Speed)
	settings.seed = C.uint64_t(request.Seed)
	settings.queue_frames = C.uint32_t(request.QueueFrames)
	settings.chunk_frames = C.uint32_t(request.ChunkFrames)
	settings.max_segment_frames = C.uint32_t(request.MaxSegmentFrames)
	var handle, failure C.MlsrHandle
	status := C.mlsr_job_start(C.MlsrHandle(engine), t, l, v, &settings, &handle, &failure)
	if err := nativeError(status, failure); err != nil {
		return 0, 0, err
	}
	return uint64(handle), uint64(C.mlsr_job_id(handle)), nil
}
func nativePoll(job uint64, timeout uint32) (Event, int, error) {
	var handle, failure C.MlsrHandle
	status := C.mlsr_job_poll(C.MlsrHandle(job), C.uint32_t(timeout), &handle, &failure)
	if err := nativeError(status, failure); err != nil {
		return Event{}, int(status), err
	}
	if status != 0 {
		return Event{}, int(status), nil
	}
	defer C.mlsr_event_release(handle)
	var info C.MlsrEventInfo
	status = C.mlsr_event_read(handle, &info, C.sizeof_MlsrEventInfo, &failure)
	if err := nativeError(status, failure); err != nil {
		return Event{}, int(status), err
	}
	if info.kind < 1 || info.kind > 5 || info.sample_count > 65536 {
		return Event{}, -1, &Error{Code: "invalid_event", Message: "unexpected native event layout"}
	}
	e := Event{Kind: []string{"predicted", "audio", "completed", "cancelled", "failed"}[int(info.kind)-1],
		JobID: uint64(info.job_id), SegmentID: uint32(info.segment_id), TextStart: uint64(info.text_start), TextEnd: uint64(info.text_end),
		StartFrame: uint64(info.start_frame), Frames: uint64(info.frames), PredictedFrames: uint64(info.predicted_frames),
		SampleRate: uint32(info.sample_rate), Channels: uint32(info.channels), CompletedSegments: uint32(info.completed_segments)}
	if info.sample_count > 0 {
		e.Samples = append([]float32(nil), unsafe.Slice((*float32)(unsafe.Pointer(info.samples)), int(info.sample_count))...)
	}
	if info.kind == 5 {
		e.Failure = &Error{Code: gostring(info.error.code), Message: gostring(info.error.message)}
	}
	return e, 0, nil
}
func nativeCancel(job uint64) error {
	var failure C.MlsrHandle
	status := C.mlsr_job_cancel(C.MlsrHandle(job), &failure)
	return nativeError(status, failure)
}
func nativeWait(job uint64, timeout uint32) (bool, error) {
	var failure C.MlsrHandle
	status := C.mlsr_job_wait(C.MlsrHandle(job), C.uint32_t(timeout), &failure)
	return status == 0, nativeError(status, failure)
}
func nativeReleaseEngine(engine uint64) error {
	if C.mlsr_engine_release(C.MlsrHandle(engine)) != 0 {
		return &Error{Code: "release_failed", Message: "engine release failed"}
	}
	return nil
}
func nativeReleaseJob(job uint64) error {
	if C.mlsr_job_release(C.MlsrHandle(job)) != 0 {
		return &Error{Code: "release_failed", Message: "job release failed"}
	}
	return nil
}
func nativeHandleCount() uint64 { return uint64(C.mlsr_live_handles()) }
