package mlsr

import (
	"context"
	"runtime"
	"sync"
)

// ModelDescriptor selects exact assets from a trusted application-owned YAML
// catalog. Platform is a catalog key, for example "macos-arm64".
type ModelDescriptor struct{ CatalogYAML, ID, Platform string }

// ModelHandle verifies an immutable directory without loading models or opening
// output. The application owns the files and must retain them for all derived
// engines. Do not copy this handle; Close waits for concurrent engine creation.
type ModelHandle struct {
	mu     sync.RWMutex
	handle uint64
}

func OpenModel(ctx context.Context, directory string, descriptor ModelDescriptor) (*ModelHandle, error) {
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	handle, err := sdkModelOpen(directory, descriptor)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		_ = sdkModelRelease(handle)
		return nil, err
	}
	model := &ModelHandle{handle: handle}
	runtime.SetFinalizer(model, func(model *ModelHandle) { _ = model.Close() })
	return model, nil
}
func (m *ModelHandle) get() (uint64, error) {
	if m.handle == 0 {
		return 0, &SDKError{Code: "handle_closed", Message: "model handle is closed"}
	}
	return m.handle, nil
}
func (m *ModelHandle) Close() error {
	defer runtime.KeepAlive(m)
	m.mu.Lock()
	defer m.mu.Unlock()
	handle := m.handle
	m.handle = 0
	return sdkModelRelease(handle)
}
func (m *ModelHandle) Voices() ([]Voice, error) {
	defer runtime.KeepAlive(m)
	m.mu.RLock()
	defer m.mu.RUnlock()
	handle, err := m.get()
	if err != nil {
		return nil, err
	}
	return sdkVoices(handle)
}
func (m *ModelHandle) ResolveVoice(locale, voice string, fallback VoiceFallback) (ResolvedVoice, error) {
	defer runtime.KeepAlive(m)
	m.mu.RLock()
	defer m.mu.RUnlock()
	handle, err := m.get()
	if err != nil {
		return ResolvedVoice{}, err
	}
	return sdkResolve(handle, locale, voice, fallback)
}

// TTSFromModel takes its own native model lease; model can be closed afterward.
func TTSFromModel(ctx context.Context, model *ModelHandle, engine EngineOptions, options TTSOptions) (*TTS, error) {
	if err := validContext(ctx); err != nil {
		return nil, err
	}
	if model == nil {
		return nil, &SDKError{Code: "invalid_argument", Message: "nil model"}
	}
	defer runtime.KeepAlive(model)
	model.mu.RLock()
	defer model.mu.RUnlock()
	handle, err := model.get()
	if err != nil {
		return nil, err
	}
	result, err := sdkFromModel(handle, engine, options)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		_ = sdkRelease(result, false)
		return nil, err
	}
	return newTTS(result), nil
}
