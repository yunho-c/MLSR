#include "mlsr.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static MlsrString str(const char *s) {
    MlsrString value = {(const uint8_t *)s, strlen(s)};
    return value;
}
static void ok(MlsrStatus status, MlsrHandle error) {
    if (status == MLSR_OK) return;
    MlsrErrorInfo info = {0};
    if (mlsr_error_read(error, &info, sizeof(info)) == MLSR_OK)
        fprintf(stderr, "%.*s: %.*s\n", (int)info.code.len, info.code.data,
                (int)info.message.len, info.message.data);
    mlsr_error_release(error);
    assert(status == MLSR_OK);
}
static void contains(MlsrHandle json, const char *expected) {
    MlsrString view = {0};
    MlsrHandle error = 0;
    MlsrStatus status = mlsr_json_read(json, &view, &error);
    ok(status, error);
    char *copy = malloc(view.len + 1);
    assert(copy);
    memcpy(copy, view.data, view.len);
    copy[view.len] = 0;
    assert(strstr(copy, "\"schema_version\":1") && strstr(copy, expected));
    free(copy);
    assert(mlsr_sdk_release(json) == MLSR_OK);
}
int main(int argc, char **argv) {
    assert(argc == 4 || argc == 5); // Manifest, bundle, new evidence directory, optional qnn_htp.
    const int qnn = argc == 5 && strcmp(argv[4], "qnn_htp") == 0;
    const int windows_output = (mlsr_features() & MLSR_FEATURE_WINDOWS_OUTPUT) != 0;
    assert(argc == 4 || qnn);
    assert(mlsr_features() & MLSR_FEATURE_DEVELOPER_SDK);
    if (qnn) assert(mlsr_features() & MLSR_FEATURE_QNN_HTP);
    else assert(mlsr_features() & MLSR_FEATURE_APPLE_OUTPUT);
    const uint64_t initial = mlsr_live_handles();
    MlsrEngineOptions engine;
    MlsrTtsOptions options;
    assert(mlsr_engine_options_default(&engine) == MLSR_OK);
    if (qnn) engine.compute_policy = 2;
    assert(mlsr_tts_options_default(&options) == MLSR_OK);
    options.output_mode = 1; // Headless: never opens output for rendering.
    MlsrHandle tts = 0, error = 0, value = 0;
    MlsrStatus status = mlsr_tts_open(str(argv[1]), str(argv[2]), &engine, &options, &tts, &error);
    ok(status, error);
    status = mlsr_voice_catalog(tts, &value, &error); ok(status, error);
    contains(value, "\"readiness\":\"engine_ready\"");
    status = mlsr_voice_catalog(tts, &value, &error); ok(status, error);
    contains(value, qnn ? "\"backend\":\"qnn\"" : "\"backend\":\"coreml\"");
    MlsrRequest request;
    assert(mlsr_request_default(&request) == MLSR_OK);
    request.text = str("Never start from observation.");
    MlsrHandle lazy = 0;
    if (qnn) {
        request.supertonic_steps = 7;
        status = mlsr_tts_synthesize(tts, &request, &lazy, &error);
        assert(status == MLSR_ERROR && lazy == 0);
        assert(mlsr_error_release(error) == MLSR_OK); error = 0;
        request.supertonic_steps = 8;
    }
    status = mlsr_tts_synthesize(tts, &request, &lazy, &error); ok(status, error);
    status = mlsr_sdk_wait(lazy, 1, &value, &error);
    assert(status == MLSR_TIMEOUT && value == 0 && error == 0);
    status = mlsr_sdk_snapshot(lazy, &value, &error); ok(status, error);
    contains(value, "\"phase\":\"created\"");
    status = mlsr_synthesis_cancel(lazy, 0, &error); ok(status, error);
    status = mlsr_sdk_wait(lazy, 30000, &value, &error); ok(status, error);
    contains(value, "\"reason\":\"requested\"");
    assert(mlsr_sdk_release(lazy) == MLSR_OK);

    const char *languages[] = {"en", "en", "ko", "ko"};
    const char *voices[] = {"supertonic3:M1", "supertonic3:F1", "supertonic3:M1", "supertonic3:F1"};
    const char *names[] = {"en_M1", "en_F1", "ko_M1", "ko_F1"};
    MlsrHandle retained = 0;
    for (unsigned i = 0; i < 4; i++) {
        request.text = str(i < 2 ? "Hello from MLSR." : "안녕하세요. 반갑습니다.");
        request.locale = str(languages[i]); request.voice = str(voices[i]); request.seed = 1729 + i;
        MlsrHandle audio = 0;
        status = mlsr_tts_render(tts, &request, 32 * 1024 * 1024, &audio, &error); ok(status, error);
        MlsrPcmView pcm = {0};
        status = mlsr_audio_read(audio, &pcm, sizeof(pcm), &error); ok(status, error);
        assert(pcm.sample_rate == 44100 && pcm.channels == 1 && pcm.frames == pcm.sample_count);
        assert(pcm.frames > 44100);
        for (uintptr_t n = 0; n < pcm.sample_count; n++) assert(isfinite(pcm.samples[n]));
        char path[4096];
        int count = snprintf(path, sizeof(path), "%s/%s.wav", argv[3], names[i]);
        assert(count > 0 && (size_t)count < sizeof(path));
        status = mlsr_pcm_write_wav(&pcm, str(path), &value, &error); ok(status, error);
        contains(value, "\"type\":\"wav_file\"");
        status = mlsr_pcm_write_wav(&pcm, str(path), &value, &error);
        assert(status == MLSR_ERROR && value == 0);
        assert(mlsr_error_release(error) == MLSR_OK); error = 0;
        if (i == 0) retained = audio;
        else assert(mlsr_sdk_release(audio) == MLSR_OK);
    }
    assert(mlsr_sdk_release(tts) == MLSR_OK);
    // Borrowed view is rooted in its audio handle, not in the facade/model.
    MlsrPcmView pcm = {0};
    status = mlsr_audio_read(retained, &pcm, sizeof(pcm), &error); ok(status, error);
    assert(pcm.frames > 44100 && isfinite(pcm.samples[0]));
    assert(mlsr_sdk_release(retained) == MLSR_OK);

    options.output_mode = 0;
    status = mlsr_tts_open(str(argv[1]), str(argv[2]), &engine, &options, &tts, &error); ok(status, error);
    request.text = str("Hello from MLSR."); request.locale = str("en"); request.voice = str("supertonic3:M1");
    MlsrSpeechOptions playback;
    assert(mlsr_speech_options_default(&playback) == MLSR_OK);
    playback.volume = 0; playback.playback_rate = windows_output ? 1 : 2;
    MlsrHandle speech = 0;
    status = mlsr_speak(tts, &request, &playback, &speech, &error);
    if (qnn && !windows_output) {
        assert(status == MLSR_ERROR && speech == 0);
        MlsrErrorInfo info = {0};
        assert(mlsr_error_read(error, &info, sizeof(info)) == MLSR_OK);
        assert(info.code.len == strlen("unsupported_capability"));
        assert(memcmp(info.code.data, "unsupported_capability", info.code.len) == 0);
        assert(mlsr_error_release(error) == MLSR_OK);
        assert(mlsr_sdk_release(tts) == MLSR_OK);
        assert(mlsr_live_handles() == initial);
        puts("Developer C ABI passed: QNN headless SDK, lazy cancellation, owned PCM, four WAVs, unsupported playback");
        return 0;
    }
    ok(status, error);
    assert(mlsr_sdk_id(speech) != 0);
    status = mlsr_sdk_wait(speech, 30000, &value, &error); ok(status, error);
    contains(value, "\"kind\":\"completed\"");
    status = mlsr_sdk_snapshot(speech, &value, &error); ok(status, error);
    contains(value, "\"playback\":\"ended\"");
    assert(mlsr_sdk_release(tts) == MLSR_OK);
    assert(mlsr_sdk_release(speech) == MLSR_OK);
    assert(mlsr_live_handles() == initial);
    puts("Developer C ABI passed: lazy cancellation, owned PCM, four WAVs and native playback drain");
    return 0;
}
