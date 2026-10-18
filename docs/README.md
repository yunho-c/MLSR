# MLSR documentation

Start with the [developer SDK guide](developer-sdk.md) for the shared Rust, C,
Dart, Go and Python API. See [platform support](platforms.md) for tested targets,
toolchain requirements and current limitations, and [model delivery](model-delivery.md)
for preparing and loading model assets.

Language-specific setup and examples are in the [Dart](../bindings/dart/README.md),
[Flutter](../bindings/flutter/README.md), [Go](../bindings/go/README.md) and
[Python](../bindings/python/README.md) package guides. The
[Flutter example](../examples/flutter_speak/README.md) demonstrates speech,
reader controls and WAV export.

## API guides

| Guide | Use it for |
| --- | --- |
| [Developer API contract](developer-api.md) | Request ownership, completion, cancellation, errors and resource limits |
| [Headless synthesis](headless-sdk.md) | Lazy PCM streams, in-memory rendering and WAV export |
| [Speech controllers](speech-controller.md) | Replace/enqueue behavior, playback controls and state observation |
| [Reader controls](reader-api.md) | Retained audio, seeking, replay, timelines and sentence navigation |
| [Voices and text plans](voices-and-text-plans.md) | Voice readiness, locale resolution and original-text coordinates |

## Advanced interfaces

| Guide | Use it for |
| --- | --- |
| [Developer C API](developer-c-api.md) | Facade, speech and rendering handles through the shared C ABI |
| [Engine/job C ABI](ffi.md) | Lower-level synthesis events, PCM ownership and ABI compatibility |
| [Engine jobs](jobs.md) | Direct Rust engine jobs, segmentation, backpressure and cancellation |

## Contributing

[Maintainer documentation](internal/README.md) contains architecture and runtime
notes, implementation checklists, experiments and qualification evidence.
