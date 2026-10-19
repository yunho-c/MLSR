# Maintainer documentation

These pages support development and maintenance of MLSR. For library setup and
API usage, start with the [public documentation](../README.md).

## Architecture and runtime notes

| Document | Contents |
| --- | --- |
| [Architecture](architecture.md) | Module responsibilities, dependency direction and language adapter boundaries |
| [Core ML runtime](coreml-runtime.md) | Native ownership, worker sessions, tensor transfers and output-buffer reuse |
| [QNN runtime](../qnn-runtime.md) | Experimental Windows ARM64 Superfast setup and headless SDK usage |
| [Supertonic3 synthesis](supertonic3.md) | Model pipeline, preprocessing, duration/noise preparation and reference replay |
| [Apple output](apple-output.md) | Native playback implementation, clock behavior and adapter qualification |
| [Execution variants](execution-variants.md) | Experimental graphs, routing, parity, benchmarking and quantization candidates |
| [Playback feasibility probe](playback-probe.md) | Initial native clock, completion and pitch-preservation investigation |

## Implementation and qualification records

| Milestone | Implementation checklist | Qualification evidence |
| --- | --- | --- |
| Core ML proof of concept | [Progress](poc-progress.md) | [POC qualification](poc-qualification.md) |
| Bundled model delivery | [Progress](model-delivery-progress.md) | [Delivery qualification](model-delivery-qualification.md) |
| Developer API v0 | [Progress](developer-api-progress.md) | [SDK qualification](developer-api-qualification.md) |
| QNN Superfast integration | [Runtime and reproduction](../qnn-runtime.md) | [Adapter, model delivery and headless SDK qualification](../qnn-qualification.md) |

Qualification records describe their recorded source revisions, artifacts and
tested hosts. Historical results do not automatically qualify subsequent changes.
Keep compact evidence in [test fixtures](../../tests/fixtures/) and large model,
audio, trace and application artifacts in the external locations named by each
record. Commands assume the repository root unless stated otherwise.

Keep application-facing behavior, setup requirements and current platform limits
in the public guides. Link to these records when detailed engineering evidence
is useful.
