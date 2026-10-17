# Reference-tool provenance

`frontend.py` adapts normalization rules from Supertone's `py/helper.py` at
`1e9799e964ea4c0dad7cde993b65c3c813a7b373`. The MIT notice below applies to
those portions. `freeze_frontend.py` executes that pinned implementation to
produce independent expected text and token IDs.

The pipeline structure and fixture approach also follow Apache-2.0 sources in:

- [FluidAudio](https://github.com/yunho-c/FluidAudio/tree/e560c6445ffdc42e60c7b282d162aae271f18e2a/Sources/FluidAudio/TTS/Supertonic3)
- [Möbius](https://github.com/FluidInference/mobius/tree/26f5fa8059d5ab3d34449adce6ff6ed86734e522/models/tts/supertonic-3/coreml)

Their licenses are available in their respective repositories. Model weights
and voice assets have separate terms recorded in the model manifest. This
repository's Apache-2.0 license does not relicense those assets.

## Supertone normalization notice

MIT License

Copyright (c) 2025 Supertone Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
