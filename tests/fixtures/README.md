Keep shared test inputs small, redistributable, and documented with their source
and license. Add model fixtures only when a test requires them. Large downloaded
weights belong in a user-selected cache outside this repository.

`supertonic3/` contains the first model corpus, frozen upstream-derived text/token
expectations, and a compact real-capture summary. Full model weights, saved noise,
intermediate tensors, and WAVs remain outside Git. See
[the fixture contract](supertonic3/README.md).
