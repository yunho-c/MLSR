# Core ML runtime fixture

`affine.mlmodel` is a tiny, MLSR-authored model (Apache-2.0) with one Float32
input `x` and output `y`, both shape `[1, 2, 3]`. Its operation is `y = 2*x + 1`.
It contains no downloaded or trained weights. `generate.py` records its complete
construction using coremltools 9.0; regeneration is an explicit development step.

Apple tests compile the checked-in model with the installed Xcode
`coremlcompiler`, execute it using CPU-only Core ML, and remove their temporary
compiled directories. No Python/Core ML Tools installation is needed to run
these tests. Their simple arithmetic has independently known expected outputs.
Supertonic's real multi-stage replay is a separate opt-in qualification.
