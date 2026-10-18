# Core ML runtime fixture

`affine.mlmodel` is a tiny, MLSR-authored model (Apache-2.0) with Float32 input
`x` and outputs `y` and `z`, all shape `[1, 2, 3]`. Its operations are
`y = 2*x + 1` and `z = 3*x - 1`.
It contains no downloaded or trained weights. `generate.py` records its complete
construction using coremltools 9.0; regeneration is an explicit development step.

Apple tests compile the checked-in model with the installed Xcode
`coremlcompiler`, execute it using CPU-only Core ML, and remove their temporary
compiled directories. No Python/Core ML Tools installation is needed to run
these tests. Their simple arithmetic has independently known expected outputs.
Supertonic's real multi-stage replay is a separate opt-in qualification.
