/// Dart bindings to the MLSR Rust library.
library;

import 'package:ffi/ffi.dart';

import 'src/bindings.g.dart' as native;
export 'src/synthesis.dart';

/// C ABI version supported by this package.
const supportedAbiVersion = 1;

/// ABI version exposed by the bundled native library.
int get nativeAbiVersion => native.mlsr_abi_version();

/// Version of the Rust implementation, after checking ABI compatibility.
String get nativeVersion {
  final actual = nativeAbiVersion;
  if (actual != supportedAbiVersion) {
    throw StateError(
      'Unsupported MLSR ABI $actual; require $supportedAbiVersion.',
    );
  }
  return native.mlsr_version().cast<Utf8>().toDartString();
}
