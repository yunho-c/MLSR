import 'package:flutter/services.dart';
import 'package:mlsr/mlsr.dart';

/// Platform boundary for model delivery. Implementations must stream files in
/// bounded memory, close writers before completing, and reject unsafe paths.
abstract interface class ModelResources {
  Future<String> platformKey();
  Future<String?> locateDirectory(BundledModel model);
  Future<String> storeDirectory();
  Future<void> copyFile(
    BundledModel model,
    ModelAssetFile file,
    String staging,
  );
}

class NativeModelResources implements ModelResources {
  const NativeModelResources();
  static const channel = MethodChannel('dev.mlsr.flutter/models');

  Future<T?> _call<T>(String method, [Map<String, Object?>? args]) async {
    try {
      return await channel.invokeMethod<T>(method, args);
    } on MissingPluginException {
      throw const SynthesisException(
        'model_platform_unsupported',
        'MLSR model delivery is currently supported on macOS.',
      );
    } on PlatformException catch (error) {
      throw SynthesisException(
        error.code,
        error.message ?? 'Model resource operation failed',
      );
    }
  }

  @override
  Future<String> platformKey() async =>
      await _call<String>('platform') ??
      (throw StateError('Missing platform key'));

  @override
  Future<String?> locateDirectory(BundledModel model) => _call<String>(
    'locate',
    {'package': model.package, 'assetRoot': model.assetPath},
  );

  @override
  Future<String> storeDirectory() async =>
      await _call<String>('storeDirectory') ??
      (throw StateError('Missing application-support directory'));

  @override
  Future<void> copyFile(
    BundledModel model,
    ModelAssetFile file,
    String staging,
  ) async {
    await _call<void>('copyFile', {
      'package': model.package,
      'assetRoot': model.assetPath,
      'relative': file.path,
      'sizeBytes': file.sizeBytes,
      'staging': staging,
    });
  }
}
