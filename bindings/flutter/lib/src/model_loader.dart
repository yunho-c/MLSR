import 'package:mlsr/mlsr.dart';
import 'model_resources.dart';

enum ModelLoadStage { locating, waiting, copying, verifying, loading }

/// Model preparation status; byte counts cover completed files during copying.
class ModelLoadProgress {
  final ModelLoadStage stage;
  final int copiedBytes, totalBytes;
  const ModelLoadProgress(
    this.stage, {
    this.copiedBytes = 0,
    this.totalBytes = 0,
  });
}

/// Automatic delivery uses immutable app resources directly when addressable.
/// installedCopy is useful for exercising materialization or choosing a stable
/// application-support location even on platforms with filesystem resources.
enum ModelDelivery { automatic, installedCopy }

/// Convenience entry point for generated AppModels selections.
abstract final class FlutterMlsr {
  static Future<Engine> open({
    required BundledModel model,
    ComputePolicy computePolicy = ComputePolicy.cpuAndNeuralEngine,
    bool reuseOutputBuffers = true,
    ModelDelivery delivery = ModelDelivery.automatic,
    void Function(ModelLoadProgress)? onProgress,
  }) => const FlutterModelLoader().open(
    model: model,
    computePolicy: computePolicy,
    reuseOutputBuffers: reuseOutputBuffers,
    delivery: delivery,
    onProgress: onProgress,
  );
}

/// Resolves platform resources and delegates identity, publication and ownership
/// to Rust. No asset payload travels through a Dart platform-channel buffer.
class FlutterModelLoader {
  final ModelResources resources;
  final Duration installWaitTimeout;
  const FlutterModelLoader({
    this.resources = const NativeModelResources(),
    this.installWaitTimeout = const Duration(seconds: 30),
  });

  Future<Engine> open({
    required BundledModel model,
    ComputePolicy computePolicy = ComputePolicy.cpuAndNeuralEngine,
    bool reuseOutputBuffers = true,
    ModelDelivery delivery = ModelDelivery.automatic,
    void Function(ModelLoadProgress)? onProgress,
  }) async {
    final handle = await loadModel(
      model: model,
      delivery: delivery,
      onProgress: onProgress,
    );
    try {
      onProgress?.call(const ModelLoadProgress(ModelLoadStage.loading));
      return await Engine.fromModel(
        handle,
        computePolicy: computePolicy,
        reuseOutputBuffers: reuseOutputBuffers,
      );
    } finally {
      await handle.dispose();
    }
  }

  /// Obtain a verified model without creating an inference engine. The returned
  /// model belongs to the caller. Corruption is an error, never an implicit copy
  /// or download that hides a damaged release.
  Future<ModelHandle> loadModel({
    required BundledModel model,
    ModelDelivery delivery = ModelDelivery.automatic,
    void Function(ModelLoadProgress)? onProgress,
  }) async {
    _validateInventory(model);
    onProgress?.call(const ModelLoadProgress(ModelLoadStage.locating));
    final descriptor = model.descriptorFor(await resources.platformKey());
    if (delivery == ModelDelivery.automatic) {
      final directory = await resources.locateDirectory(model);
      if (directory != null) {
        onProgress?.call(const ModelLoadProgress(ModelLoadStage.verifying));
        return ModelHandle.openDirectory(
          model: descriptor,
          directory: directory,
        );
      }
    }
    final store = ModelStore(await resources.storeDirectory());
    final wait = Stopwatch()..start();
    while (true) {
      try {
        return await store.open(descriptor);
      } on SynthesisException catch (error) {
        if (error.code != 'model_not_installed' &&
            error.code != 'model_store_busy') {
          rethrow;
        }
      }
      ModelInstallation? transaction;
      try {
        transaction = await store.begin(descriptor);
      } on SynthesisException catch (error) {
        if (error.code != 'model_store_busy' &&
            error.code != 'model_already_installed') {
          rethrow;
        }
        if (wait.elapsed >= installWaitTimeout) {
          throw const SynthesisException(
            'model_store_busy',
            'Another process is still preparing this model. Retry shortly.',
          );
        }
        onProgress?.call(const ModelLoadProgress(ModelLoadStage.waiting));
        await Future<void>.delayed(const Duration(milliseconds: 50));
        continue;
      }
      try {
        final total = model.files.fold<int>(
          0,
          (sum, file) => sum + file.sizeBytes,
        );
        var copied = 0;
        onProgress?.call(
          ModelLoadProgress(ModelLoadStage.copying, totalBytes: total),
        );
        for (final file in model.files) {
          await resources.copyFile(model, file, transaction.directory);
          copied += file.sizeBytes;
          onProgress?.call(
            ModelLoadProgress(
              ModelLoadStage.copying,
              copiedBytes: copied,
              totalBytes: total,
            ),
          );
        }
        onProgress?.call(const ModelLoadProgress(ModelLoadStage.verifying));
        return await transaction.commit();
      } finally {
        await transaction.dispose();
      }
    }
  }
}

bool _relative(String path) =>
    path.isNotEmpty &&
    path.length <= 4096 &&
    !path.contains(RegExp(r'[\\:\x00]')) &&
    path
        .split('/')
        .every((part) => part.isNotEmpty && part != '.' && part != '..');

void _validateInventory(BundledModel model) {
  if (!RegExp(r'^[a-z][a-z0-9_]{2,63}$').hasMatch(model.package) ||
      !_relative(model.assetPath) ||
      model.files.isEmpty ||
      model.files.length > 2561) {
    throw const SynthesisException(
      'invalid_model_package',
      'Invalid package identity or inventory',
    );
  }
  final seen = <String>{};
  for (final file in model.files) {
    if (!_relative(file.path) ||
        !seen.add(file.path) ||
        file.sizeBytes <= 0 ||
        !RegExp(r'^[0-9a-f]{64}$').hasMatch(file.sha256)) {
      throw const SynthesisException(
        'invalid_model_package',
        'Invalid or duplicate model asset',
      );
    }
  }
  if (!seen.contains('mlsr-bundle.yaml')) {
    throw const SynthesisException(
      'invalid_model_package',
      'Model manifest is absent from inventory',
    );
  }
}
