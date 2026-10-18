part of 'synthesis.dart';

/// Exact model release in a trusted application catalog. Generated model asset
/// packages supply these constants; Flutter determines the current platform.
class ModelDescriptor {
  final String catalogYaml, id, platform;
  const ModelDescriptor({
    required this.catalogYaml,
    required this.id,
    required this.platform,
  });
  Map<String, Object?> get _values => {
    'catalog': catalogYaml,
    'id': id,
    'platform': platform,
  };
}

Future<Object?> _modelOperation(
  String operation,
  Map<String, Object?> args,
) async {
  checkModelApi(); // Keep the native asset loaded in the owning isolate too.
  final adapter = await _Adapter.spawn();
  try {
    return await adapter.call(operation, args);
  } finally {
    adapter.close();
  }
}

/// Verified model with an optional persistent-store lease. Always await dispose.
/// Engines retain their own lease and stay usable after the model is disposed.
class ModelHandle {
  final int _handle;
  bool _disposed = false;
  int _users = 0;
  Completer<void>? _idle;
  Future<void>? _disposal;
  ModelHandle._(this._handle);

  /// Verify an immutable app-bundled directory without materializing a copy.
  /// The app owns these files and must retain them for every derived engine.
  static Future<ModelHandle> openDirectory({
    required ModelDescriptor model,
    required String directory,
  }) async => ModelHandle._(
    await _modelOperation('modelOpen', {...model._values, 'root': directory})
        as int,
  );

  Future<T> _use<T>(Future<T> Function(int) action) async {
    if (_disposed) {
      throw const SynthesisException('model_closed', 'Model is disposed');
    }
    _users++;
    try {
      return await action(_handle);
    } finally {
      if (--_users == 0) {
        _idle?.complete();
        _idle = null;
      }
    }
  }

  /// Release this lease after any in-flight engine creation finishes. Idempotent.
  Future<void> dispose() => _disposal ??= _dispose();
  Future<void> _dispose() async {
    _disposed = true;
    if (_users != 0) await (_idle ??= Completer<void>()).future;
    await _modelOperation('releaseModel', {'model': _handle});
  }
}

/// Store at a host-selected application-support directory. Flutter supplies that
/// location; the portable Dart/Rust APIs never assume a HOME or app bundle path.
class ModelStore {
  final String directory;
  const ModelStore(this.directory);

  Future<ModelHandle> open(ModelDescriptor model) async => ModelHandle._(
    await _modelOperation('storeOpen', {...model._values, 'root': directory})
        as int,
  );

  Future<ModelInstallation> begin(ModelDescriptor model) async {
    final result =
        await _modelOperation('installBegin', {
              ...model._values,
              'root': directory,
            })
            as List;
    return ModelInstallation._(result[0] as int, result[1] as String);
  }

  /// Remove only this version, failing with model_in_use if an engine/model
  /// still leases it. Absent versions succeed. Does not touch bundled assets.
  Future<void> remove(ModelDescriptor model) async {
    await _modelOperation('storeRemove', {...model._values, 'root': directory});
  }
}

/// Unpublished staging transaction. Stream regular files into [directory],
/// close all writers, then commit. Dispose aborts an unfinished transaction.
class ModelInstallation {
  final int _handle;
  final String directory;
  bool _finished = false;
  Future<ModelHandle>? _completion;
  Future<void>? _disposal;
  ModelInstallation._(this._handle, this.directory);

  /// Verify and atomically publish. A failed commit aborts and cleans staging.
  /// Cannot be called twice. Never write to directory after calling this method.
  Future<ModelHandle> commit() {
    if (_finished) {
      return Future.error(
        const SynthesisException(
          'model_store_invalid_state',
          'Installation has already finished',
        ),
      );
    }
    _finished = true;
    return _completion = _commit();
  }

  Future<ModelHandle> _commit() async {
    try {
      return ModelHandle._(
        await _modelOperation('installCommit', {'installation': _handle})
            as int,
      );
    } finally {
      await _modelOperation('releaseInstallation', {'installation': _handle});
    }
  }

  /// Abort before commit; otherwise wait for its completion. Idempotent. The
  /// caller of commit owns its returned model and receives any commit failure.
  Future<void> dispose() => _disposal ??= _dispose();
  Future<void> _dispose() async {
    if (_finished) {
      await _completion?.then<void>(
        (_) {},
        onError: (Object _, StackTrace _) {},
      );
      return;
    }
    _finished = true;
    await _modelOperation('releaseInstallation', {'installation': _handle});
  }
}
