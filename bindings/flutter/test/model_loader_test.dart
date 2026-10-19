import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';
import 'package:mlsr_flutter/src/model_resources.dart';
import 'package:mlsr/src/bindings.g.dart' as native;
import '../../dart/test/model_fixture.dart';

Matcher code(String value) =>
    throwsA(isA<SynthesisException>().having((e) => e.code, 'code', value));

class Resources implements ModelResources {
  final ModelFixture fixture;
  final Directory store;
  bool direct = true;
  int copies = 0, storeQueries = 0;
  int? failAt;
  String platform = 'macos-arm64';
  Resources(this.fixture, this.store);
  @override
  Future<String> platformKey() async => platform;
  @override
  Future<String?> locateDirectory(BundledModel model) async =>
      direct ? fixture.source.path : null;
  @override
  Future<String> storeDirectory() async {
    storeQueries++;
    return store.path;
  }

  @override
  Future<void> copyFile(
    BundledModel model,
    ModelAssetFile file,
    String staging,
  ) async {
    if (++copies == failAt) {
      throw const SynthesisException(
        'model_resource_error',
        'injected copy failure',
      );
    }
    final target = File('$staging/${file.path}');
    await target.parent.create(recursive: true);
    final input = await File('${fixture.source.path}/${file.path}').open();
    final output = await target.open(mode: FileMode.write);
    try {
      while (true) {
        final chunk = await input.read(65536);
        if (chunk.isEmpty) break;
        await output.writeFrom(chunk);
      }
    } finally {
      await input.close();
      await output.close();
    }
  }
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late ModelFixture fixture;
  late Directory store;
  late Resources resources;
  late BundledModel model;
  late int before;
  setUp(() async {
    before = native.mlsr_live_handles();
    fixture = await ModelFixture.create(
      fixtureDirectory: '../dart/test/fixtures',
    );
    store = await Directory.systemTemp.createTemp('mlsr-flutter-store-');
    resources = Resources(fixture, store);
    final files = <ModelAssetFile>[];
    await for (final entry in fixture.source.list(recursive: true)) {
      if (entry is File) {
        final bytes = await entry.readAsBytes();
        files.add(
          ModelAssetFile(
            path: entry.path.substring(fixture.source.path.length + 1),
            sha256: sha256.convert(bytes).toString(),
            sizeBytes: bytes.length,
          ),
        );
      }
    }
    files.sort((a, b) => a.path.compareTo(b.path));
    model = BundledModel(
      package: 'test_models',
      id: fixture.model.id,
      assetPath: 'models/test',
      catalogYaml: fixture.model.catalogYaml,
      platforms: ['macos-arm64'],
      files: files,
    );
  });
  tearDown(() async {
    expect(native.mlsr_live_handles(), before);
    await fixture.dispose();
    await store.delete(recursive: true);
  });
  test(
    'developer facade failure releases verified delivery ownership',
    () async {
      final progress = <ModelLoadStage>[];
      await expectLater(
        FlutterModelLoader(resources: resources).openTts(
          model: model,
          options: const TtsOptions(maxRequests: 0),
          onProgress: (value) => progress.add(value.stage),
        ),
        code('invalid_options'),
      );
      expect(progress, [
        ModelLoadStage.locating,
        ModelLoadStage.verifying,
        ModelLoadStage.loading,
      ]);
      expect(resources.copies, 0);
      expect(resources.storeQueries, 0);
      expect(native.mlsr_live_handles(), before);
    },
  );
  test('direct resources verify without copying or creating a store', () async {
    final progress = <ModelLoadStage>[];
    final handle = await FlutterModelLoader(
      resources: resources,
    ).loadModel(model: model, onProgress: (p) => progress.add(p.stage));
    await handle.dispose();
    expect(resources.copies, 0);
    expect(resources.storeQueries, 0);
    expect(progress, [ModelLoadStage.locating, ModelLoadStage.verifying]);
  });
  test(
    'fallback installs once, reopens, and leases cleanup across callers',
    () async {
      resources.direct = false;
      final loader = FlutterModelLoader(resources: resources);
      final progress = <ModelLoadProgress>[];
      final handles = await Future.wait([
        loader.loadModel(model: model, onProgress: progress.add),
        loader.loadModel(model: model),
      ]);
      expect(resources.copies, model.files.length);
      expect(
        progress.any(
          (p) =>
              p.stage == ModelLoadStage.waiting ||
              p.stage == ModelLoadStage.copying,
        ),
        isTrue,
      );
      await expectLater(
        ModelStore(store.path).remove(fixture.model),
        code('model_in_use'),
      );
      for (final handle in handles) {
        await handle.dispose();
      }
      final repeat = await loader.loadModel(model: model);
      await repeat.dispose();
      expect(resources.copies, model.files.length);
      await ModelStore(store.path).remove(fixture.model);
    },
  );
  test('copy failure cleans staging and leaves a retryable version', () async {
    resources.direct = false;
    resources.failAt = 3;
    final loader = FlutterModelLoader(resources: resources);
    await expectLater(
      loader.loadModel(model: model),
      code('model_resource_error'),
    );
    expect(await Directory('${store.path}/staging').list().toList(), isEmpty);
    await expectLater(
      ModelStore(store.path).open(fixture.model),
      code('model_not_installed'),
    );
    resources.failAt = null;
    final handle = await loader.loadModel(model: model);
    await handle.dispose();
  });
  test(
    'installedCopy uses native streaming even when directory is addressable',
    () async {
      final handle = await FlutterModelLoader(
        resources: resources,
      ).loadModel(model: model, delivery: ModelDelivery.installedCopy);
      expect(resources.copies, model.files.length);
      await handle.dispose();
    },
  );
  test(
    'corrupt direct resource fails without hidden materialization',
    () async {
      await File('${fixture.source.path}/tts.json').writeAsString('bad');
      await expectLater(
        FlutterModelLoader(resources: resources).loadModel(model: model),
        code('invalid_asset'),
      );
      expect(resources.copies, 0);
      expect(resources.storeQueries, 0);
    },
  );
  test('engine creation failure releases its source model lease', () async {
    await expectLater(
      FlutterModelLoader(resources: resources).open(model: model),
      code('runtime_error'),
    );
  });
  test('wrong platform and unsafe inventory fail before copying', () async {
    resources.platform = 'macos-x64';
    await expectLater(
      FlutterModelLoader(resources: resources).loadModel(model: model),
      code('model_platform_unsupported'),
    );
    resources.platform = 'macos-arm64';
    final invalid = BundledModel(
      package: model.package,
      id: model.id,
      assetPath: model.assetPath,
      catalogYaml: model.catalogYaml,
      platforms: model.platforms,
      files: [
        ModelAssetFile(path: '../escape', sha256: '0' * 64, sizeBytes: 1),
      ],
    );
    await expectLater(
      FlutterModelLoader(resources: resources).loadModel(model: invalid),
      code('invalid_model_package'),
    );
    expect(resources.copies, 0);
  });
  test('waiting for another installer has a bounded timeout', () async {
    resources.direct = false;
    final transaction = await ModelStore(store.path).begin(fixture.model);
    try {
      await expectLater(
        FlutterModelLoader(
          resources: resources,
          installWaitTimeout: Duration.zero,
        ).loadModel(model: model),
        code('model_store_busy'),
      );
    } finally {
      await transaction.dispose();
    }
  });
  test(
    'native channel uses logical keys and paths rather than payload buffers',
    () async {
      final calls = <MethodCall>[];
      final messenger =
          TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
      messenger.setMockMethodCallHandler(NativeModelResources.channel, (
        call,
      ) async {
        calls.add(call);
        return switch (call.method) {
          'platform' => 'macos-arm64',
          'locate' => '/app/models',
          'storeDirectory' => '/support/mlsr',
          _ => null,
        };
      });
      try {
        const nativeResources = NativeModelResources();
        expect(await nativeResources.platformKey(), 'macos-arm64');
        expect(await nativeResources.locateDirectory(model), '/app/models');
        expect(await nativeResources.storeDirectory(), '/support/mlsr');
        await nativeResources.copyFile(
          model,
          model.files.first,
          '/support/mlsr/staging/install-test',
        );
        expect(calls.last.arguments, {
          'package': model.package,
          'assetRoot': model.assetPath,
          'relative': model.files.first.path,
          'sizeBytes': model.files.first.sizeBytes,
          'staging': '/support/mlsr/staging/install-test',
        });
      } finally {
        messenger.setMockMethodCallHandler(NativeModelResources.channel, null);
      }
    },
  );
}
