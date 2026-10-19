import 'dart:io';
import 'package:mlsr/mlsr.dart';
import 'package:mlsr/src/bindings.g.dart' as native;
import 'package:test/test.dart';
import 'model_fixture.dart';

Matcher _code(String code) =>
    throwsA(isA<SynthesisException>().having((e) => e.code, 'code', code));

void modelStoreTests() {
  test(
    'store ownership, competing installation, abort and exact identity through FFI',
    () async {
      final before = native.mlsr_live_handles();
      final fixture = await ModelFixture.create();
      final root = await Directory.systemTemp.createTemp('mlsr-store-test-');
      final store = ModelStore(root.path);
      try {
        await expectLater(
          store.open(fixture.model),
          _code('model_not_installed'),
        );
        final transaction = await store.begin(fixture.model);
        await expectLater(
          store.begin(fixture.model),
          _code('model_store_busy'),
        );
        await fixture.fill(transaction);
        await expectLater(
          store.open(fixture.model),
          _code('model_not_installed'),
        );
        final model = await transaction.commit();
        await transaction.dispose();
        expect(Directory(transaction.directory).existsSync(), isFalse);
        await expectLater(
          transaction.commit(),
          _code('model_store_invalid_state'),
        );
        await expectLater(store.remove(fixture.model), _code('model_in_use'));
        final reopened = await store.open(fixture.model);
        await model.dispose();
        await model.dispose();
        await expectLater(Engine.fromModel(model), _code('model_closed'));
        await expectLater(store.remove(fixture.model), _code('model_in_use'));
        await reopened.dispose();
        await store.remove(fixture.model);
        await store.remove(fixture.model);
        final aborted = await store.begin(fixture.model);
        await File(
          '${aborted.directory}/partial',
        ).writeAsString('not published');
        await aborted.dispose();
        await aborted.dispose();
        expect(Directory(aborted.directory).existsSync(), isFalse);
        final invalid = await store.begin(fixture.model);
        await fixture.fill(invalid);
        await File('${invalid.directory}/tts.json').writeAsString('corrupted');
        await expectLater(invalid.commit(), _code('invalid_asset'));
        await invalid.dispose();
        expect(Directory(invalid.directory).existsSync(), isFalse);
        await expectLater(
          store.open(fixture.model),
          _code('model_not_installed'),
        );
        final bundled = await ModelHandle.openDirectory(
          model: fixture.model,
          directory: fixture.source.path,
        );
        await bundled.dispose();
      } finally {
        await root.delete(recursive: true);
        await fixture.dispose();
      }
      expect(native.mlsr_live_handles(), before);
    },
  );

  test(
    'real engine retains store lease after caller model release',
    () async {
      final before = native.mlsr_live_handles();
      final root = await Directory.systemTemp.createTemp('mlsr-real-store-');
      final store = ModelStore(root.path);
      final descriptor = ModelDescriptor(
        catalogYaml: await File(
          Platform.environment['MLSR_TEST_DELIVERY_CATALOG'] ??
              'test/fixtures/catalog.yaml',
        ).readAsString(),
        id:
            Platform.environment['MLSR_TEST_DELIVERY_ID'] ??
            'supertonic3-coreml-int4-v1',
        platform:
            Platform.environment['MLSR_TEST_DELIVERY_PLATFORM'] ??
            'macos-arm64',
      );
      final transaction = await store.begin(descriptor);
      Engine? engine;
      try {
        final source = Directory(
          Platform.environment['MLSR_TEST_DELIVERY_BUNDLE']!,
        );
        await for (final entry in source.list(recursive: true)) {
          if (entry is File) {
            final relative = entry.path.substring(source.path.length + 1);
            final target = File('${transaction.directory}/$relative');
            await target.parent.create(recursive: true);
            await entry.copy(target.path);
          }
        }
        final model = await transaction.commit();
        // Queue creation and disposal together: disposal must wait until Rust has
        // retained its own lease, even before the adapter isolate starts the call.
        final opening = Engine.fromModel(
          model,
          computePolicy: Platform.environment['MLSR_TEST_POLICY'] == 'qnn_htp'
              ? ComputePolicy.qnnHtp
              : ComputePolicy.cpuAndNeuralEngine,
        );
        final disposing = model.dispose();
        engine = await opening;
        await disposing;
        await expectLater(store.remove(descriptor), _code('model_in_use'));
        final job = await engine.start('Hello from MLSR.');
        var frames = 0;
        await for (final event in job.events) {
          if (event.kind == EventKind.audio) frames += event.frames;
          if (event.terminal) expect(event.kind, EventKind.completed);
        }
        expect(frames, greaterThan(44100));
        await engine.dispose();
        await store.remove(descriptor);
        await expectLater(store.open(descriptor), _code('model_not_installed'));
      } finally {
        await engine?.dispose();
        await transaction.dispose();
        await root.delete(recursive: true);
      }
      expect(native.mlsr_live_handles(), before);
    },
    skip: Platform.environment['MLSR_TEST_DELIVERY_BUNDLE'] == null,
    timeout: const Timeout(Duration(minutes: 3)),
  );
}
