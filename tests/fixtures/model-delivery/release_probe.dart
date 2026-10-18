// Copied into the separate qualification consumer, never the ordinary app entry.
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
import 'package:flutter/widgets.dart';
import 'package:mlsr_app_models/app_models.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';
// Diagnostic platform/resource and handle observations, intentionally test-only.
// ignore: implementation_imports
import 'package:mlsr_flutter/src/model_resources.dart';
// ignore: implementation_imports
import 'package:mlsr/src/bindings.g.dart' as native;
import 'package:mlsr_speak/main.dart' as app;
import 'package:mlsr_speak/speech_controller.dart';

void require(bool condition, String message) {
  if (!condition) throw StateError(message);
}

Future<void> waitUntil(bool Function() condition) async {
  final deadline = DateTime.now().add(const Duration(seconds: 60));
  while (!condition()) {
    if (DateTime.now().isAfter(deadline)) {
      throw StateError('UI state timed out');
    }
    await Future<void>.delayed(const Duration(milliseconds: 20));
  }
}

String pcmHash(Uint8List bytes) {
  final view = ByteData.sublistView(bytes);
  for (var offset = 12; offset + 8 <= bytes.length;) {
    final length = view.getUint32(offset + 4, Endian.little);
    require(offset + 8 + length <= bytes.length, 'truncated WAV');
    if (String.fromCharCodes(bytes.sublist(offset, offset + 4)) == 'data') {
      return sha256
          .convert(
            Uint8List.sublistView(bytes, offset + 8, offset + 8 + length),
          )
          .toString();
    }
    offset += 8 + length + (length & 1);
  }
  throw StateError('missing WAV data');
}

class ProbeResources implements ModelResources {
  final nativeResources = const NativeModelResources();
  int copies = 0, bytes = 0;
  @override
  Future<String> platformKey() => nativeResources.platformKey();
  @override
  Future<String?> locateDirectory(BundledModel model) =>
      nativeResources.locateDirectory(model);
  @override
  Future<String> storeDirectory() => nativeResources.storeDirectory();
  @override
  Future<void> copyFile(
    BundledModel model,
    ModelAssetFile file,
    String staging,
  ) async {
    await nativeResources.copyFile(model, file, staging);
    copies++;
    bytes += file.sizeBytes;
  }
}

Future<Map<String, int>> inventory(String path) async {
  final root = Directory('$path/models');
  var files = 0, bytes = 0;
  if (await root.exists()) {
    await for (final entry in root.list(recursive: true, followLinks: false)) {
      if (entry is File) {
        files++;
        bytes += await entry.length();
      }
    }
  }
  return {'files': files, 'bytes': bytes};
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final speech = DemoController(muted: true);
  runApp(app.SpeakApp(controller: speech));
  final report = <String, Object?>{'passed': false};
  Tts? tts;
  Directory? audioDirectory;
  try {
    final phase = Platform.environment['MLSR_DELIVERY_PHASE']!;
    final installed = phase.startsWith('installed');
    report['phase'] = phase;
    report['handles_before'] = native.mlsr_live_handles();
    require(native.mlsr_live_handles() == 0, 'preexisting native handles');
    final resources = ProbeResources();
    final descriptor = AppModels.supertonic3.descriptorFor(
      await resources.platformKey(),
    );
    final root = await resources.locateDirectory(AppModels.supertonic3);
    require(root != null, 'bundled directory was not found');
    report['bundled_directory'] = root;
    report['executable'] = Platform.resolvedExecutable;
    final storeRoot = await resources.storeDirectory();
    report['store_directory'] = storeRoot;
    report['store_before'] = await inventory(storeRoot);

    final forbidden =
        jsonDecode(Platform.environment['MLSR_FORBIDDEN_FILES']!) as List;
    final denials = <Map<String, Object?>>[];
    for (final path in forbidden.cast<String>()) {
      try {
        final file = await File(path).open();
        await file.close();
        throw StateError('developer file remained readable: $path');
      } on FileSystemException catch (error) {
        require(
          [1, 13].contains(error.osError?.errorCode),
          'file failed for a reason other than sandbox denial: $error',
        );
        denials.add({'path': path, 'errno': error.osError?.errorCode});
      }
    }
    report['developer_file_denials'] = denials;
    try {
      final socket = await Socket.connect(
        InternetAddress.loopbackIPv4,
        int.parse(Platform.environment['MLSR_DENIED_NETWORK_PORT']!),
        timeout: const Duration(seconds: 2),
      );
      socket.destroy();
      throw StateError('network connection was permitted');
    } on SocketException catch (error) {
      require(
        [1, 13].contains(error.osError?.errorCode),
        'network was not explicitly denied: $error',
      );
      report['network_denied_errno'] = error.osError?.errorCode;
    }

    final parent = Directory(storeRoot).parent;
    await parent.create(recursive: true);
    audioDirectory = await parent.createTemp('delivery-audio-');
    report['audio_evidence_directory'] = audioDirectory.path;

    final controllerStart = Stopwatch()..start();
    await speech.load();
    report['controller_load_ms'] = controllerStart.elapsedMicroseconds / 1000;
    require(speech.ready, 'release controller loading failed: ${speech.error}');
    final spoken = speech.speak('Hello from MLSR.', 'en', 'M1')!;
    require(
      (await spoken.done).kind == OutcomeKind.completed,
      'speech did not complete',
    );
    await waitUntil(() => speech.canExport || speech.error != null);
    require(
      speech.canExport && speech.error == null,
      'release controller failed: ${speech.error}',
    );
    final controllerPath = '${audioDirectory.path}/controller-export.wav';
    await speech.exportTo(controllerPath);
    final controllerBytes = await File(controllerPath).readAsBytes();
    report['controller_wav_sha256'] = sha256
        .convert(controllerBytes)
        .toString();
    report['controller_pcm_sha256'] = pcmHash(controllerBytes);
    await speech.close();
    require(native.mlsr_live_handles() == 0, 'controller handles leaked');
    report['controller_handles_released'] = true;

    final start = Stopwatch()..start();
    stdout.writeln('MLSR_DELIVERY_STAGE=resolve-model');
    final model = await FlutterModelLoader(resources: resources).loadModel(
      model: AppModels.supertonic3,
      delivery: installed
          ? ModelDelivery.installedCopy
          : ModelDelivery.automatic,
    );
    report['model_resolution_ms'] = start.elapsedMicroseconds / 1000;
    start.reset();
    try {
      tts = await Tts.fromModel(model);
    } finally {
      await model.dispose();
    }
    report['engine_load_ms'] = start.elapsedMicroseconds / 1000;
    stdout.writeln('MLSR_DELIVERY_STAGE=engine-loaded');
    report['copied_files'] = resources.copies;
    report['copied_bytes'] = resources.bytes;
    report['store_loaded'] = await inventory(storeRoot);
    if (installed) {
      try {
        await ModelStore(storeRoot).remove(descriptor);
        throw StateError('active engine did not protect its files');
      } on SynthesisException catch (error) {
        require(error.code == 'model_in_use', 'unexpected lease error: $error');
        report['engine_lease_blocked_removal'] = true;
      }
    }
    final outputs = <String, Object?>{};
    for (final test in [
      ('hello', 'Hello from MLSR.', 'en', 'M1'),
      ('korean', '안녕하세요. 반갑습니다.', 'ko', 'F1'),
    ]) {
      final options = RequestOptions(
        locale: test.$3,
        voice: 'supertonic3:${test.$4}',
      );
      final rendered = await tts.renderToFile(
        test.$2,
        '${audioDirectory.path}/${test.$1}-render.wav',
        options: options,
      );
      final renderedBytes = await File(rendered.path).readAsBytes();
      final request = tts.speak(
        test.$2,
        request: options,
        options: const SpeechOptions(volume: 0),
      );
      final clock = Stopwatch()..start();
      double? firstBuffered, firstPosition;
      final observer = request.states.listen((state) {
        if (state.bufferedUntilFrame > 0) {
          firstBuffered ??= clock.elapsedMicroseconds / 1000;
        }
        if (state.positionFrame > 0) {
          firstPosition ??= clock.elapsedMicroseconds / 1000;
        }
      });
      try {
        require(
          (await request.done).kind == OutcomeKind.completed,
          'speech failed',
        );
        final state = await request.snapshot();
        require(
          state.playback == PlaybackPhase.ended &&
              state.positionFrame == state.durationFrames &&
              state.durationFrames == rendered.frames,
          'playback did not drain',
        );
        final exported = '${audioDirectory.path}/${test.$1}-export.wav';
        await tts.writeSpeech(request, exported);
        final bytes = await File(exported).readAsBytes();
        require(
          pcmHash(bytes) == pcmHash(renderedBytes),
          'retained speech and render PCM differ',
        );
        try {
          await tts.writeSpeech(request, exported);
          throw StateError('export overwrote a file');
        } on SynthesisException catch (error) {
          require(error.code == 'io_error', 'unexpected export error: $error');
        }
        require(
          sha256.convert(await File(exported).readAsBytes()).toString() ==
              sha256.convert(bytes).toString(),
          'existing WAV changed',
        );
        outputs[test.$1] = {
          'frames': rendered.frames,
          'wav_sha256': sha256.convert(bytes).toString(),
          'pcm_sha256': pcmHash(bytes),
          'render_pcm_sha256': pcmHash(renderedBytes),
          'first_buffered_observed_ms': firstBuffered,
          'first_position_observed_ms': firstPosition,
          'playback': state.playback.name,
        };
      } finally {
        await observer.cancel();
        await request.dispose();
      }
    }
    report['outputs'] = outputs;

    final request = tts.speak(
      List.filled(12, app.english).join('\n'),
      options: const SpeechOptions(volume: 0),
    );
    final ready = Completer<void>();
    final observer = request.states.listen((state) {
      if ((state.bufferedUntilFrame > 0 || state.completion != null) &&
          !ready.isCompleted) {
        ready.complete();
      }
    });
    await ready.future.timeout(const Duration(seconds: 60));
    final cancellation = Stopwatch()..start();
    request.cancel();
    require(
      (await request.done).reason == CancelReason.requested,
      'cancellation failed',
    );
    report['cancel_to_quiescent_observed_ms'] =
        cancellation.elapsedMicroseconds / 1000;
    require(
      (await request.snapshot()).playback == PlaybackPhase.stopped,
      'playback did not stop',
    );
    await observer.cancel();
    await request.dispose();
    final retained = (await tts.snapshot()).retention;
    require(
      retained.usedBytes == 0 && retained.liveStores == 0,
      'retained audio leaked',
    );
    report['retention_peak_bytes'] = retained.peakBytes;
    report['retention_after_bytes'] = retained.usedBytes;
    await tts.dispose();
    tts = null;

    if (phase == 'installed_repeat') {
      await ModelStore(storeRoot).remove(descriptor);
      try {
        final handle = await ModelStore(storeRoot).open(descriptor);
        await handle.dispose();
        throw StateError('removed version remained installed');
      } on SynthesisException catch (error) {
        require(
          error.code == 'model_not_installed',
          'unexpected cleanup failure',
        );
      }
      report['removed_after_shutdown'] = true;
    }
    report['store_after'] = await inventory(storeRoot);
    require(native.mlsr_live_handles() == 0, 'native handles leaked');
    report['handles_after'] = native.mlsr_live_handles();
    report['passed'] = true;
  } catch (error, stack) {
    report['failure'] = error.toString();
    report['stack'] = stack.toString();
  } finally {
    await tts?.dispose();
    await speech.close();
    speech.dispose();
    // User-owned exports remain as qualification evidence.
  }
  stdout.writeln('MLSR_DELIVERY_RESULT=${jsonEncode(report)}');
  await stdout.flush();
  exit(report['passed'] == true ? 0 : 1);
}
