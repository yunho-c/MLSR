// Copied into the separate qualification consumer, never the ordinary app entry.
import 'dart:async';
import 'dart:convert';
import 'dart:io';
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
import 'package:mlsr_speak/pcm_player.dart';
import 'package:mlsr_speak/speech_controller.dart';
import 'package:mlsr_speak/wav_sink.dart';

void require(bool condition, String message) {
  if (!condition) throw StateError(message);
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
  final speech = SpeechController(muted: true);
  runApp(app.SpeakApp(controller: speech));
  final report = <String, Object?>{'passed': false};
  Engine? engine;
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

    // The ordinary controller owns the native player's initial epoch sequence.
    // Exercise it before the direct probe advances that same platform player.
    final controllerStart = Stopwatch()..start();
    await speech.load();
    report['controller_load_ms'] = controllerStart.elapsedMicroseconds / 1000;
    require(speech.ready, 'release controller loading failed: ${speech.error}');
    await speech.speak('Hello from MLSR.', 'en', 'M1');
    require(
      speech.error == null && speech.wav != null,
      'release controller failed: ${speech.error}',
    );
    final controllerWav = speech.wav!;
    report['controller_wav_sha256'] = sha256
        .convert(await controllerWav.readAsBytes())
        .toString();
    await WavSink.export(
      controllerWav,
      '${audioDirectory.path}/controller-export.wav',
    );
    await speech.close();
    require(!await controllerWav.exists(), 'controller temporary WAV leaked');
    report['controller_temporary_wav_removed'] = true;

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
      engine = await Engine.fromModel(model);
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
    final player = PcmPlayer();
    final outputs = <String, Object?>{};
    var epoch = (await player.state())['epoch'] as int;
    epoch++;
    for (final test in [
      ('hello', 'Hello from MLSR.', 'en', 'M1'),
      ('korean', '안녕하세요. 반갑습니다.', 'ko', 'F1'),
    ]) {
      await player.begin(epoch, muted: true);
      final sink = await WavSink.create(audioDirectory, epoch);
      final job = await engine.start(
        test.$2,
        language: test.$3,
        voice: test.$4,
      );
      var frames = 0, terminals = 0;
      final clock = Stopwatch()..start();
      double? firstAudio;
      await for (final event in job.events) {
        if (event.kind == EventKind.audio) {
          firstAudio ??= clock.elapsedMicroseconds / 1000;
          require(
            event.startFrame == frames &&
                event.samples.every((v) => v.isFinite),
            'invalid PCM',
          );
          await sink.add(event.samples);
          await player.enqueue(epoch, event.samples);
          frames += event.frames;
        }
        if (event.terminal) {
          terminals++;
          require(event.kind == EventKind.completed, 'synthesis failed');
        }
      }
      require(frames > 44100 && terminals == 1, 'incomplete speech');
      await player.finish(epoch);
      final wav = await sink.finish();
      final bytes = await wav.readAsBytes();
      final exported = '${audioDirectory.path}/${test.$1}-export.wav';
      await WavSink.export(wav, exported);
      try {
        await WavSink.export(wav, exported);
        throw StateError('export overwrote a file');
      } on FileSystemException {
        /* expected create-exclusive behavior */
      }
      final state = await player.state();
      require(
        state['active'] == false &&
            state['queuedFrames'] == 0 &&
            (state['peakQueuedFrames'] as int) <= 22050,
        'playback did not drain within its bound',
      );
      outputs[test.$1] = {
        'frames': frames,
        'wav_sha256': sha256.convert(bytes).toString(),
        'first_audio_ms': firstAudio,
        'playback': state,
      };
      epoch++;
    }
    report['outputs'] = outputs;

    await player.begin(epoch, muted: true);
    final job = await engine.start(List.filled(12, app.english).join('\n'));
    final events = StreamIterator(job.events);
    while (await events.moveNext()) {
      if (events.current.kind == EventKind.audio) {
        await player.enqueue(epoch, events.current.samples);
        break;
      }
    }
    final cancellation = Stopwatch()..start();
    await player.stop(epoch);
    final stopped = await player.state();
    require(
      stopped['active'] == false && stopped['queuedFrames'] == 0,
      'stop did not flush playback',
    );
    report['stop_to_playback_flush_ms'] =
        cancellation.elapsedMicroseconds / 1000;
    job.cancel();
    require(await job.wait(timeoutMs: 30000), 'cancellation did not quiesce');
    await events.cancel();
    report['stop_to_quiescent_ms'] = cancellation.elapsedMicroseconds / 1000;
    await engine.dispose();
    engine = null;

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
    await engine?.dispose();
    await speech.close();
    speech.dispose();
    // Keep probe WAVs as qualification evidence. SpeechController.close above
    // still verifies cleanup of the ordinary application's temporary audio.
  }
  stdout.writeln('MLSR_DELIVERY_RESULT=${jsonEncode(report)}');
  await stdout.flush();
  exit(report['passed'] == true ? 0 : 1);
}
