import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:mlsr/mlsr.dart';

Future<void> main(List<String> args) async {
  final options = <String, String>{};
  var cancelFirst = false;
  for (var i = 0; i < args.length; i++) {
    if (args[i] == '--cancel-first') {
      cancelFirst = true;
      continue;
    }
    if (!args[i].startsWith('--') || i + 1 == args.length) {
      throw ArgumentError('Expected --name value');
    }
    options[args[i].substring(2)] = args[++i];
  }
  for (final required in ['manifest', 'bundle', 'output']) {
    if (!options.containsKey(required)) {
      throw ArgumentError('--$required is required');
    }
  }
  final file = File(options['output']!);
  await file.create(exclusive: true);
  final wav = await file.open(mode: FileMode.writeOnly);
  Engine? engine;
  try {
    await wav.writeFrom(Uint8List(56));
    engine = await Engine.load(
      manifest: options['manifest']!,
      bundle: options['bundle']!,
    );
    final text = options['text'] ?? 'Hello from MLSR.',
        language = options['language'] ?? 'en',
        voice = options['voice'] ?? 'M1';
    if (cancelFirst) {
      final job = await engine.start(
        text,
        language: language,
        voice: voice,
        options: const SynthesisOptions(queueFrames: 1, chunkFrames: 1),
      );
      final events = StreamIterator(job.events);
      await events.moveNext();
      job.cancel();
      if (!await job.wait(timeoutMs: 30000)) {
        throw StateError('Cancellation did not quiesce');
      }
      await events.cancel();
      await job.dispose();
    }
    final job = await engine.start(
      text,
      language: language,
      voice: voice,
      options: const SynthesisOptions(seed: 1729),
    );
    var frames = 0;
    await for (final event in job.events) {
      if (event.kind == EventKind.audio) {
        frames += event.frames;
        if (frames * 4 > 0xffffffff - 48) {
          throw StateError('Classic WAV size limit exceeded');
        }
        final bytes = ByteData(event.samples.length * 4);
        for (var i = 0; i < event.samples.length; i++) {
          bytes.setFloat32(i * 4, event.samples[i], Endian.little);
        }
        await wav.writeFrom(bytes.buffer.asUint8List());
      }
      if (event.terminal && event.kind != EventKind.completed) {
        throw event.error ?? StateError('Synthesis cancelled');
      }
    }
    final header = ByteData(56);
    final bytes = header.buffer.asUint8List();
    for (final item in {
      0: 'RIFF',
      8: 'WAVEfmt ',
      36: 'fact',
      48: 'data',
    }.entries) {
      bytes.setAll(item.key, ascii.encode(item.value));
    }
    for (final item in {
      4: 48 + frames * 4,
      16: 16,
      24: 44100,
      28: 176400,
      40: 4,
      44: frames,
      52: frames * 4,
    }.entries) {
      header.setUint32(item.key, item.value, Endian.little);
    }
    for (final item in {20: 3, 22: 1, 32: 4, 34: 32}.entries) {
      header.setUint16(item.key, item.value, Endian.little);
    }
    await wav.setPosition(0);
    await wav.writeFrom(bytes);
    await wav.close();
    stdout.writeln('Wrote $frames frames to ${file.path}');
  } catch (_) {
    await wav.close();
    await file.delete();
    rethrow;
  } finally {
    await engine?.dispose();
  }
}
