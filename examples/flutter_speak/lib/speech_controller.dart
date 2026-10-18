import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:mlsr/mlsr.dart';
import 'pcm_player.dart';
import 'wav_sink.dart';

/// Owns one persistent engine and one generation. Epoch checks reject events
/// from stopped/replaced requests at every await boundary before publication.
class SpeechController extends ChangeNotifier {
  final PcmPlayer player;
  final bool muted;
  Engine? _engine;
  Job? _job;
  Directory? _temporary;
  Future<void>? _work;
  Future<void>? _closing;
  Future<void>? _loading;
  bool _closed = false;
  bool _starting = false;
  int _epoch = 0;
  bool loading = false,
      ready = false,
      speaking = false,
      generating = false,
      exporting = false;
  int generatedFrames = 0, predictedFrames = 0, playedFrames = 0;
  String? error;
  String status = 'Choose a model bundle to begin';
  File? wav;
  final List<double> levels = [];
  Timer? _playbackTimer;
  bool _readingPlayback = false;
  SpeechController({PcmPlayer? player, this.muted = false})
    : player = player ?? PcmPlayer();

  void _update() {
    if (!_closed) notifyListeners();
  }

  Future<void> load(String manifest, String bundle) {
    if (_closed) return Future.error(StateError('Player is closed'));
    return _loading ??= _load(
      manifest,
      bundle,
    ).whenComplete(() => _loading = null);
  }

  Future<void> _load(String manifest, String bundle) async {
    loading = true;
    error = null;
    status = 'Loading your voices…';
    _update();
    try {
      await stop();
      ready = false;
      final old = _engine;
      _engine = null;
      await old?.dispose();
      final engine = await Engine.load(manifest: manifest, bundle: bundle);
      if (_closed) {
        await engine.dispose();
        return;
      }
      _engine = engine;
      _temporary ??= await Directory.systemTemp.createTemp('mlsr-speak-');
      ready = true;
      status = 'Ready when you are';
    } catch (failure) {
      error = failure.toString();
      status = 'Could not load voices';
    } finally {
      loading = false;
      _update();
    }
  }

  Future<void> speak(String text, String language, String voice) async {
    if (!ready ||
        loading ||
        _closed ||
        _starting ||
        exporting ||
        text.trim().isEmpty) {
      return;
    }
    _starting = true;
    try {
      await stop();
      if (_closed || !ready) return;
      final epoch = ++_epoch;
      speaking = true;
      generating = true;
      error = null;
      generatedFrames = 0;
      predictedFrames = 0;
      playedFrames = 0;
      levels.clear();
      if (wav != null) {
        await wav!.delete();
        wav = null;
      }
      if (epoch != _epoch || _closed) return;
      status = 'Finding the first words…';
      _update();
      _work = _run(epoch, text, language, voice);
      await _work;
    } finally {
      _starting = false;
    }
  }

  Future<void> _run(
    int epoch,
    String text,
    String language,
    String voice,
  ) async {
    WavSink? sink;
    Job? job;
    bool saved = false;
    try {
      await player.begin(epoch, muted: muted);
      if (epoch != _epoch || _closed) return;
      sink = await WavSink.create(_temporary!, epoch);
      if (epoch != _epoch || _closed) return;
      job = await _engine!.start(text, language: language, voice: voice);
      if (epoch != _epoch || _closed) {
        job.cancel();
        return;
      }
      _job = job;
      _playbackTimer = Timer.periodic(
        const Duration(milliseconds: 100),
        (_) => _readPlayback(epoch),
      );
      await for (final event in job.events) {
        if (epoch != _epoch || _closed) break;
        switch (event.kind) {
          case EventKind.predicted:
            predictedFrames += event.predictedFrames;
          case EventKind.audio:
            await sink.add(event.samples);
            if (epoch != _epoch || _closed) break;
            await player.enqueue(epoch, event.samples);
            if (epoch != _epoch || _closed) break;
            generatedFrames += event.frames;
            var energy = 0.0;
            for (final value in event.samples) {
              energy += value * value;
            }
            levels.add(math.sqrt(energy / event.samples.length).clamp(0, 1));
            if (levels.length > 90) levels.removeAt(0);
            status = 'Reading aloud';
          case EventKind.completed:
            generating = false;
            final file = await sink.finish();
            if (epoch == _epoch && !_closed) {
              wav = file;
              saved = true;
              status = 'Finishing playback';
            }
          case EventKind.failed:
            throw event.error!;
          case EventKind.cancelled:
            return;
        }
        _update();
      }
      if (epoch != _epoch || _closed) return;
      await player.finish(epoch);
      if (epoch != _epoch || _closed) return;
      playedFrames = generatedFrames;
      status = 'All spoken. Yours to keep.';
    } catch (failure) {
      if (epoch == _epoch && !_closed) {
        error = failure.toString();
        status = 'Speech interrupted';
        await player.stop(epoch);
      }
    } finally {
      await job?.dispose();
      if (!saved) await sink?.discard();
      if (epoch == _epoch) {
        _job = null;
        speaking = false;
        generating = false;
        _playbackTimer?.cancel();
        _update();
      }
    }
  }

  Future<void> _readPlayback(int epoch) async {
    if (_readingPlayback) return;
    _readingPlayback = true;
    try {
      final value = await player.state();
      if (epoch == _epoch && !_closed && value['epoch'] == epoch) {
        playedFrames = value['playedFrames'] as int;
        _update();
      }
    } catch (failure) {
      if (epoch == _epoch && !_closed) {
        error = failure.toString();
        _update();
      }
    } finally {
      _readingPlayback = false;
    }
  }

  /// Flush playback before waiting for inference to finish. Awaiting this method
  /// also joins the previous job and file writer before a new request may start.
  Future<void> stop() async {
    final previous = _epoch++;
    final stopped = _epoch;
    _playbackTimer?.cancel();
    _job?.cancel();
    final work = _work;
    try {
      await player.stop(previous);
    } finally {
      await work;
    }
    if (identical(work, _work)) {
      _work = null;
      _job = null;
    }
    if (!_closed && _epoch == stopped) {
      speaking = false;
      generating = false;
      if (ready) status = 'Stopped';
      _update();
    }
  }

  Future<void> exportTo(String path) async {
    final file = wav;
    if (file == null) throw StateError('No completed speech to export');
    await WavSink.export(file, path);
  }

  Future<void> export() async {
    if (exporting || wav == null) return;
    exporting = true;
    _update();
    try {
      final path = await player.destination();
      if (path != null) await exportTo(path);
    } catch (failure) {
      error = failure.toString();
    } finally {
      exporting = false;
      _update();
    }
  }

  Future<void> close() => _closing ??= _close();
  Future<void> _close() async {
    _closed = true;
    await _loading;
    await stop();
    await _engine?.dispose();
    _engine = null;
    await _temporary?.delete(recursive: true);
    _temporary = null;
  }
}
