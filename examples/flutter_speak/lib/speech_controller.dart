import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';
import 'package:mlsr_app_models/app_models.dart';
import 'file_dialog.dart';

class _Presentation {
  final Speech speech;
  final String original;
  SpeechState? state;
  StreamSubscription<SpeechState>? subscription;
  _Presentation(this.speech, this.original);
  Future<void> close() async {
    await subscription?.cancel();
    await speech.dispose();
  }
}

/// UI ownership and presentation only. Rust chooses admission, queue order,
/// replacement, playback completion, retention and navigation. No PCM crosses
/// Flutter and this class never advances a queue or implements a playback clock.
class DemoController extends ChangeNotifier {
  final bool muted;
  Tts? _tts;
  final Map<int, _Presentation> _presentations = {};
  final Set<Future<void>> _retiring = {};
  _Presentation? _current;
  Future<void>? _loading, _closing;
  bool _closed = false;
  bool loading = false, ready = false, exporting = false;
  String? error;
  String _loadStatus = 'Getting your voices ready…';
  double playbackRate = 1, volume = 1;
  DemoController({this.muted = false});

  Speech? get speech => _current?.speech;
  SpeechState? get state => _current?.state;
  String get original => _current?.original ?? '';
  bool get speaking => _current != null && state?.completion == null;
  bool get paused => state?.playback == PlaybackPhase.paused;
  bool get generating =>
      speaking &&
      state?.synthesis != SynthesisPhase.generated &&
      state?.synthesis != SynthesisPhase.terminal;
  bool get canExport => state?.completion?.kind == OutcomeKind.completed;
  int get generatedFrames => state?.bufferedUntilFrame ?? 0;
  int get playedFrames => state?.positionFrame ?? 0;
  int get pendingCount => _presentations.values
      .where(
        (entry) =>
            !identical(entry, _current) && entry.state?.completion == null,
      )
      .length;
  String get status {
    if (loading || !ready) return _loadStatus;
    if (_current == null) return 'Ready when you are';
    return switch (state?.playback) {
      null || PlaybackPhase.queued => 'Waiting to speak…',
      PlaybackPhase.buffering => 'Finding the next words…',
      PlaybackPhase.playing => 'Reading aloud',
      PlaybackPhase.draining => 'Finishing playback',
      PlaybackPhase.paused => 'Paused',
      PlaybackPhase.ended => 'All spoken. Yours to keep.',
      PlaybackPhase.stopped => 'Stopped',
      PlaybackPhase.failed => 'Speech interrupted',
    };
  }

  void _update() {
    if (!_closed) notifyListeners();
  }

  void _failure(Object failure) {
    error = failure.toString();
    _update();
  }

  Future<void> load({ModelDelivery delivery = ModelDelivery.automatic}) {
    if (_closed) return Future.error(StateError('Player is closed'));
    if (ready) return Future.value();
    return _loading ??= _load(delivery).whenComplete(() => _loading = null);
  }

  Future<void> _load(ModelDelivery delivery) async {
    loading = true;
    error = null;
    _update();
    try {
      final tts = await FlutterMlsr.openTts(
        model: AppModels.supertonic3,
        delivery: delivery,
        onProgress: (progress) {
          _loadStatus = switch (progress.stage) {
            ModelLoadStage.locating => 'Finding your voices…',
            ModelLoadStage.waiting => 'Waiting for voices…',
            ModelLoadStage.copying =>
              progress.totalBytes == 0
                  ? 'Preparing voices…'
                  : 'Preparing voices… ${(100 * progress.copiedBytes / progress.totalBytes).floor()}%',
            ModelLoadStage.verifying => 'Checking voices…',
            ModelLoadStage.loading => 'Loading your voices…',
          };
          _update();
        },
      );
      if (_closed) {
        await tts.dispose();
        return;
      }
      _tts = tts;
      ready = true;
    } catch (failure) {
      error = failure.toString();
      _loadStatus = 'Could not load voices';
    } finally {
      loading = false;
      _update();
    }
  }

  SpeechOptions _options({bool paused = false}) => SpeechOptions(
    volume: muted ? 0 : volume,
    playbackRate: playbackRate,
    paused: paused,
  );

  /// Admission is synchronous: a rejected replacement preserves the current
  /// native request and its presentation. Accepted enqueue stays native-owned.
  Speech? speak(
    String text,
    String language,
    String voice, {
    SpeakMode mode = SpeakMode.replace,
    bool paused = false,
  }) {
    if (!ready || loading || exporting || _closed || text.trim().isEmpty) {
      return null;
    }
    try {
      final next = _tts!.speak(
        text,
        request: RequestOptions(locale: language, voice: 'supertonic3:$voice'),
        options: _options(paused: paused),
        mode: mode,
      );
      if (mode == SpeakMode.replace) {
        for (final old in _presentations.values.toList()) {
          _retire(old);
        }
      }
      error = null;
      _observe(next, text, select: mode == SpeakMode.replace || !speaking);
      return next;
    } catch (failure) {
      _failure(failure);
      return null;
    }
  }

  void _observe(Speech next, String original, {bool select = true}) {
    final entry = _Presentation(next, original);
    _presentations[next.id] = entry;
    if (select) _select(entry);
    entry.subscription = next.states.listen(
      (snapshot) {
        if (_closed || !_presentations.containsKey(next.id)) return;
        entry.state = snapshot;
        // State from the native controller determines which queued request is
        // active. This observer does not start, stop, or advance queued work.
        if (snapshot.playback != PlaybackPhase.queued &&
            snapshot.completion == null &&
            !identical(entry, _current)) {
          _select(entry);
        }
        if (identical(entry, _current)) {
          if (snapshot.completion?.failure case final failure?) {
            error = failure.message;
          }
        } else if (snapshot.completion != null) {
          _retire(entry);
        }
        _update();
      },
      onError: (Object failure) {
        if (identical(entry, _current)) _failure(failure);
      },
    );
    _update();
  }

  void _select(_Presentation entry) {
    final old = _current;
    _current = entry;
    if (old != null && old.state?.completion != null) _retire(old);
  }

  void _retire(_Presentation entry) {
    if (_presentations.remove(entry.speech.id) == null) return;
    if (identical(entry, _current)) _current = null;
    late final Future<void> work;
    work = entry
        .close()
        .catchError((Object failure) {
          if (!_closed) _failure(failure);
        })
        .whenComplete(() => _retiring.remove(work));
    _retiring.add(work);
  }

  Future<void> stop() async {
    final entries = _presentations.values.toList();
    for (final entry in entries) {
      entry.speech.cancel();
    }
    await Future.wait(
      entries.map((entry) async {
        try {
          await entry.speech.done;
        } catch (failure) {
          _failure(failure);
        }
      }),
    );
  }

  void _control(void Function(Speech) action) {
    final current = speech;
    if (current == null || _closed) return;
    try {
      action(current);
    } catch (failure) {
      _failure(failure);
    }
  }

  void togglePause() => _control((current) {
    if (canExport) {
      final next = current.replayFrom(0, options: _options());
      final text = original;
      _retire(_current!);
      _observe(next, text);
    } else if (paused) {
      current.resume();
    } else {
      current.pause();
    }
  });
  void setRate(double value) {
    _control((current) {
      if (speaking) current.setPlaybackRate(value);
    });
    playbackRate = value;
    _update();
  }

  void setVolume(double value) {
    _control((current) {
      if (speaking) current.setVolume(muted ? 0 : value);
    });
    volume = value;
    _update();
  }

  void seek(int frame) => _control((current) {
    if (canExport) {
      final next = current.replayFrom(frame, options: _options(paused: true));
      final text = original;
      _retire(_current!);
      _observe(next, text);
    } else {
      current.seek(frame);
    }
  });
  void navigate({required bool next}) => _control((current) {
    final result = next ? current.nextSentence() : current.previousSentence();
    if (result.speech case final replacement?) {
      final text = original;
      _retire(_current!);
      _observe(replacement, text);
    }
  });

  Future<TtsState> engineState() => _tts!.snapshot();
  Future<WavFile> exportTo(String path) {
    if (!canExport) throw StateError('No completed speech to export');
    return _tts!.writeSpeech(speech!, path);
  }

  Future<void> export() async {
    if (exporting || !canExport) return;
    exporting = true;
    _update();
    try {
      final path = await FileDialog.destination();
      if (path != null && !_closed) await exportTo(path);
    } catch (failure) {
      _failure(failure);
    } finally {
      exporting = false;
      _update();
    }
  }

  Future<void> close() => _closing ??= _close();
  Future<void> _close() async {
    _closed = true;
    await _loading;
    await _tts?.dispose();
    for (final entry in _presentations.values.toList()) {
      _retire(entry);
    }
    await Future.wait(_retiring.toList());
    _tts = null;
    ready = false;
  }
}
