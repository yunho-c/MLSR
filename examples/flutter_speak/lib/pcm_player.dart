import 'dart:typed_data';
import 'package:flutter/services.dart';

/// Host playback is independent of the synthesis library. enqueue waits when
/// the native player has 0.5 seconds queued; only one caller may enqueue at once.
class PcmPlayer {
  static const _channel = MethodChannel('dev.mlsr.speak/pcm');
  Future<void> begin(int epoch, {bool muted = false}) =>
      _channel.invokeMethod('begin', {'epoch': epoch, 'muted': muted});
  Future<void> enqueue(int epoch, Float32List samples) =>
      _channel.invokeMethod('enqueue', {
        'epoch': epoch,
        'pcm': samples.buffer.asUint8List(
          samples.offsetInBytes,
          samples.lengthInBytes,
        ),
      });
  Future<void> finish(int epoch) =>
      _channel.invokeMethod('finish', {'epoch': epoch});
  Future<void> stop(int epoch) =>
      _channel.invokeMethod('stop', {'epoch': epoch});
  Future<Map<String, Object?>> state() async => Map<String, Object?>.from(
    await _channel.invokeMethod<Map>('state') ?? {},
  );
  Future<String?> destination() => _channel.invokeMethod<String>('save');
}
