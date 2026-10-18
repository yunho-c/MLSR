import 'dart:io';
import 'dart:typed_data';

/// Incremental IEEE-float WAV writer. Audio stays on disk, not in a growing list.
class WavSink {
  final File file;
  final RandomAccessFile _output;
  int frames = 0;
  bool _closed = false;
  WavSink._(this.file, this._output);
  static Future<WavSink> create(Directory directory, int epoch) async {
    final file = File('${directory.path}/speech-$epoch.wav');
    await file.create(exclusive: true);
    final output = await file.open(mode: FileMode.write);
    await output.writeFrom(Uint8List(56));
    return WavSink._(file, output);
  }

  Future<void> add(Float32List samples) async {
    if (_closed || (frames + samples.length) * 4 > 0xffffffff - 48) {
      throw StateError('Speech exceeds the WAV size limit');
    }
    await _output.writeFrom(
      samples.buffer.asUint8List(samples.offsetInBytes, samples.lengthInBytes),
    );
    frames += samples.length;
  }

  Future<File> finish() async {
    final header = ByteData(56);
    void label(int start, String text) {
      for (var i = 0; i < text.length; i++) {
        header.setUint8(start + i, text.codeUnitAt(i));
      }
    }

    void u32(int offset, int value) =>
        header.setUint32(offset, value, Endian.little);
    void u16(int offset, int value) =>
        header.setUint16(offset, value, Endian.little);
    label(0, 'RIFF');
    u32(4, 48 + frames * 4);
    label(8, 'WAVE');
    label(12, 'fmt ');
    u32(16, 16);
    u16(20, 3);
    u16(22, 1);
    u32(24, 44100);
    u32(28, 176400);
    u16(32, 4);
    u16(34, 32);
    label(36, 'fact');
    u32(40, 4);
    u32(44, frames);
    label(48, 'data');
    u32(52, frames * 4);
    await _output.setPosition(0);
    await _output.writeFrom(header.buffer.asUint8List());
    await _output.flush();
    await _output.close();
    _closed = true;
    return file;
  }

  Future<void> discard() async {
    if (!_closed) {
      await _output.close();
      _closed = true;
    }
    if (await file.exists()) {
      await file.delete();
    }
  }

  static Future<void> export(File source, String path) async {
    final destination = File(path);
    await destination.create(exclusive: true);
    try {
      final sink = destination.openWrite();
      try {
        await sink.addStream(source.openRead());
        await sink.flush();
      } finally {
        await sink.close();
      }
    } catch (_) {
      await destination.delete();
      rethrow;
    }
  }
}
