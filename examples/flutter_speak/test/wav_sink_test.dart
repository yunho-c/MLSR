import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mlsr_speak/wav_sink.dart';

void main() {
  test(
    'incremental float WAV preserves samples and export never clobbers',
    () async {
      final directory = await Directory.systemTemp.createTemp('mlsr-wav-test-');
      try {
        final sink = await WavSink.create(directory, 1);
        final samples = Float32List.fromList([.1, -.2, -0.0, 1, -1]);
        await sink.add(Float32List.sublistView(samples, 0, 2));
        await sink.add(Float32List.sublistView(samples, 2));
        final source = await sink.finish();
        final bytes = await source.readAsBytes();
        final header = ByteData.sublistView(bytes);
        expect(header.getUint16(20, Endian.little), 3);
        expect(header.getUint32(24, Endian.little), 44100);
        expect(header.getUint32(44, Endian.little), samples.length);
        expect(header.getUint32(52, Endian.little), samples.lengthInBytes);
        expect(bytes.sublist(56), samples.buffer.asUint8List());
        final path = '${directory.path}/export.wav';
        await WavSink.export(source, path);
        expect(await File(path).readAsBytes(), bytes);
        await expectLater(
          WavSink.export(source, path),
          throwsA(isA<FileSystemException>()),
        );
        expect(await File(path).readAsBytes(), bytes);
        final partial = await WavSink.create(directory, 2);
        await partial.add(samples);
        await partial.discard();
        expect(await partial.file.exists(), isFalse);
      } finally {
        await directory.delete(recursive: true);
      }
    },
  );
}
