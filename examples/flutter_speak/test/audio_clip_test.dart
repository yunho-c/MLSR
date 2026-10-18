import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';

void main() {
  test(
    'public AudioClip writes exact float PCM and preserves an existing file',
    () async {
      final directory = await Directory.systemTemp.createTemp(
        'mlsr-clip-test-',
      );
      try {
        final samples = Float32List.fromList([.1, -.2, -0.0, 1, -1]);
        final clip = AudioClip(
          samples,
          const AudioFormat(sampleRate: 44100, channels: 1),
        );
        final path = '${directory.path}/speech.wav';
        final file = await clip.writeWav(path);
        expect(file.frames, samples.length);
        final bytes = await File(path).readAsBytes();
        final view = ByteData.sublistView(bytes);
        var found = false;
        for (var offset = 12; offset + 8 <= bytes.length;) {
          final size = view.getUint32(offset + 4, Endian.little);
          if (String.fromCharCodes(bytes.sublist(offset, offset + 4)) ==
              'data') {
            expect(
              bytes.sublist(offset + 8, offset + 8 + size),
              samples.buffer.asUint8List(),
            );
            found = true;
          }
          offset += 8 + size + (size & 1);
        }
        expect(found, isTrue);
        await expectLater(
          clip.writeWav(path),
          throwsA(isA<SynthesisException>()),
        );
        expect(await File(path).readAsBytes(), bytes);
        expect(await directory.list().length, 1);
      } finally {
        await directory.delete(recursive: true);
      }
    },
  );
}
