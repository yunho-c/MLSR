import 'dart:io';
import 'package:mlsr/mlsr.dart';

/// Public facade example. The optional `speak` argument uses native output;
/// otherwise this exports headlessly with bounded PCM and the shared WAV writer.
Future<void> main(List<String> args) async {
  if (args.length < 3 ||
      args.length > 4 ||
      (args.length == 4 && args[3] != 'speak')) {
    stderr.writeln(
      'Usage: dart run example/developer_api.dart MANIFEST BUNDLE NEW_WAV [speak]',
    );
    exitCode = 64;
    return;
  }
  final speak = args.length == 4;
  final tts = await Tts.open(
    manifest: args[0],
    bundle: args[1],
    options: TtsOptions(
      output: speak ? OutputMode.platformDefault : OutputMode.headless,
    ),
  );
  const options = RequestOptions(backend: SupertonicOptions(seed: 1729));
  try {
    WavFile result;
    if (speak) {
      final speech = tts.speak('Hello from MLSR.', request: options);
      try {
        await speech.done; // Wait for actual native playback drain.
        result = await tts.writeSpeech(speech, args[2]);
      } finally {
        await speech.dispose();
      }
    } else {
      final job = tts.synthesize('Hello from MLSR.', options: options);
      try {
        result = await tts.writeJob(job, args[2], maxFrames: 60 * 44100);
        await job.done;
      } finally {
        await job.dispose();
      }
    }
    stdout.writeln('Wrote ${result.frames} frames to ${result.path}');
  } finally {
    await tts.dispose();
  }
}
