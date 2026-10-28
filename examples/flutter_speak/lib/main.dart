import 'dart:async';
import 'dart:convert';
import 'dart:ui' show AppExitResponse;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:mlsr_app_models/app_models.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart' show SpeakMode, Voice;
import 'speech_controller.dart';
import 'reader_controls.dart';
import 'voice_controls.dart';

const paper = Color(0xfff6f4ed);
const ink = Color(0xff243e33);
const softInk = Color(0xff718075);
const line = Color(0xffd8ddd1);
const accent = Color(0xffdfefb3);
const english =
    'Some words are better heard.\n\nTake a little time to slow down, look around, and listen. There is a whole world in an ordinary day.';
const korean =
    '어떤 말은 소리로 들을 때 더 좋습니다.\n\n잠시 속도를 늦추고 주변을 둘러보세요. 평범한 하루에도 새로운 이야기가 있습니다.';

void main() {
  LicenseRegistry.addLicense(() async* {
    for (final font in ['fraunces', 'manrope']) {
      yield LicenseEntryWithLineBreaks([
        font,
      ], await rootBundle.loadString('assets/fonts/$font-OFL.txt'));
    }
  });
  LicenseRegistry.addLicense(() async* {
    for (final entry in {
      'Supertonic3 model weights': 'supertonic3-OpenRAIL-M.txt',
      'Supertone preprocessing': 'supertonic-MIT.txt',
    }.entries) {
      yield LicenseEntryWithLineBreaks(
        [entry.key],
        await rootBundle.loadString(
          'packages/${AppModels.supertonic3.package}/notices/${entry.value}',
        ),
      );
    }
  });
  runApp(const SpeakApp());
}

class SpeakApp extends StatelessWidget {
  final DemoController? controller;
  const SpeakApp({super.key, this.controller});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'MLSR Speak',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      useMaterial3: true,
      fontFamily: 'Manrope',
      scaffoldBackgroundColor: paper,
      colorScheme: ColorScheme.fromSeed(
        seedColor: ink,
        surface: paper,
        primary: ink,
      ),
      textTheme: ThemeData.light().textTheme.apply(
        fontFamily: 'Manrope',
        bodyColor: ink,
        displayColor: ink,
      ),
      filledButtonTheme: FilledButtonThemeData(
        style: FilledButton.styleFrom(
          backgroundColor: ink,
          foregroundColor: paper,
          minimumSize: const Size(0, 54),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          textStyle: const TextStyle(
            fontFamily: 'Manrope',
            fontSize: 15,
            fontWeight: FontWeight.w700,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: ink,
          minimumSize: const Size(0, 46),
          side: const BorderSide(color: line),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
    ),
    home: RepaintBoundary(
      key: const Key('surface'),
      child: SpeakPage(controller: controller),
    ),
  );
}

class SpeakPage extends StatefulWidget {
  final DemoController? controller;
  const SpeakPage({super.key, this.controller});
  @override
  State<SpeakPage> createState() => _SpeakPageState();
}

class _SpeakPageState extends State<SpeakPage> {
  late final DemoController speech = widget.controller ?? DemoController();
  final text = TextEditingController(text: english);
  String language = 'en', voice = '';
  double synthesisSpeed = 1;
  int steps = 8;
  Voice? get selectedVoice {
    for (final entry in speech.voices) {
      if (entry.id == voice) return entry;
    }
    return speech.voices.firstOrNull;
  }

  String get selectedLanguage {
    final supported = selectedVoice?.languages ?? const <String>[];
    return supported.contains(language)
        ? language
        : supported.firstOrNull ?? language;
  }

  String get example => switch (selectedLanguage) {
    'en' => english,
    'ko' => korean,
    _ => speech.languages[selectedLanguage]?.sample ?? '',
  };
  bool enqueue = false;
  late final AppLifecycleListener lifecycle;
  bool get busy => speech.loading || speech.exporting;
  @override
  void initState() {
    super.initState();
    lifecycle = AppLifecycleListener(
      onExitRequested: () async {
        await speech.close();
        return AppExitResponse.exit;
      },
    );
    if (widget.controller == null && !speech.ready) {
      unawaited(speech.load());
    }
  }

  @override
  void dispose() {
    lifecycle.dispose();
    unawaited(speech.close());
    text.dispose();
    if (widget.controller == null) speech.dispose();
    super.dispose();
  }

  void speak() {
    if (!busy && utf8.encode(text.text).length <= 65536) {
      speech.speak(
        text.text,
        selectedLanguage,
        selectedVoice?.id ?? '',
        synthesisSpeed: synthesisSpeed,
        steps: steps,
        mode: enqueue ? SpeakMode.enqueue : SpeakMode.replace,
      );
    }
  }

  Future<void> settings() async {
    final action = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Your local voices'),
        content: SizedBox(
          width: 420,
          child: Text(
            '${speech.voices.length} voices are available in ${selectedVoice?.languages.length ?? 0} languages.\n\n'
            'Voices are included with this app. Speech runs on your Mac, even when you are offline.\n\n'
            'Voice quality can vary by language. Speech speed and quality settings apply to your next recording.',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, 'licenses'),
            child: const Text('Licenses'),
          ),
          if (!speech.ready)
            TextButton(
              onPressed: () => Navigator.pop(context, 'retry'),
              child: const Text('Try again'),
            ),
          FilledButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Done'),
          ),
        ],
      ),
    );
    if (!mounted) return;
    if (action == 'retry') await speech.load();
    if (action == 'licenses' && mounted) {
      showLicensePage(context: context, applicationName: 'MLSR Speak');
    }
  }

  @override
  Widget build(BuildContext context) => CallbackShortcuts(
    bindings: {
      const SingleActivator(LogicalKeyboardKey.enter, meta: true): speak,
      const SingleActivator(LogicalKeyboardKey.escape): () {
        if (speech.speaking) unawaited(speech.stop());
      },
    },
    child: AnimatedBuilder(
      animation: speech,
      builder: (context, _) => Scaffold(
        body: SafeArea(
          child: SingleChildScrollView(
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1200),
                child: Padding(
                  padding: const EdgeInsets.fromLTRB(40, 26, 40, 24),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Container(
                            width: 34,
                            height: 34,
                            decoration: BoxDecoration(
                              color: ink,
                              borderRadius: BorderRadius.circular(11),
                            ),
                            child: const Icon(
                              Icons.graphic_eq,
                              color: accent,
                              size: 22,
                            ),
                          ),
                          const SizedBox(width: 12),
                          const Text(
                            'mlsr',
                            style: TextStyle(
                              fontSize: 25,
                              fontWeight: FontWeight.w800,
                              letterSpacing: -1.1,
                            ),
                          ),
                          const SizedBox(width: 16),
                          label('SPEAK / 01'),
                          const Spacer(),
                          Container(
                            width: 6,
                            height: 6,
                            decoration: const BoxDecoration(
                              color: softInk,
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 8),
                          const Text(
                            'Local by nature',
                            style: TextStyle(fontSize: 12, color: softInk),
                          ),
                          const SizedBox(width: 12),
                          IconButton(
                            tooltip: 'About voices',
                            onPressed: busy ? null : settings,
                            icon: const Icon(Icons.info_outline, size: 20),
                          ),
                        ],
                      ),
                      const SizedBox(height: 26),
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.end,
                        children: [
                          const Expanded(
                            child: Text(
                              'Give your words\na voice.',
                              style: TextStyle(
                                fontFamily: 'Fraunces',
                                fontSize: 56,
                                height: 1.02,
                                letterSpacing: -2,
                              ),
                            ),
                          ),
                          Padding(
                            padding: const EdgeInsets.only(bottom: 7),
                            child: SizedBox(
                              width: 252,
                              child: Text(
                                'A small space to write, listen,\nand make something yours.',
                                style: TextStyle(
                                  color: ink.withValues(alpha: .65),
                                  height: 1.7,
                                  fontSize: 14,
                                ),
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 28),
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Expanded(
                            child: Container(
                              decoration: BoxDecoration(
                                color: const Color(0xfffdfcf7),
                                border: Border.all(color: line),
                                borderRadius: BorderRadius.circular(18),
                              ),
                              child: Padding(
                                padding: const EdgeInsets.fromLTRB(
                                  24,
                                  20,
                                  24,
                                  12,
                                ),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    Row(
                                      children: [
                                        label('YOUR SCRIPT'),
                                        const Spacer(),
                                        TextButton(
                                          onPressed: busy
                                              ? null
                                              : () {
                                                  text.text = example;
                                                  setState(() {});
                                                },
                                          child: const Text(
                                            'Try an example',
                                            style: TextStyle(fontSize: 11),
                                          ),
                                        ),
                                      ],
                                    ),
                                    SizedBox(
                                      height: 226,
                                      child: TextField(
                                        key: const Key('script'),
                                        controller: text,
                                        textDirection:
                                            speech
                                                    .languages[selectedLanguage]
                                                    ?.rtl ==
                                                true
                                            ? TextDirection.rtl
                                            : TextDirection.ltr,
                                        readOnly: busy,
                                        maxLines: null,
                                        expands: true,
                                        textAlignVertical:
                                            TextAlignVertical.top,
                                        onChanged: (_) => setState(() {}),
                                        style: const TextStyle(
                                          fontSize: 19,
                                          height: 1.7,
                                          letterSpacing: -.25,
                                        ),
                                        decoration: const InputDecoration(
                                          border: InputBorder.none,
                                          hintText:
                                              'What would you like to hear?',
                                          contentPadding: EdgeInsets.only(
                                            top: 10,
                                          ),
                                        ),
                                      ),
                                    ),
                                    const Divider(color: line, height: 1),
                                    const SizedBox(height: 10),
                                    Row(
                                      children: [
                                        Text(
                                          '${text.text.runes.length} characters',
                                          style: const TextStyle(
                                            fontSize: 11,
                                            color: softInk,
                                          ),
                                        ),
                                        const Spacer(),
                                        Text(
                                          utf8.encode(text.text).length > 65536
                                              ? 'Please shorten this script'
                                              : '⌘ Enter to speak',
                                          style: const TextStyle(
                                            fontSize: 11,
                                            color: softInk,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ),
                          const SizedBox(width: 28),
                          SizedBox(
                            width: 252,
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Padding(
                                  padding: const EdgeInsets.only(
                                    top: 6,
                                    bottom: 14,
                                  ),
                                  child: label('MAKE IT SOUND LIKE YOU'),
                                ),
                                VoiceControls(
                                  voices: speech.voices,
                                  voice: selectedVoice,
                                  languages: speech.languages,
                                  language: selectedLanguage,
                                  enabled: !busy && speech.ready,
                                  speed: synthesisSpeed,
                                  steps: steps,
                                  onVoice: (value) =>
                                      setState(() => voice = value),
                                  onLanguage: (value) =>
                                      setState(() => language = value),
                                  onSpeed: (value) =>
                                      setState(() => synthesisSpeed = value),
                                  onSteps: (value) =>
                                      setState(() => steps = value),
                                ),
                                const SizedBox(height: 20),
                                SizedBox(
                                  width: double.infinity,
                                  child: FilledButton.icon(
                                    key: const Key('speak'),
                                    onPressed:
                                        busy ||
                                            text.text.trim().isEmpty ||
                                            utf8.encode(text.text).length >
                                                65536
                                        ? null
                                        : speech.ready
                                        ? speak
                                        : () => unawaited(speech.load()),
                                    icon: const Icon(Icons.play_arrow_rounded),
                                    label: Text(
                                      speech.loading
                                          ? 'Loading voices…'
                                          : !speech.ready
                                          ? 'Try loading again'
                                          : speech.speaking
                                          ? enqueue
                                                ? 'Add to queue'
                                                : 'Speak new text'
                                          : 'Speak',
                                    ),
                                  ),
                                ),
                                Row(
                                  children: [
                                    Checkbox(
                                      key: const Key('enqueue'),
                                      value: enqueue,
                                      onChanged: busy
                                          ? null
                                          : (value) => setState(
                                              () => enqueue = value!,
                                            ),
                                    ),
                                    const Expanded(
                                      child: Text(
                                        'Queue after current',
                                        style: TextStyle(fontSize: 12),
                                      ),
                                    ),
                                  ],
                                ),
                                if (speech.speaking || speech.pendingCount > 0)
                                  SizedBox(
                                    width: double.infinity,
                                    child: OutlinedButton.icon(
                                      key: const Key('stop'),
                                      onPressed: () => unawaited(speech.stop()),
                                      icon: const Icon(Icons.stop_rounded),
                                      label: Text(
                                        speech.pendingCount > 0
                                            ? 'Stop all (${speech.pendingCount} queued)'
                                            : 'Stop',
                                      ),
                                    ),
                                  ),
                                const SizedBox(height: 10),
                                SizedBox(
                                  width: double.infinity,
                                  child: OutlinedButton.icon(
                                    key: const Key('export'),
                                    onPressed:
                                        !speech.canExport || speech.exporting
                                        ? null
                                        : () => unawaited(speech.export()),
                                    icon: const Icon(
                                      Icons.arrow_outward,
                                      size: 18,
                                    ),
                                    label: const Text('Export WAV'),
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 24),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 22,
                          vertical: 18,
                        ),
                        decoration: BoxDecoration(
                          color: ink,
                          borderRadius: BorderRadius.circular(16),
                        ),
                        child: Row(
                          children: [
                            Container(
                              width: 40,
                              height: 40,
                              decoration: BoxDecoration(
                                color: accent,
                                borderRadius: BorderRadius.circular(12),
                              ),
                              child: Icon(
                                speech.speaking
                                    ? Icons.volume_up_outlined
                                    : Icons.headphones_outlined,
                                size: 23,
                                color: ink,
                              ),
                            ),
                            const SizedBox(width: 16),
                            SizedBox(
                              width: 220,
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    speech.status,
                                    key: const Key('status'),
                                    style: const TextStyle(
                                      color: paper,
                                      fontSize: 12,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                  const SizedBox(height: 6),
                                  Text(
                                    speech.speaking
                                        ? '${speech.paused ? 'Paused' : 'Played'} ${speech.timeline.positionLabel} / ${speech.timeline.totalLabel}'
                                        : 'Supertonic 3  ·  44.1 kHz',
                                    key: const Key('timeline-position'),
                                    style: TextStyle(
                                      color: paper.withValues(alpha: .55),
                                      fontSize: 10,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            const SizedBox(width: 20),
                            Expanded(
                              child: SizedBox(
                                height: 42,
                                child: Center(
                                  child: Stack(
                                    children: [
                                      if (speech.timeline.hasTotal)
                                        LinearProgressIndicator(
                                          key: const Key('buffered-progress'),
                                          value:
                                              speech.timeline.bufferedProgress,
                                          color: accent.withValues(alpha: .3),
                                          backgroundColor: paper.withValues(
                                            alpha: .15,
                                          ),
                                          semanticsLabel: 'Available audio',
                                        ),
                                      LinearProgressIndicator(
                                        key: const Key('playback-progress'),
                                        value:
                                            speech.timeline.progress ??
                                            (speech.speaking ? null : 0.0),
                                        color: accent,
                                        backgroundColor:
                                            speech.timeline.hasTotal
                                            ? Colors.transparent
                                            : paper.withValues(alpha: .15),
                                        semanticsLabel: speech.timeline.hasTotal
                                            ? 'Played fraction of ${speech.timeline.estimated ? 'estimated ' : ''}total duration'
                                            : 'Estimating total duration',
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 22),
                            Text(
                              speech.timeline.totalLabel,
                              key: const Key('timeline-total'),
                              style: const TextStyle(
                                color: accent,
                                fontSize: 22,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ],
                        ),
                      ),
                      if (speech.original.isNotEmpty) ...[
                        const SizedBox(height: 18),
                        ReaderControls(controller: speech),
                      ],
                      if (speech.error != null)
                        Padding(
                          padding: const EdgeInsets.only(top: 14),
                          child: SelectableText(
                            speech.error!,
                            key: const Key('error'),
                            style: const TextStyle(
                              color: Color(0xff9e392d),
                              fontSize: 12,
                            ),
                          ),
                        ),
                      const SizedBox(height: 18),
                      Row(
                        children: [
                          label('YOUR WORDS. YOUR DEVICE.'),
                          const Spacer(),
                          TextButton(
                            onPressed: () => showLicensePage(
                              context: context,
                              applicationName: 'MLSR Speak',
                              applicationVersion: '0.1.0',
                            ),
                            child: const Text(
                              'Made with MLSR ↗',
                              style: TextStyle(fontSize: 11, color: softInk),
                            ),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    ),
  );
}

Widget label(String text) => Text(
  text,
  style: const TextStyle(
    fontSize: 9,
    fontWeight: FontWeight.w700,
    letterSpacing: 1.5,
    color: softInk,
  ),
);
