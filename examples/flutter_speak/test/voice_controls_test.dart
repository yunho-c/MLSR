import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';
import 'package:mlsr_speak/main.dart';
import 'package:mlsr_speak/speech_controller.dart';
import 'package:mlsr_speak/speech_language.dart';

void main() {
  testWidgets(
    'catalog drives selectors, preserves text and supports Arabic input',
    (tester) async {
      tester.view.physicalSize = const Size(1100, 1100);
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final controller = (await tester.runAsync(
        () async => DemoController()
          ..voices = await getVoices()
          ..languages = await SpeechLanguage.load()
          ..ready = true,
      ))!;
      await tester.pumpWidget(SpeakApp(controller: controller));
      await tester.pumpAndSettle();
      final voiceField = find.byKey(const Key('voice'));
      final languageField = find.byKey(const Key('language'));
      DropdownButton<String> dropdown(Finder field) =>
          tester.widget<DropdownButton<String>>(
            find.descendant(
              of: field,
              matching: find.byType(DropdownButton<String>),
            ),
          );
      expect(dropdown(voiceField).items!.length, 10);
      expect(dropdown(languageField).items!.length, 31);
      expect(
        controller.languages.keys,
        containsAll(controller.voices.first.languages),
      );
      await tester.enterText(
        find.byKey(const Key('script')),
        'Keep my writing.',
      );
      await tester.tap(voiceField);
      await tester.pumpAndSettle();
      await tester.scrollUntilVisible(
        find.text('Olivia · F4'),
        100,
        scrollable: find.byType(Scrollable).last,
      );
      await tester.tap(find.text('Olivia · F4').last);
      await tester.pumpAndSettle();
      expect(
        tester.widget<DropdownButtonFormField<String>>(voiceField).initialValue,
        'supertonic3:F4',
      );
      await tester.tap(languageField);
      await tester.pumpAndSettle();
      await tester.tap(find.text('العربية · Arabic').last);
      await tester.pumpAndSettle();
      final editor = tester.widget<TextField>(find.byKey(const Key('script')));
      expect(editor.controller!.text, 'Keep my writing.');
      expect(editor.textDirection, TextDirection.rtl);
      await tester.tap(find.text('Try an example'));
      await tester.pump();
      expect(editor.controller!.text, controller.languages['ar']!.sample);
      expect(tester.takeException(), isNull);
      await tester.pumpWidget(const SizedBox());
      await controller.close();
      controller.dispose();
    },
  );
}
