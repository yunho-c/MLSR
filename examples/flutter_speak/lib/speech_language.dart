import 'dart:convert';
import 'package:flutter/services.dart';

/// Presentation labels and examples. Available languages come from the model's
/// voice catalog, never from this table; unknown tags keep their original label.
class SpeechLanguage {
  final String name, sample;
  final bool rtl;
  const SpeechLanguage(this.name, this.sample, {this.rtl = false});

  static Future<Map<String, SpeechLanguage>> load() async {
    final values =
        jsonDecode(await rootBundle.loadString('assets/languages.json'))
            as Map<String, dynamic>;
    return Map.unmodifiable(
      values.map(
        (id, value) => MapEntry(
          id,
          SpeechLanguage(
            value['name'] as String,
            value['sample'] as String,
            rtl: value['rtl'] as bool,
          ),
        ),
      ),
    );
  }
}
