import 'package:flutter/material.dart';
import 'package:mlsr_flutter/mlsr_flutter.dart';
import 'speech_language.dart';

/// Selection and generation controls for the actual loaded model catalog.
class VoiceControls extends StatelessWidget {
  final List<Voice> voices;
  final Voice? voice;
  final Map<String, SpeechLanguage> languages;
  final String language;
  final bool enabled;
  final double speed;
  final int steps;
  final ValueChanged<String> onVoice, onLanguage;
  final ValueChanged<double> onSpeed;
  final ValueChanged<int> onSteps;
  const VoiceControls({
    super.key,
    required this.voices,
    required this.voice,
    required this.languages,
    required this.language,
    required this.enabled,
    required this.speed,
    required this.steps,
    required this.onVoice,
    required this.onLanguage,
    required this.onSpeed,
    required this.onSteps,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final supported = voice?.languages ?? const <String>[];
    final active = enabled && voice != null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        DropdownButtonFormField<String>(
          key: const Key('language'),
          initialValue: supported.contains(language) ? language : null,
          isExpanded: true,
          menuMaxHeight: 360,
          decoration: const InputDecoration(
            labelText: 'Language',
            border: OutlineInputBorder(),
          ),
          items: [
            for (final id in supported)
              DropdownMenuItem(
                value: id,
                child: Text(
                  languages[id]?.name ?? id,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
          ],
          onChanged: active ? (value) => onLanguage(value!) : null,
        ),
        const SizedBox(height: 16),
        DropdownButtonFormField<String>(
          key: const Key('voice'),
          initialValue: voice?.id,
          isExpanded: true,
          menuMaxHeight: 360,
          decoration: const InputDecoration(
            labelText: 'Voice',
            border: OutlineInputBorder(),
          ),
          items: [
            for (final entry in voices)
              DropdownMenuItem(
                key: Key('voice-${entry.modelVoiceId}'),
                value: entry.id,
                child: Text(
                  '${entry.displayName} · ${entry.modelVoiceId}',
                  overflow: TextOverflow.ellipsis,
                ),
              ),
          ],
          onChanged: active ? (value) => onVoice(value!) : null,
        ),
        const SizedBox(height: 8),
        Text(
          voices.isEmpty
              ? 'Getting your voices ready…'
              : '${voices.length} voices · ${supported.length} languages · Offline',
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
        const SizedBox(height: 18),
        Row(
          children: [
            const Expanded(child: Text('Speech speed')),
            Text(
              '${speed.toStringAsFixed(2)}×',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ],
        ),
        Slider(
          key: const Key('synthesis-speed'),
          value: speed,
          min: .8,
          max: 1.3,
          divisions: 10,
          label: '${speed.toStringAsFixed(2)}×',
          onChanged: active ? onSpeed : null,
        ),
        Row(
          children: [
            const Expanded(child: Text('Quality')),
            Text(
              '$steps steps',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ],
        ),
        Slider(
          key: const Key('quality-steps'),
          value: steps.toDouble(),
          min: 2,
          max: 16,
          divisions: 14,
          label: '$steps steps',
          onChanged: active ? (value) => onSteps(value.round()) : null,
        ),
      ],
    );
  }
}
