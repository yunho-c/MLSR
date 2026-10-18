import 'dart:convert';
import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:mlsr/mlsr.dart';
import 'package:yaml/yaml.dart';

/// Integrity/lifetime fixture only. Graphs are deliberately not executable.
class ModelFixture {
  final Directory source;
  final ModelDescriptor model;
  ModelFixture._(this.source, this.model);

  static Future<ModelFixture> create() async {
    final source = await Directory.systemTemp.createTemp('mlsr-model-fixture-');
    try {
      final manifest =
          jsonDecode(
                jsonEncode(
                  loadYaml(
                    await File(
                      'test/fixtures/supertonic3-coreml.yaml',
                    ).readAsString(),
                  ),
                ),
              )
              as Map<String, dynamic>;
      final artifacts = manifest['artifacts'] as Map<String, dynamic>;
      for (final entry in artifacts.entries) {
        final artifact = entry.value as Map<String, dynamic>;
        final graph = artifact['kind'] == 'coreml_compiled';
        final path = graph
            ? '${artifact['path']}/payload.bin'
            : artifact['path'] as String;
        final bytes = switch (entry.key) {
          'tts_config' => utf8.encode(
            jsonEncode({
              'ae': {'sample_rate': 44100, 'base_chunk_size': 512},
              'ttl': {'chunk_compress_factor': 6, 'latent_dim': 24},
            }),
          ),
          'unicode_indexer' => utf8.encode(jsonEncode(List.filled(65536, -1))),
          'voice_M1' || 'voice_F1' => utf8.encode(
            jsonEncode({
              'style_ttl': {
                'dims': [1, 50, 256],
                'type': 'float32',
                'data': [List.generate(50, (_) => List.filled(256, 0.25))],
              },
              'style_dp': {
                'dims': [1, 8, 16],
                'type': 'float32',
                'data': [List.generate(8, (_) => List.filled(16, -0.5))],
              },
            }),
          ),
          _ => utf8.encode('integrity fixture, not an executable model'),
        };
        final file = File('${source.path}/$path');
        await file.parent.create(recursive: true);
        await file.writeAsBytes(bytes);
        artifact['files'] = [
          {
            'path': path,
            'size_bytes': bytes.length,
            'sha256': sha256.convert(bytes).toString(),
          },
        ];
      }
      // JSON is a YAML subset and avoids a test-only YAML serializer. Production
      // preparation copies original YAML bytes, including comments, unchanged.
      final bytes = utf8.encode(jsonEncode(manifest));
      await File('${source.path}/mlsr-bundle.yaml').writeAsBytes(bytes);
      final catalog =
          jsonDecode(
                jsonEncode(
                  loadYaml(
                    await File('test/fixtures/catalog.yaml').readAsString(),
                  ),
                ),
              )
              as Map<String, dynamic>;
      (catalog['models'] as List).single['manifest_sha256'] = sha256
          .convert(bytes)
          .toString();
      return ModelFixture._(
        source,
        ModelDescriptor(
          catalogYaml: jsonEncode(catalog),
          id: manifest['id'] as String,
          platform: 'macos-arm64',
        ),
      );
    } catch (_) {
      await source.delete(recursive: true);
      rethrow;
    }
  }

  Future<void> fill(ModelInstallation installation) async {
    await for (final entry in source.list(
      recursive: true,
      followLinks: false,
    )) {
      if (entry is File) {
        final relative = entry.path.substring(source.path.length + 1);
        final target = File('${installation.directory}/$relative');
        await target.parent.create(recursive: true);
        await entry.copy(target.path);
      }
    }
  }

  Future<void> dispose() => source.delete(recursive: true);
}
