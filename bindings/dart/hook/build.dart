import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

Future<void> main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final config = input.config.code;
    final coreml = input.userDefines['coreml'] ?? false;
    final qnn = input.userDefines['qnn'] ?? false;
    if (coreml is! bool) {
      throw FormatException('MLSR coreml must be true or false.');
    }
    if (qnn is! bool) {
      throw FormatException('MLSR qnn must be true or false.');
    }
    if (qnn &&
        (config.targetOS != OS.windows ||
            config.targetArchitecture != Architecture.arm64)) {
      throw UnsupportedError(
        'QNN requires Windows ARM64 and prepared native dependencies.',
      );
    }
    final appleOutput = input.userDefines['apple-output'] ?? false;
    if (appleOutput is! bool) {
      throw FormatException('MLSR apple-output must be true or false.');
    }
    if (appleOutput && config.targetOS != OS.macOS) {
      throw UnsupportedError('The current native output hook supports macOS.');
    }
    if (coreml && config.targetOS != OS.macOS) {
      throw UnsupportedError('The current Core ML build hook supports macOS.');
    }
    if (config.linkModePreference == LinkModePreference.static) {
      throw UnsupportedError('MLSR currently bundles a shared library.');
    }
    if (config.targetOS != OS.current ||
        config.targetArchitecture != Architecture.current) {
      throw UnsupportedError(
        'MLSR currently supports native desktop host builds. '
        'Cross-compilation and mobile packaging are not configured yet.',
      );
    }
    final target = switch ((config.targetOS, config.targetArchitecture)) {
      (OS.macOS, Architecture.arm64) => 'aarch64-apple-darwin',
      (OS.macOS, Architecture.x64) => 'x86_64-apple-darwin',
      (OS.linux, Architecture.arm64) => 'aarch64-unknown-linux-gnu',
      (OS.linux, Architecture.x64) => 'x86_64-unknown-linux-gnu',
      (OS.windows, Architecture.arm64) => 'aarch64-pc-windows-msvc',
      (OS.windows, Architecture.x64) => 'x86_64-pc-windows-msvc',
      _ => throw UnsupportedError('Unsupported MLSR desktop target.'),
    };
    final libraryName = config.targetOS.dylibFileName('mlsr_ffi');
    final supplied = input.userDefines['native-library'];
    if (supplied != null && supplied is! String) {
      throw FormatException('MLSR native-library must be a filesystem path.');
    }
    Uri source;
    // Older supported hooks versions parse Windows drive letters as URI schemes.
    final prebuilt = supplied is String && File(supplied).isAbsolute
        ? File(supplied).uri
        : input.userDefines.path('native-library');
    if (prebuilt != null) {
      source = prebuilt;
      output.dependencies.add(source);
    } else {
      final root = input.packageRoot.resolve('../../');
      final manifest = root.resolve('Cargo.toml');
      if (!File.fromUri(
        root.resolve('crates/mlsr-ffi/Cargo.toml'),
      ).existsSync()) {
        throw StateError(
          'MLSR Rust sources are absent. Use the repository checkout or set '
          'hooks.user_defines.mlsr.native-library in your pubspec.yaml.',
        );
      }
      final cargoOutput = input.outputDirectory.resolve('cargo/');
      final cargoEnvironment = Map<String, String>.of(Platform.environment);
      if (qnn) {
        // Hooks intentionally filter environment variables. Resolve this build
        // dependency explicitly and track it so replacing the bridge rebuilds
        // the asset. The SDK/FastRPC DLLs remain runtime prerequisites.
        final suppliedBridge = input.userDefines['qnn-bridge-directory'];
        if (suppliedBridge != null && suppliedBridge is! String) {
          throw FormatException('MLSR qnn-bridge-directory must be a path.');
        }
        final bridge =
            suppliedBridge is String && Directory(suppliedBridge).isAbsolute
            ? Directory(suppliedBridge).uri
            : input.userDefines.path('qnn-bridge-directory') ??
                  root.resolve('target/qnn-native/Release/');
        final bridgeDirectory = Directory.fromUri(bridge).absolute;
        final importLibrary = File(
          '${bridgeDirectory.path}${Platform.pathSeparator}mlsr_qnn_bridge.lib',
        );
        if (!importLibrary.existsSync()) {
          throw StateError(
            'Build tools/build_qnn.ps1 first or supply '
            'qnn-bridge-directory containing mlsr_qnn_bridge.lib.',
          );
        }
        cargoEnvironment['MLSR_QNN_BRIDGE_DIR'] = bridgeDirectory.path;
        output.dependencies.add(importLibrary.uri);
      }
      if (config.targetOS == OS.macOS) {
        // Xcode puts its toolchain ahead of /usr/bin. A bare `cc` can then
        // bypass Apple's SDK-selecting wrapper, leaving cidre without headers.
        // Resolve the active toolchain explicitly; preserve deliberate CC
        // overrides. Never inherit an iOS deployment target into a host build.
        cargoEnvironment.remove('IPHONEOS_DEPLOYMENT_TARGET');
        Future<String> xcrun(String option) async {
          final result = await Process.run('xcrun', [
            '--sdk',
            'macosx',
            option,
            if (option == '--find') 'clang',
          ]);
          if (result.exitCode != 0) {
            throw StateError(
              'Could not resolve the macOS SDK: ${result.stderr}',
            );
          }
          return (result.stdout as String).trim();
        }

        cargoEnvironment['SDKROOT'] = await xcrun('--show-sdk-path');
        final compilerKey = 'CC_${target.replaceAll('-', '_')}';
        if (!cargoEnvironment.containsKey(compilerKey) &&
            !cargoEnvironment.containsKey('CC_$target') &&
            !cargoEnvironment.containsKey('CC')) {
          cargoEnvironment[compilerKey] = await xcrun('--find');
        }
        cargoEnvironment['MACOSX_DEPLOYMENT_TARGET'] =
            (coreml || appleOutput) && config.macOS.targetVersion < 15
            ? '15.0'
            : '${config.macOS.targetVersion}.0';
      }
      final result = await Process.run(
        'cargo',
        [
          'build',
          '--manifest-path',
          manifest.toFilePath(),
          '--locked',
          '--release',
          '--package',
          'mlsr-ffi',
          if (coreml || qnn || appleOutput) ...[
            '--features',
            [
              if (coreml) 'coreml',
              if (qnn) 'qnn',
              if (appleOutput) 'apple-output',
            ].join(','),
          ],
          '--target',
          target,
          '--target-dir',
          cargoOutput.toFilePath(),
        ],
        workingDirectory: root.toFilePath(),
        environment: cargoEnvironment,
        includeParentEnvironment: false,
      );
      if (result.exitCode != 0) {
        throw StateError('Cargo failed:\n${result.stdout}\n${result.stderr}');
      }
      source = cargoOutput.resolve('$target/release/$libraryName');
      for (final name in [
        'Cargo.toml',
        'Cargo.lock',
        'rust-toolchain.toml',
        'tools/codegen/Cargo.toml',
      ]) {
        output.dependencies.add(root.resolve(name));
      }
      // Rust embeds the YAML contracts too; edits must invalidate native assets.
      for (final folder in ['crates/', 'model-manifests/']) {
        final directory = Directory.fromUri(root.resolve(folder));
        output.dependencies.add(directory.uri);
        await for (final entry in directory.list(
          recursive: true,
          followLinks: false,
        )) {
          output.dependencies.add(entry.uri);
        }
      }
    }

    final destination = input.outputDirectory.resolve(libraryName);
    await File.fromUri(source).copy(destination.toFilePath());
    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'src/bindings.g.dart',
        file: destination,
        linkMode: DynamicLoadingBundled(),
      ),
    );
  });
}
