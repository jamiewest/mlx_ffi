import 'dart:async';
import 'dart:io';

const _defaultIosMin = '16.4';
const _defaultMacosMin = '14.0';

Future<void> main(List<String> args) async {
  final options = _BuildOptions.parse(args);
  if (options.showHelp) {
    _printUsage();
    return;
  }

  if (!Platform.isMacOS) {
    stderr.writeln('This script currently supports macOS hosts only.');
    exitCode = 2;
    return;
  }

  final repoRoot = Directory.current;
  final sourceRoot = Directory(options.sourcePath(repoRoot));
  if (!sourceRoot.existsSync()) {
    stderr.writeln('mlx-c source not found: ${sourceRoot.path}');
    stderr.writeln(
      'Use --source to point to a valid mlx-c checkout (expects CMakeLists.txt).',
    );
    exitCode = 2;
    return;
  }

  await _requireTools(<String>['cmake', 'xcodebuild', 'libtool']);

  final appleBuildRoot = Directory('${repoRoot.path}/build/apple');
  final outputPath = options.outputPath(repoRoot);
  final output = Directory(outputPath);
  final headersDir = Directory('${appleBuildRoot.path}/headers');

  if (options.clean && appleBuildRoot.existsSync()) {
    stdout.writeln('Cleaning ${appleBuildRoot.path}');
    await appleBuildRoot.delete(recursive: true);
  }

  await headersDir.create(recursive: true);
  await _prepareHeaders(sourceRoot, headersDir);

  final slices = _resolveSlices(options.platforms);
  if (slices.isEmpty) {
    stderr.writeln('No valid platforms selected.');
    stderr.writeln('Use --platforms with: macos,ios,ios-sim');
    exitCode = 2;
    return;
  }

  final mergedLibs = <_Slice, String>{};
  for (final slice in slices) {
    final buildDir = Directory('${appleBuildRoot.path}/cmake-${slice.id}');

    stdout.writeln('--- Configuring ${slice.id} ---');
    await _run(
      'cmake',
      <String>[
        '-B',
        buildDir.path,
        '-G',
        'Xcode',
        ..._commonCmakeArgs(options),
        ...slice.cmakeArgs(options),
        '-S',
        sourceRoot.path,
      ],
      cwd: repoRoot.path,
    );

    stdout.writeln('--- Building ${slice.id} ---');
    await _run(
      'cmake',
      <String>[
        '--build',
        buildDir.path,
        '--config',
        'Release',
        '--',
        '-quiet',
      ],
      cwd: repoRoot.path,
    );

    stdout.writeln('--- Merging static libraries for ${slice.id} ---');
    final mergedPath = await _mergeStaticLibraries(buildDir);
    mergedLibs[slice] = mergedPath;
  }

  if (output.existsSync()) {
    await output.delete(recursive: true);
  }

  stdout.writeln('--- Creating XCFramework ---');
  final xcArgs = <String>['-create-xcframework'];
  for (final slice in slices) {
    final libPath = mergedLibs[slice]!;
    xcArgs
      ..add('-library')
      ..add(libPath)
      ..add('-headers')
      ..add(headersDir.path);
  }
  xcArgs
    ..add('-output')
    ..add(output.path);

  await _run('xcodebuild', xcArgs, cwd: repoRoot.path);

  stdout.writeln('Built XCFramework: ${output.path}');
  stdout.writeln('Selected slices: ${slices.map((s) => s.id).join(', ')}');
}

List<String> _commonCmakeArgs(_BuildOptions options) {
  return <String>[
    '-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_REQUIRED=NO',
    '-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=',
    '-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO',
    '-DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=',
    '-DBUILD_SHARED_LIBS=OFF',
    '-DMLX_C_BUILD_EXAMPLES=OFF',
    '-DMLX_C_USE_SYSTEM_MLX=${options.useSystemMlx ? 'ON' : 'OFF'}',
  ];
}

Future<void> _prepareHeaders(Directory sourceRoot, Directory headersDir) async {
  final sourceMlx = Directory('${sourceRoot.path}/mlx');
  if (!sourceMlx.existsSync()) {
    throw StateError('Missing headers directory: ${sourceMlx.path}');
  }

  final targetMlx = Directory('${headersDir.path}/mlx');
  await targetMlx.create(recursive: true);

  await for (final entity
      in sourceMlx.list(recursive: true, followLinks: false)) {
    if (entity is! File || !entity.path.endsWith('.h')) {
      continue;
    }

    if (entity.path.contains('/private/')) {
      continue;
    }

    final relativePath = entity.path.substring(sourceMlx.path.length + 1);
    final destination = File('${targetMlx.path}/$relativePath');
    await destination.parent.create(recursive: true);
    await entity.copy(destination.path);
  }

  await File('${headersDir.path}/mlxc.h').writeAsString(
    '#ifndef MLXC_UMBRELLA_H\n'
    '#define MLXC_UMBRELLA_H\n\n'
    '#include "mlx/c/mlx.h"\n\n'
    '#endif\n',
  );

  await File('${headersDir.path}/module.modulemap').writeAsString(
    'module MLXC {\n'
    '  umbrella header "mlxc.h"\n\n'
    '  link "c++"\n'
    '  link framework "Accelerate"\n'
    '  link framework "Foundation"\n'
    '  link framework "Metal"\n\n'
    '  export *\n'
    '}\n',
  );
}

Future<String> _mergeStaticLibraries(Directory buildDir) async {
  final archives = <File>[];

  await for (final entity
      in buildDir.list(recursive: true, followLinks: false)) {
    if (entity is! File || !entity.path.endsWith('.a')) {
      continue;
    }

    // Restrict to actual build products to avoid CMake scratch artifacts.
    if (!entity.path.contains('/Release')) {
      continue;
    }

    archives.add(entity);
  }

  archives.sort((a, b) => a.path.compareTo(b.path));
  if (archives.isEmpty) {
    throw StateError('No static libraries found under ${buildDir.path}.');
  }

  final mergedDir = Directory('${buildDir.path}/merged');
  await mergedDir.create(recursive: true);

  final mergedPath = '${mergedDir.path}/libmlxc_combined.a';
  if (File(mergedPath).existsSync()) {
    await File(mergedPath).delete();
  }

  final args = <String>[
    '-static',
    '-o',
    mergedPath,
    ...archives.map((f) => f.path)
  ];
  await _run('libtool', args);
  return mergedPath;
}

Future<void> _requireTools(List<String> tools) async {
  for (final tool in tools) {
    final result = await Process.run('which', <String>[tool]);
    if (result.exitCode != 0) {
      throw StateError('Required tool not found: $tool');
    }
  }
}

Future<void> _run(
  String executable,
  List<String> args, {
  String? cwd,
}) async {
  stdout.writeln('\$ $executable ${args.join(' ')}');
  final process = await Process.start(
    executable,
    args,
    runInShell: false,
    workingDirectory: cwd,
  );

  await stdout.addStream(process.stdout);
  await stderr.addStream(process.stderr);

  final code = await process.exitCode;
  if (code != 0) {
    throw ProcessException(
        executable, args, 'Command failed with exit code $code', code);
  }
}

Set<String> _parsePlatforms(String raw) {
  return raw
      .split(',')
      .map((part) => part.trim())
      .where((part) => part.isNotEmpty)
      .toSet();
}

List<_Slice> _resolveSlices(Set<String> platforms) {
  final out = <_Slice>[];
  if (platforms.contains('macos')) {
    out.add(_Slice.macos);
  }
  if (platforms.contains('ios')) {
    out.add(_Slice.iosDevice);
  }
  if (platforms.contains('ios-sim')) {
    out.add(_Slice.iosSimulator);
  }
  return out;
}

void _printUsage() {
  stdout.writeln(
      'Build MLX C into an Apple XCFramework using Dart orchestration.');
  stdout.writeln('');
  stdout.writeln('Usage:');
  stdout.writeln('  dart run tool/build_apple_xcframework.dart [options]');
  stdout.writeln('');
  stdout.writeln('Options:');
  stdout.writeln('  --source <path>      Path to mlx-c checkout');
  stdout.writeln('                       default: third_party/mlx-c');
  stdout.writeln('  --output <path>      Output XCFramework path');
  stdout
      .writeln('                       default: build/apple/MLXC.xcframework');
  stdout.writeln('  --platforms <list>   Comma-separated: macos,ios,ios-sim');
  stdout.writeln('                       default: macos,ios,ios-sim');
  stdout.writeln('  --ios-min <version>  iOS minimum deployment target');
  stdout.writeln('                       default: $_defaultIosMin');
  stdout.writeln('  --macos-min <ver>    macOS minimum deployment target');
  stdout.writeln('                       default: $_defaultMacosMin');
  stdout.writeln(
      '  --use-system-mlx     Use find_package(MLX) instead of FetchContent');
  stdout.writeln('  --no-clean           Reuse existing build/apple directory');
  stdout.writeln('  -h, --help           Show this message');
}

final class _BuildOptions {
  const _BuildOptions({
    required this.source,
    required this.output,
    required this.platforms,
    required this.iosMin,
    required this.macosMin,
    required this.useSystemMlx,
    required this.clean,
    required this.showHelp,
  });

  final String source;
  final String output;
  final Set<String> platforms;
  final String iosMin;
  final String macosMin;
  final bool useSystemMlx;
  final bool clean;
  final bool showHelp;

  String sourcePath(Directory root) {
    if (_isAbsolute(source)) {
      return source;
    }
    return '${root.path}/$source';
  }

  String outputPath(Directory root) {
    if (_isAbsolute(output)) {
      return output;
    }
    return '${root.path}/$output';
  }

  static _BuildOptions parse(List<String> args) {
    var source = 'third_party/mlx-c';
    var output = 'build/apple/MLXC.xcframework';
    var platforms = <String>{'macos', 'ios', 'ios-sim'};
    var iosMin = _defaultIosMin;
    var macosMin = _defaultMacosMin;
    var useSystemMlx = false;
    var clean = true;
    var showHelp = false;

    for (var i = 0; i < args.length; i++) {
      final arg = args[i];

      if (arg == '-h' || arg == '--help') {
        showHelp = true;
        continue;
      }

      if (arg == '--use-system-mlx') {
        useSystemMlx = true;
        continue;
      }

      if (arg == '--no-clean') {
        clean = false;
        continue;
      }

      if (arg == '--source') {
        if (i + 1 >= args.length) {
          throw ArgumentError('Missing value for --source');
        }
        source = args[++i];
        continue;
      }

      if (arg.startsWith('--source=')) {
        source = arg.substring('--source='.length);
        continue;
      }

      if (arg == '--output') {
        if (i + 1 >= args.length) {
          throw ArgumentError('Missing value for --output');
        }
        output = args[++i];
        continue;
      }

      if (arg.startsWith('--output=')) {
        output = arg.substring('--output='.length);
        continue;
      }

      if (arg == '--platforms') {
        if (i + 1 >= args.length) {
          throw ArgumentError('Missing value for --platforms');
        }
        platforms = _parsePlatforms(args[++i]);
        continue;
      }

      if (arg.startsWith('--platforms=')) {
        platforms = _parsePlatforms(arg.substring('--platforms='.length));
        continue;
      }

      if (arg == '--ios-min') {
        if (i + 1 >= args.length) {
          throw ArgumentError('Missing value for --ios-min');
        }
        iosMin = args[++i];
        continue;
      }

      if (arg.startsWith('--ios-min=')) {
        iosMin = arg.substring('--ios-min='.length);
        continue;
      }

      if (arg == '--macos-min') {
        if (i + 1 >= args.length) {
          throw ArgumentError('Missing value for --macos-min');
        }
        macosMin = args[++i];
        continue;
      }

      if (arg.startsWith('--macos-min=')) {
        macosMin = arg.substring('--macos-min='.length);
        continue;
      }

      throw ArgumentError('Unknown argument: $arg');
    }

    return _BuildOptions(
      source: source,
      output: output,
      platforms: platforms,
      iosMin: iosMin,
      macosMin: macosMin,
      useSystemMlx: useSystemMlx,
      clean: clean,
      showHelp: showHelp,
    );
  }
}

enum _Slice {
  macos('macos'),
  iosDevice('ios'),
  iosSimulator('ios-sim');

  const _Slice(this.id);

  final String id;

  List<String> cmakeArgs(_BuildOptions options) {
    switch (this) {
      case _Slice.macos:
        return <String>[
          '-DCMAKE_OSX_DEPLOYMENT_TARGET=${options.macosMin}',
          '-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64',
        ];
      case _Slice.iosDevice:
        return <String>[
          '-DCMAKE_SYSTEM_NAME=iOS',
          '-DCMAKE_OSX_DEPLOYMENT_TARGET=${options.iosMin}',
          '-DCMAKE_OSX_SYSROOT=iphoneos',
          '-DCMAKE_OSX_ARCHITECTURES=arm64',
          '-DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphoneos',
        ];
      case _Slice.iosSimulator:
        return <String>[
          '-DIOS=ON',
          '-DCMAKE_SYSTEM_NAME=iOS',
          '-DCMAKE_OSX_DEPLOYMENT_TARGET=${options.iosMin}',
          '-DCMAKE_OSX_SYSROOT=iphonesimulator',
          '-DCMAKE_OSX_ARCHITECTURES=arm64;x86_64',
          '-DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphonesimulator',
        ];
    }
  }
}

bool _isAbsolute(String path) {
  return path.startsWith('/') || RegExp(r'^[A-Za-z]:[\\/]').hasMatch(path);
}
