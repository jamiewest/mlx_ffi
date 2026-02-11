import 'dart:io';

import 'package:mlx_ffi/mlx_ffi.dart';

void main(List<String> args) {
  final options = _parseArgs(args);

  if (options.showHelp) {
    _printUsage();
    return;
  }

  try {
    final mlx = Mlx.open(
      libraryPath: options.libraryPath,
      useGpuDefaultStream: options.useGpu,
    );

    final a = mlx.fromFloat64List([1.0, 2.0, 3.0], shape: [3]);
    final b = mlx.ones([3], dtype: mlx_dtype_.MLX_FLOAT64);
    final c = a.add(b);
    final sum = c.sum();

    try {
      print('MLX version: ${mlx.version()}');
      print('a: ${a.toFloat64List()}');
      print('b: ${b.toFloat64List()}');
      print('c = a + b: ${c.toFloat64List()}');
      print('sum(c): ${sum.scalarFloat64()}');
    } finally {
      sum.dispose();
      c.dispose();
      b.dispose();
      a.dispose();
    }
  } on MlxException catch (e) {
    stderr.writeln('MLX call failed: $e');
    exitCode = 2;
  } on ArgumentError catch (e) {
    stderr.writeln('Invalid arguments: ${e.message}');
    _printUsage();
    exitCode = 64;
  } catch (e) {
    stderr.writeln('Failed to run MLX example: $e');
    stderr.writeln(
      'Tip: pass --library /absolute/path/to/libmlx.dylib if the runtime '
      'library is not on your loader path.',
    );
    exitCode = 1;
  }
}

_Options _parseArgs(List<String> args) {
  String? libraryPath;
  var useGpu = false;
  var showHelp = false;

  for (var i = 0; i < args.length; i++) {
    final arg = args[i];

    if (arg == '--help' || arg == '-h') {
      showHelp = true;
      continue;
    }

    if (arg == '--gpu') {
      useGpu = true;
      continue;
    }

    if (arg == '--library' || arg == '-l') {
      if (i + 1 >= args.length) {
        throw ArgumentError('Missing value for $arg');
      }
      libraryPath = args[++i];
      continue;
    }

    if (arg.startsWith('--library=')) {
      final value = arg.substring('--library='.length);
      if (value.isEmpty) {
        throw ArgumentError('Missing value for --library');
      }
      libraryPath = value;
      continue;
    }

    throw ArgumentError('Unknown argument: $arg');
  }

  return _Options(
    libraryPath: libraryPath,
    useGpu: useGpu,
    showHelp: showHelp,
  );
}

void _printUsage() {
  print('Usage: dart run bin/mlx_cli_example.dart [options]');
  print('');
  print('Options:');
  print('  -l, --library <path>  Path to libmlx dynamic library');
  print('      --gpu             Use default GPU stream instead of CPU');
  print('  -h, --help            Show this help');
}

final class _Options {
  const _Options({
    required this.libraryPath,
    required this.useGpu,
    required this.showHelp,
  });

  final String? libraryPath;
  final bool useGpu;
  final bool showHelp;
}
