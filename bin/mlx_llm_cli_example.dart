import 'dart:io';

import 'package:mlx_ffi/mlx_ffi.dart';

Future<void> main(List<String> args) async {
  final options = _parseArgs(args);
  if (options.showHelp) {
    _printUsage();
    return;
  }

  if (options.modelDirectory == null || options.modelDirectory!.isEmpty) {
    stderr.writeln('Missing required --model-dir.');
    _printUsage();
    exitCode = 64;
    return;
  }
  if (options.prompt == null || options.prompt!.isEmpty) {
    stderr.writeln('Missing required --prompt.');
    _printUsage();
    exitCode = 64;
    return;
  }

  try {
    final llm = MlxLlm.open(
      libraryPath: options.libraryPath,
      modelDirectory: options.modelDirectory!,
      useGpuDefaultStream: options.useGpu,
    );

    final samplingOptions = MlxLlmSamplingOptions(
      temperature: options.temperature,
      topP: options.topP,
      topK: options.topK,
      maxTokens: options.maxTokens,
      repetitionPenalty: options.repetitionPenalty,
      seed: options.seed,
      stopSequences: options.stopSequences,
      stopHandlingStrategy: options.includeStop
          ? MlxLlmStopHandlingStrategy.includeStop
          : MlxLlmStopHandlingStrategy.truncate,
    );

    try {
      await for (final token in llm.generateStream(
        prompt: options.prompt!,
        options: samplingOptions,
      )) {
        stdout.write(token);
      }
      stdout.writeln();
    } finally {
      llm.dispose();
    }
  } on MlxException catch (e) {
    stderr.writeln('MLX LLM call failed: $e');
    exitCode = 2;
  } on ArgumentError catch (e) {
    stderr.writeln('Invalid arguments: ${e.message}');
    _printUsage();
    exitCode = 64;
  } catch (e) {
    stderr.writeln('Failed to run MLX LLM example: $e');
    stderr.writeln(
      'Tip: pass --library /absolute/path/to/libmlxc.dylib if needed.',
    );
    exitCode = 1;
  }
}

_Options _parseArgs(List<String> args) {
  String? libraryPath;
  String? modelDirectory;
  String? prompt;
  var useGpu = false;
  var showHelp = false;
  var includeStop = false;
  var temperature = 0.8;
  var topP = 0.95;
  var topK = 40;
  var maxTokens = 256;
  var repetitionPenalty = 1.0;
  int? seed;
  final stopSequences = <String>[];

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
    if (arg == '--include-stop') {
      includeStop = true;
      continue;
    }

    if (arg == '--library' || arg == '-l') {
      libraryPath = _readValue(args, ++i, arg);
      continue;
    }
    if (arg == '--model-dir') {
      modelDirectory = _readValue(args, ++i, arg);
      continue;
    }
    if (arg == '--prompt') {
      prompt = _readValue(args, ++i, arg);
      continue;
    }
    if (arg == '--temperature') {
      temperature = double.parse(_readValue(args, ++i, arg));
      continue;
    }
    if (arg == '--top-p') {
      topP = double.parse(_readValue(args, ++i, arg));
      continue;
    }
    if (arg == '--top-k') {
      topK = int.parse(_readValue(args, ++i, arg));
      continue;
    }
    if (arg == '--max-tokens') {
      maxTokens = int.parse(_readValue(args, ++i, arg));
      continue;
    }
    if (arg == '--repetition-penalty') {
      repetitionPenalty = double.parse(_readValue(args, ++i, arg));
      continue;
    }
    if (arg == '--seed') {
      seed = int.parse(_readValue(args, ++i, arg));
      continue;
    }
    if (arg == '--stop') {
      stopSequences.add(_readValue(args, ++i, arg));
      continue;
    }

    if (arg.startsWith('--library=')) {
      libraryPath = arg.substring('--library='.length);
      continue;
    }
    if (arg.startsWith('--model-dir=')) {
      modelDirectory = arg.substring('--model-dir='.length);
      continue;
    }
    if (arg.startsWith('--prompt=')) {
      prompt = arg.substring('--prompt='.length);
      continue;
    }
    if (arg.startsWith('--temperature=')) {
      temperature = double.parse(arg.substring('--temperature='.length));
      continue;
    }
    if (arg.startsWith('--top-p=')) {
      topP = double.parse(arg.substring('--top-p='.length));
      continue;
    }
    if (arg.startsWith('--top-k=')) {
      topK = int.parse(arg.substring('--top-k='.length));
      continue;
    }
    if (arg.startsWith('--max-tokens=')) {
      maxTokens = int.parse(arg.substring('--max-tokens='.length));
      continue;
    }
    if (arg.startsWith('--repetition-penalty=')) {
      repetitionPenalty =
          double.parse(arg.substring('--repetition-penalty='.length));
      continue;
    }
    if (arg.startsWith('--seed=')) {
      seed = int.parse(arg.substring('--seed='.length));
      continue;
    }
    if (arg.startsWith('--stop=')) {
      stopSequences.add(arg.substring('--stop='.length));
      continue;
    }

    throw ArgumentError('Unknown argument: $arg');
  }

  return _Options(
    libraryPath: libraryPath,
    modelDirectory: modelDirectory,
    prompt: prompt,
    useGpu: useGpu,
    showHelp: showHelp,
    includeStop: includeStop,
    temperature: temperature,
    topP: topP,
    topK: topK,
    maxTokens: maxTokens,
    repetitionPenalty: repetitionPenalty,
    seed: seed,
    stopSequences: stopSequences,
  );
}

String _readValue(List<String> args, int index, String flag) {
  if (index >= args.length || args[index].startsWith('-')) {
    throw ArgumentError('Missing value for $flag');
  }
  return args[index];
}

void _printUsage() {
  print('Usage: dart run bin/mlx_llm_cli_example.dart [options]');
  print('');
  print('Required:');
  print('  --model-dir <path>       Local HF MLX model folder path');
  print('  --prompt <text>          Prompt text');
  print('');
  print('Options:');
  print('  -l, --library <path>     Path to MLX C dynamic library');
  print('      --gpu                Use default GPU stream');
  print('      --temperature <f>    Sampling temperature (default: 0.8)');
  print('      --top-p <f>          Nucleus sampling top-p (default: 0.95)');
  print('      --top-k <n>          Top-k sampling (default: 40)');
  print('      --max-tokens <n>     Max generated tokens (default: 256)');
  print('      --repetition-penalty <f>  Repetition penalty (default: 1.0)');
  print('      --seed <n>           Optional random seed');
  print('      --stop <text>        Stop sequence (repeatable)');
  print('      --include-stop       Include stop sequence text in output');
  print('  -h, --help               Show this help');
}

final class _Options {
  const _Options({
    required this.libraryPath,
    required this.modelDirectory,
    required this.prompt,
    required this.useGpu,
    required this.showHelp,
    required this.includeStop,
    required this.temperature,
    required this.topP,
    required this.topK,
    required this.maxTokens,
    required this.repetitionPenalty,
    required this.seed,
    required this.stopSequences,
  });

  final String? libraryPath;
  final String? modelDirectory;
  final String? prompt;
  final bool useGpu;
  final bool showHelp;
  final bool includeStop;
  final double temperature;
  final double topP;
  final int topK;
  final int maxTokens;
  final double repetitionPenalty;
  final int? seed;
  final List<String> stopSequences;
}
