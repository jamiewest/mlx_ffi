part of '../../mlx_ffi.dart';

/// Stop-sequence handling mode during generation.
enum MlxLlmStopHandlingStrategy {
  truncate,
  includeStop,
}

/// Sampling options for text generation.
final class MlxLlmSamplingOptions {
  const MlxLlmSamplingOptions({
    this.temperature = 0.8,
    this.topP = 0.95,
    this.topK = 40,
    this.maxTokens = 256,
    this.repetitionPenalty = 1.0,
    this.seed,
    this.stopSequences = const <String>[],
    this.stopHandlingStrategy = MlxLlmStopHandlingStrategy.truncate,
  });

  final double temperature;
  final double topP;
  final int topK;
  final int maxTokens;
  final double repetitionPenalty;
  final int? seed;
  final List<String> stopSequences;
  final MlxLlmStopHandlingStrategy stopHandlingStrategy;
}
