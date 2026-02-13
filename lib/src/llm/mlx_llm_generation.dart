part of '../../mlx_ffi.dart';

typedef MlxLlmGenerationPollResult = ({String tokenText, bool isDone});

final class MlxLlmGeneration {
  MlxLlmGeneration._(this._bindings, this._raw, [this._rawStorage]);

  final MlxCBindings _bindings;
  final mlx_llm_generation _raw;
  final Pointer<mlx_llm_generation>? _rawStorage;
  bool _disposed = false;

  factory MlxLlmGeneration.start({
    required MlxCBindings bindings,
    required MlxLlmModel model,
    required String prompt,
    required MlxLlmSamplingOptions options,
  }) {
    if (prompt.isEmpty) {
      throw ArgumentError.value(prompt, 'prompt', 'Must not be empty.');
    }

    final outPtr = calloc<mlx_llm_generation>();
    var keepOutPtr = false;
    final promptPtr = prompt.toNativeUtf8();
    final nativeOptions = _NativeSamplingOptions.allocate(options);

    try {
      final code = bindings.mlx_llm_generation_start(
        outPtr,
        model.raw,
        promptPtr.cast<Char>(),
        nativeOptions.pointer,
      );
      if (code != 0) {
        final partial = outPtr.ref;
        if (partial.ctx.address != 0) {
          bindings.mlx_llm_generation_free(partial);
        }
        throw MlxException('mlx_llm_generation_start', code);
      }
      keepOutPtr = true;
      return MlxLlmGeneration._(bindings, outPtr.ref, outPtr);
    } finally {
      nativeOptions.dispose();
      calloc.free(promptPtr);
      if (!keepOutPtr) {
        calloc.free(outPtr);
      }
    }
  }

  MlxLlmGenerationPollResult next() {
    _ensureAlive();
    final tokenTextPtr = calloc<mlx_string>();
    final isDonePtr = calloc<Bool>();
    try {
      final code =
          _bindings.mlx_llm_generation_next(tokenTextPtr, isDonePtr, _raw);
      if (code != 0) {
        final partial = tokenTextPtr.ref;
        if (partial.ctx.address != 0) {
          _bindings.mlx_string_free(partial);
        }
        throw MlxException('mlx_llm_generation_next', code);
      }

      final tokenText = _takeOwnedString(_bindings, tokenTextPtr.ref);
      return (tokenText: tokenText, isDone: isDonePtr.value);
    } finally {
      calloc.free(tokenTextPtr);
      calloc.free(isDonePtr);
    }
  }

  void cancel() {
    _ensureAlive();
    final code = _bindings.mlx_llm_generation_cancel(_raw);
    if (code != 0) {
      throw MlxException('mlx_llm_generation_cancel', code);
    }
  }

  void dispose() {
    if (_disposed) {
      return;
    }
    final code = _bindings.mlx_llm_generation_free(_raw);
    if (code != 0) {
      throw MlxException('mlx_llm_generation_free', code);
    }
    final rawStorage = _rawStorage;
    if (rawStorage != null) {
      calloc.free(rawStorage);
    }
    _disposed = true;
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('MlxLlmGeneration has been disposed.');
    }
  }
}

final class _NativeSamplingOptions {
  _NativeSamplingOptions._(
    this.pointer,
    this._stopSequenceArray,
    this._stopSequencePointers,
  );

  final Pointer<mlx_llm_sampling_options> pointer;
  final Pointer<Pointer<Char>> _stopSequenceArray;
  final List<Pointer<Utf8>> _stopSequencePointers;

  static _NativeSamplingOptions allocate(MlxLlmSamplingOptions options) {
    _validateSamplingOptions(options);

    final stopSequencePointers = <Pointer<Utf8>>[];
    Pointer<Pointer<Char>> stopSequenceArray = nullptr.cast();

    if (options.stopSequences.isNotEmpty) {
      stopSequenceArray = calloc<Pointer<Char>>(options.stopSequences.length);
      for (var i = 0; i < options.stopSequences.length; i++) {
        final seqPtr = options.stopSequences[i].toNativeUtf8();
        stopSequencePointers.add(seqPtr);
        stopSequenceArray[i] = seqPtr.cast<Char>();
      }
    }

    final pointer = calloc<mlx_llm_sampling_options>();
    pointer.ref.temperature = options.temperature;
    pointer.ref.top_p = options.topP;
    pointer.ref.top_k = options.topK;
    pointer.ref.max_tokens = options.maxTokens;
    pointer.ref.repetition_penalty = options.repetitionPenalty;

    final seed = options.seed;
    if (seed != null) {
      pointer.ref.has_seed = true;
      pointer.ref.seed = seed;
    } else {
      pointer.ref.has_seed = false;
      pointer.ref.seed = 0;
    }

    pointer.ref.stop_sequences = stopSequenceArray;
    pointer.ref.stop_sequence_count = options.stopSequences.length;
    pointer.ref.stop_handlingAsInt = switch (options.stopHandlingStrategy) {
      MlxLlmStopHandlingStrategy.truncate =>
        mlx_llm_stop_handling_strategy_.MLX_LLM_STOP_HANDLING_TRUNCATE.value,
      MlxLlmStopHandlingStrategy.includeStop => mlx_llm_stop_handling_strategy_
          .MLX_LLM_STOP_HANDLING_INCLUDE_STOP.value,
    };

    return _NativeSamplingOptions._(
      pointer,
      stopSequenceArray,
      stopSequencePointers,
    );
  }

  void dispose() {
    for (final ptr in _stopSequencePointers) {
      calloc.free(ptr);
    }
    if (_stopSequenceArray != nullptr) {
      calloc.free(_stopSequenceArray);
    }
    calloc.free(pointer);
  }
}

void _validateSamplingOptions(MlxLlmSamplingOptions options) {
  if (options.temperature < 0) {
    throw ArgumentError.value(
      options.temperature,
      'options.temperature',
      'Must be >= 0.',
    );
  }
  if (options.topP <= 0 || options.topP > 1.0) {
    throw ArgumentError.value(
      options.topP,
      'options.topP',
      'Must be > 0 and <= 1.',
    );
  }
  if (options.topK < 0) {
    throw ArgumentError.value(options.topK, 'options.topK', 'Must be >= 0.');
  }
  if (options.maxTokens <= 0) {
    throw ArgumentError.value(
      options.maxTokens,
      'options.maxTokens',
      'Must be > 0.',
    );
  }
  if (options.repetitionPenalty <= 0) {
    throw ArgumentError.value(
      options.repetitionPenalty,
      'options.repetitionPenalty',
      'Must be > 0.',
    );
  }
  final seed = options.seed;
  if (seed != null && seed < 0) {
    throw ArgumentError.value(seed, 'options.seed', 'Must be >= 0.');
  }
  for (var i = 0; i < options.stopSequences.length; i++) {
    final stop = options.stopSequences[i];
    if (stop.isEmpty) {
      throw ArgumentError.value(
        stop,
        'options.stopSequences[$i]',
        'Stop sequences must be non-empty.',
      );
    }
  }
}

String _takeOwnedString(MlxCBindings bindings, mlx_string str) {
  try {
    final data = bindings.mlx_string_data(str);
    if (data.address == 0) {
      return '';
    }
    return data.cast<Utf8>().toDartString();
  } finally {
    final code = bindings.mlx_string_free(str);
    if (code != 0) {
      throw MlxException('mlx_string_free', code);
    }
  }
}
