part of '../../mlx_ffi.dart';

/// High-level LLM runtime wrapper on top of generated low-level bindings.
final class MlxLlm {
  MlxLlm._(
    this.dynamicLibrary,
    this.bindings,
    this.defaultStream,
    this._model,
  );

  final DynamicLibrary dynamicLibrary;
  final MlxCBindings bindings;
  final mlx_stream defaultStream;
  final MlxLlmModel _model;

  MlxLlmGeneration? _activeGeneration;
  bool _disposed = false;

  factory MlxLlm.open({
    required String modelDirectory,
    String? libraryPath,
    bool useGpuDefaultStream = false,
  }) {
    final dylib = Mlx._openDynamicLibrary(libraryPath);
    final bindings = MlxCBindings(dylib);
    final stream = useGpuDefaultStream
        ? bindings.mlx_default_gpu_stream_new()
        : bindings.mlx_default_cpu_stream_new();
    final model = MlxLlmModel.load(
      bindings: bindings,
      modelDirectory: modelDirectory,
      stream: stream,
    );
    return MlxLlm._(dylib, bindings, stream, model);
  }

  List<int> tokenize(String text, {bool addBos = true, bool addEos = false}) {
    _ensureAlive();

    final outPtr = calloc<mlx_vector_int>();
    final textPtr = text.toNativeUtf8();
    try {
      final code = bindings.mlx_llm_tokenize(
        outPtr,
        _model.raw,
        textPtr.cast<Char>(),
        addBos,
        addEos,
      );
      if (code != 0) {
        final partial = outPtr.ref;
        if (partial.ctx.address != 0) {
          bindings.mlx_vector_int_free(partial);
          outPtr.ref.ctx = nullptr;
        }
        throw MlxException('mlx_llm_tokenize', code);
      }

      final vector = outPtr.ref;
      final size = bindings.mlx_vector_int_size(vector);
      final result = List<int>.filled(size, 0, growable: false);
      final valuePtr = calloc<Int>();
      try {
        for (var i = 0; i < size; i++) {
          final getCode = bindings.mlx_vector_int_get(valuePtr, vector, i);
          if (getCode != 0) {
            throw MlxException('mlx_vector_int_get', getCode);
          }
          result[i] = valuePtr.value;
        }
      } finally {
        calloc.free(valuePtr);
      }

      return result;
    } finally {
      final vector = outPtr.ref;
      if (vector.ctx.address != 0) {
        final freeCode = bindings.mlx_vector_int_free(vector);
        if (freeCode != 0) {
          throw MlxException('mlx_vector_int_free', freeCode);
        }
      }
      calloc.free(textPtr);
      calloc.free(outPtr);
    }
  }

  String decodeTokens(List<int> tokens) {
    _ensureAlive();

    Pointer<Int> tokenPtr = nullptr;
    if (tokens.isNotEmpty) {
      tokenPtr = calloc<Int>(tokens.length);
      for (var i = 0; i < tokens.length; i++) {
        tokenPtr[i] = tokens[i];
      }
    }

    final outPtr = calloc<mlx_string>();
    try {
      final code = bindings.mlx_llm_decode(
        outPtr,
        _model.raw,
        tokenPtr,
        tokens.length,
      );
      if (code != 0) {
        final partial = outPtr.ref;
        if (partial.ctx.address != 0) {
          bindings.mlx_string_free(partial);
        }
        throw MlxException('mlx_llm_decode', code);
      }

      return _takeOwnedString(bindings, outPtr.ref);
    } finally {
      if (tokenPtr != nullptr) {
        calloc.free(tokenPtr);
      }
      calloc.free(outPtr);
    }
  }

  Stream<String> generateStream({
    required String prompt,
    MlxLlmSamplingOptions options = const MlxLlmSamplingOptions(),
  }) async* {
    _ensureAlive();
    if (_activeGeneration != null) {
      throw StateError('A generation is already active for this model.');
    }

    final generation = MlxLlmGeneration.start(
      bindings: bindings,
      model: _model,
      prompt: prompt,
      options: options,
    );
    _activeGeneration = generation;
    try {
      while (true) {
        final next = generation.next();
        if (next.tokenText.isNotEmpty) {
          yield next.tokenText;
        }
        if (next.isDone) {
          break;
        }
      }
    } finally {
      try {
        generation.cancel();
      } catch (_) {
        // Ignore cancel errors during teardown.
      }
      try {
        generation.dispose();
      } finally {
        if (identical(_activeGeneration, generation)) {
          _activeGeneration = null;
        }
      }
    }
  }

  void dispose() {
    if (_disposed) {
      return;
    }

    _disposed = true;
    Object? firstError;

    final generation = _activeGeneration;
    _activeGeneration = null;
    if (generation != null) {
      try {
        generation.cancel();
      } catch (e) {
        firstError ??= e;
      }
      try {
        generation.dispose();
      } catch (e) {
        firstError ??= e;
      }
    }

    try {
      _model.dispose();
    } catch (e) {
      firstError ??= e;
    }

    if (firstError != null) {
      throw firstError;
    }
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('MlxLlm has been disposed.');
    }
  }
}
