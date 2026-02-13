part of '../../mlx_ffi.dart';

final class MlxLlmModel {
  MlxLlmModel._(this._bindings, this._raw, [this._rawStorage]);

  final MlxCBindings _bindings;
  final mlx_llm_model _raw;
  final Pointer<mlx_llm_model>? _rawStorage;
  bool _disposed = false;

  factory MlxLlmModel.load({
    required MlxCBindings bindings,
    required String modelDirectory,
    required mlx_stream stream,
  }) {
    if (modelDirectory.isEmpty) {
      throw ArgumentError.value(
        modelDirectory,
        'modelDirectory',
        'Must not be empty.',
      );
    }

    final outPtr = calloc<mlx_llm_model>();
    var keepOutPtr = false;
    final dirPtr = modelDirectory.toNativeUtf8();
    try {
      final code =
          bindings.mlx_llm_model_load(outPtr, dirPtr.cast<Char>(), stream);
      if (code != 0) {
        final partial = outPtr.ref;
        if (partial.ctx.address != 0) {
          bindings.mlx_llm_model_free(partial);
        }
        throw MlxException('mlx_llm_model_load', code);
      }
      keepOutPtr = true;
      return MlxLlmModel._(bindings, outPtr.ref, outPtr);
    } finally {
      calloc.free(dirPtr);
      if (!keepOutPtr) {
        calloc.free(outPtr);
      }
    }
  }

  mlx_llm_model get raw {
    _ensureAlive();
    return _raw;
  }

  void dispose() {
    if (_disposed) {
      return;
    }
    final code = _bindings.mlx_llm_model_free(_raw);
    if (code != 0) {
      throw MlxException('mlx_llm_model_free', code);
    }
    final rawStorage = _rawStorage;
    if (rawStorage != null) {
      calloc.free(rawStorage);
    }
    _disposed = true;
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('MlxLlmModel has been disposed.');
    }
  }
}
