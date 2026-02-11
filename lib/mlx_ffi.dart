import 'dart:ffi';
import 'dart:io';

import 'package:ffi/ffi.dart';

import 'src/generated/mlx_bindings.dart';

export 'src/generated/mlx_bindings.dart';

final class MlxException implements Exception {
  MlxException(this.operation, this.code, [this.message]);

  final String operation;
  final int code;
  final String? message;

  @override
  String toString() {
    if (message != null && message!.isNotEmpty) {
      return 'MlxException($operation): code=$code, message=$message';
    }
    return 'MlxException($operation): code=$code';
  }
}

/// High-level MLX runtime wrapper on top of generated low-level bindings.
final class Mlx {
  Mlx._(this.dynamicLibrary, this.bindings, this.defaultStream);

  final DynamicLibrary dynamicLibrary;
  final MlxCBindings bindings;
  final mlx_stream defaultStream;

  factory Mlx.open({String? libraryPath, bool useGpuDefaultStream = false}) {
    final dylib = _openDynamicLibrary(libraryPath);
    final bindings = MlxCBindings(dylib);
    final stream = useGpuDefaultStream
        ? bindings.mlx_default_gpu_stream_new()
        : bindings.mlx_default_cpu_stream_new();
    return Mlx._(dylib, bindings, stream);
  }

  static DynamicLibrary _openDynamicLibrary(String? overridePath) {
    if (overridePath != null && overridePath.isNotEmpty) {
      return DynamicLibrary.open(overridePath);
    }
    if (Platform.isMacOS) {
      try {
        return DynamicLibrary.open('libmlxc.dylib');
      } on ArgumentError {
        // Backward-compatible fallback for older naming.
        return DynamicLibrary.open('libmlx.dylib');
      }
    }
    if (Platform.isLinux) {
      return DynamicLibrary.open('libmlx.so');
    }
    if (Platform.isWindows) {
      return DynamicLibrary.open('mlx.dll');
    }
    throw UnsupportedError('Unsupported platform: ${Platform.operatingSystem}');
  }

  String version() {
    final strPtr = calloc<mlx_string>();
    try {
      _check(bindings.mlx_version(strPtr), 'mlx_version');
      return _takeString(strPtr.ref);
    } finally {
      calloc.free(strPtr);
    }
  }

  MlxArray scalarBool(bool value) =>
      MlxArray._(this, bindings.mlx_array_new_bool(value));

  MlxArray scalarInt(int value) =>
      MlxArray._(this, bindings.mlx_array_new_int(value));

  MlxArray scalarFloat32(double value) =>
      MlxArray._(this, bindings.mlx_array_new_float32(value));

  MlxArray scalarFloat64(double value) =>
      MlxArray._(this, bindings.mlx_array_new_float64(value));

  MlxArray fromFloat64List(List<double> values, {List<int>? shape}) {
    if (values.isEmpty) {
      throw ArgumentError.value(values, 'values', 'Must not be empty.');
    }
    final resolvedShape = shape ?? <int>[values.length];
    _validateShape(resolvedShape);
    final expected = _shapeElementCount(resolvedShape);
    if (expected != values.length) {
      throw ArgumentError(
        'Shape $resolvedShape expects $expected elements, got ${values.length}.',
      );
    }

    final dataPtr = calloc<Double>(values.length);
    final shapePtr = _allocShape(resolvedShape);
    try {
      for (var i = 0; i < values.length; i++) {
        dataPtr[i] = values[i];
      }
      final array = bindings.mlx_array_new_data(
        dataPtr.cast<Void>(),
        shapePtr,
        resolvedShape.length,
        mlx_dtype_.MLX_FLOAT64,
      );
      return MlxArray._(this, array);
    } finally {
      calloc.free(dataPtr);
      calloc.free(shapePtr);
    }
  }

  MlxArray zeros(List<int> shape, {mlx_dtype_ dtype = mlx_dtype_.MLX_FLOAT32}) {
    return _runShapeResult(
      'mlx_zeros',
      shape,
      (out, shapePtr, rank) =>
          bindings.mlx_zeros(out, shapePtr, rank, dtype, defaultStream),
    );
  }

  MlxArray ones(List<int> shape, {mlx_dtype_ dtype = mlx_dtype_.MLX_FLOAT32}) {
    return _runShapeResult(
      'mlx_ones',
      shape,
      (out, shapePtr, rank) =>
          bindings.mlx_ones(out, shapePtr, rank, dtype, defaultStream),
    );
  }

  MlxArray arange(
    double start,
    double stop,
    double step, {
    mlx_dtype_ dtype = mlx_dtype_.MLX_FLOAT32,
  }) {
    return _runArrayResult(
      'mlx_arange',
      (out) =>
          bindings.mlx_arange(out, start, stop, step, dtype, defaultStream),
    );
  }

  MlxArray linspace(
    double start,
    double stop,
    int num, {
    mlx_dtype_ dtype = mlx_dtype_.MLX_FLOAT32,
  }) {
    return _runArrayResult(
      'mlx_linspace',
      (out) =>
          bindings.mlx_linspace(out, start, stop, num, dtype, defaultStream),
    );
  }

  MlxArray add(MlxArray a, MlxArray b) {
    _requireOwned(a);
    _requireOwned(b);
    return _runArrayResult(
      'mlx_add',
      (out) => bindings.mlx_add(out, a._raw, b._raw, defaultStream),
    );
  }

  MlxArray subtract(MlxArray a, MlxArray b) {
    _requireOwned(a);
    _requireOwned(b);
    return _runArrayResult(
      'mlx_subtract',
      (out) => bindings.mlx_subtract(out, a._raw, b._raw, defaultStream),
    );
  }

  MlxArray multiply(MlxArray a, MlxArray b) {
    _requireOwned(a);
    _requireOwned(b);
    return _runArrayResult(
      'mlx_multiply',
      (out) => bindings.mlx_multiply(out, a._raw, b._raw, defaultStream),
    );
  }

  MlxArray divide(MlxArray a, MlxArray b) {
    _requireOwned(a);
    _requireOwned(b);
    return _runArrayResult(
      'mlx_divide',
      (out) => bindings.mlx_divide(out, a._raw, b._raw, defaultStream),
    );
  }

  MlxArray matmul(MlxArray a, MlxArray b) {
    _requireOwned(a);
    _requireOwned(b);
    return _runArrayResult(
      'mlx_matmul',
      (out) => bindings.mlx_matmul(out, a._raw, b._raw, defaultStream),
    );
  }

  MlxArray reshape(MlxArray array, List<int> shape) {
    _requireOwned(array);
    return _runShapeResult(
      'mlx_reshape',
      shape,
      (out, shapePtr, rank) =>
          bindings.mlx_reshape(out, array._raw, shapePtr, rank, defaultStream),
    );
  }

  MlxArray astype(MlxArray array, mlx_dtype_ dtype) {
    _requireOwned(array);
    return _runArrayResult(
      'mlx_astype',
      (out) => bindings.mlx_astype(out, array._raw, dtype, defaultStream),
    );
  }

  MlxArray sum(MlxArray array, {bool keepdims = false}) {
    _requireOwned(array);
    return _runArrayResult(
      'mlx_sum',
      (out) => bindings.mlx_sum(out, array._raw, keepdims, defaultStream),
    );
  }

  MlxArray mean(MlxArray array, {bool keepdims = false}) {
    _requireOwned(array);
    return _runArrayResult(
      'mlx_mean',
      (out) => bindings.mlx_mean(out, array._raw, keepdims, defaultStream),
    );
  }

  void synchronize() {
    _check(bindings.mlx_synchronize(defaultStream), 'mlx_synchronize');
  }

  List<double> toFloat64List(MlxArray array) {
    _requireOwned(array);
    final source = array.dtype == mlx_dtype_.MLX_FLOAT64
        ? array
        : astype(array, mlx_dtype_.MLX_FLOAT64);
    final ownsSource = !identical(source, array);
    try {
      source.eval();
      synchronize();
      final length = source.size;
      final ptr = bindings.mlx_array_data_float64(source._raw);
      if (ptr.address == 0 && length > 0) {
        throw StateError('mlx_array_data_float64 returned null pointer.');
      }
      return List<double>.generate(length, (index) => ptr[index],
          growable: false);
    } finally {
      if (ownsSource) {
        source.dispose();
      }
    }
  }

  String describe(MlxArray array) {
    _requireOwned(array);
    final strPtr = calloc<mlx_string>();
    try {
      _check(bindings.mlx_array_tostring(strPtr, array._raw),
          'mlx_array_tostring');
      return _takeString(strPtr.ref);
    } finally {
      calloc.free(strPtr);
    }
  }

  void _requireOwned(MlxArray array) {
    if (!identical(array._owner, this)) {
      throw ArgumentError('Array belongs to a different Mlx instance.');
    }
    array._ensureAlive();
  }

  MlxArray _runShapeResult(
    String opName,
    List<int> shape,
    int Function(Pointer<mlx_array>, Pointer<Int>, int) invoke,
  ) {
    _validateShape(shape);
    final shapePtr = _allocShape(shape);
    try {
      return _runArrayResult(
          opName, (out) => invoke(out, shapePtr, shape.length));
    } finally {
      calloc.free(shapePtr);
    }
  }

  MlxArray _runArrayResult(
    String opName,
    int Function(Pointer<mlx_array>) invoke,
  ) {
    final outPtr = calloc<mlx_array>();
    final code = invoke(outPtr);
    if (code != 0) {
      final partial = outPtr.ref;
      if (partial.ctx.address != 0) {
        bindings.mlx_array_free(partial);
      }
      calloc.free(outPtr);
      throw MlxException(opName, code);
    }
    return MlxArray._(this, outPtr.ref, outPtr);
  }

  String _takeString(mlx_string str) {
    try {
      final data = bindings.mlx_string_data(str);
      if (data.address == 0) {
        return '';
      }
      return data.cast<Utf8>().toDartString();
    } finally {
      _check(bindings.mlx_string_free(str), 'mlx_string_free');
    }
  }

  Pointer<Int> _allocShape(List<int> shape) {
    final shapePtr = calloc<Int>(shape.length);
    for (var i = 0; i < shape.length; i++) {
      shapePtr[i] = shape[i];
    }
    return shapePtr;
  }

  void _validateShape(List<int> shape) {
    if (shape.isEmpty) {
      throw ArgumentError.value(shape, 'shape', 'Shape must not be empty.');
    }
    for (final dim in shape) {
      if (dim < 0) {
        throw ArgumentError.value(
            shape, 'shape', 'Shape dimensions must be >= 0.');
      }
    }
  }

  int _shapeElementCount(List<int> shape) {
    var elements = 1;
    for (final dim in shape) {
      elements *= dim;
    }
    return elements;
  }

  void _check(int code, String operation) {
    if (code != 0) {
      throw MlxException(operation, code);
    }
  }
}

final class MlxArray {
  MlxArray._(this._owner, this._raw, [this._rawStorage]);

  final Mlx _owner;
  final mlx_array _raw;
  final Pointer<mlx_array>? _rawStorage;
  bool _disposed = false;

  mlx_array get raw {
    _ensureAlive();
    return _raw;
  }

  int get ndim {
    _ensureAlive();
    return _owner.bindings.mlx_array_ndim(_raw);
  }

  int get size {
    _ensureAlive();
    return _owner.bindings.mlx_array_size(_raw);
  }

  int get itemSize {
    _ensureAlive();
    return _owner.bindings.mlx_array_itemsize(_raw);
  }

  int get nbytes {
    _ensureAlive();
    return _owner.bindings.mlx_array_nbytes(_raw);
  }

  mlx_dtype_ get dtype {
    _ensureAlive();
    return _owner.bindings.mlx_array_dtype(_raw);
  }

  List<int> get shape {
    _ensureAlive();
    final rank = ndim;
    final ptr = _owner.bindings.mlx_array_shape(_raw);
    return List<int>.generate(rank, (index) => ptr[index], growable: false);
  }

  void eval() {
    _ensureAlive();
    _owner._check(_owner.bindings.mlx_array_eval(_raw), 'mlx_array_eval');
  }

  void dispose() {
    if (_disposed) {
      return;
    }
    _owner._check(_owner.bindings.mlx_array_free(_raw), 'mlx_array_free');
    final rawStorage = _rawStorage;
    if (rawStorage != null) {
      calloc.free(rawStorage);
    }
    _disposed = true;
  }

  MlxArray reshape(List<int> newShape) => _owner.reshape(this, newShape);

  MlxArray astype(mlx_dtype_ newDtype) => _owner.astype(this, newDtype);

  MlxArray add(MlxArray other) => _owner.add(this, other);

  MlxArray subtract(MlxArray other) => _owner.subtract(this, other);

  MlxArray multiply(MlxArray other) => _owner.multiply(this, other);

  MlxArray divide(MlxArray other) => _owner.divide(this, other);

  MlxArray matmul(MlxArray other) => _owner.matmul(this, other);

  MlxArray sum({bool keepdims = false}) => _owner.sum(this, keepdims: keepdims);

  MlxArray mean({bool keepdims = false}) =>
      _owner.mean(this, keepdims: keepdims);

  double scalarFloat64() {
    _requireScalar();
    eval();
    _owner.synchronize();
    final out = calloc<Double>();
    try {
      _owner._check(_owner.bindings.mlx_array_item_float64(out, _raw),
          'mlx_array_item_float64');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  double scalarFloat32() {
    _requireScalar();
    eval();
    _owner.synchronize();
    final out = calloc<Float>();
    try {
      _owner._check(_owner.bindings.mlx_array_item_float32(out, _raw),
          'mlx_array_item_float32');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  int scalarInt32() {
    _requireScalar();
    eval();
    _owner.synchronize();
    final out = calloc<Int32>();
    try {
      _owner._check(_owner.bindings.mlx_array_item_int32(out, _raw),
          'mlx_array_item_int32');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  bool scalarBool() {
    _requireScalar();
    eval();
    _owner.synchronize();
    final out = calloc<Bool>();
    try {
      _owner._check(_owner.bindings.mlx_array_item_bool(out, _raw),
          'mlx_array_item_bool');
      return out.value;
    } finally {
      calloc.free(out);
    }
  }

  List<double> toFloat64List() => _owner.toFloat64List(this);

  String describe() => _owner.describe(this);

  @override
  String toString() {
    try {
      return describe();
    } catch (_) {
      return 'MlxArray(shape: $shape, dtype: $dtype)';
    }
  }

  void _requireScalar() {
    _ensureAlive();
    if (size != 1) {
      throw StateError('Expected scalar array, got shape $shape.');
    }
  }

  void _ensureAlive() {
    if (_disposed) {
      throw StateError('MlxArray has been disposed.');
    }
  }
}
