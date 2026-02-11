import 'dart:io';

import 'package:mlx_ffi/mlx_ffi.dart';
import 'package:test/test.dart';

void main() {
  late Directory tempDir;
  late String libraryPath;
  late Mlx mlx;

  setUpAll(() async {
    if (Platform.isWindows) {
      return;
    }

    tempDir = await Directory.systemTemp.createTemp('mlx_ffi_test_');
    libraryPath = await _buildFakeMlxLibrary(tempDir);
    mlx = Mlx.open(libraryPath: libraryPath);
  });

  tearDownAll(() async {
    if (Platform.isWindows) {
      return;
    }

    await tempDir.delete(recursive: true);
  });

  test('version returns string from native lib', () {
    if (Platform.isWindows) {
      return;
    }
    expect(mlx.version(), 'fake-mlx-1.0');
  });

  test('fromFloat64List exposes metadata and data', () {
    if (Platform.isWindows) {
      return;
    }

    final array = mlx.fromFloat64List([1.0, 2.0, 3.0, 4.0], shape: [2, 2]);
    addTearDown(array.dispose);

    expect(array.dtype, mlx_dtype_.MLX_FLOAT64);
    expect(array.ndim, 2);
    expect(array.shape, [2, 2]);
    expect(array.size, 4);
    expect(array.itemSize, 8);
    expect(array.nbytes, 32);
    expect(array.toFloat64List(), [1.0, 2.0, 3.0, 4.0]);
  });

  test('scalar ops and reductions work', () {
    if (Platform.isWindows) {
      return;
    }

    final a = mlx.scalarFloat64(2.0);
    final b = mlx.scalarFloat64(3.0);
    final sum = mlx.add(a, b);
    final values = mlx.fromFloat64List([1.0, 2.0, 3.0, 4.0]);
    final reducedSum = values.sum();
    final reducedMean = values.mean();

    addTearDown(a.dispose);
    addTearDown(b.dispose);
    addTearDown(sum.dispose);
    addTearDown(values.dispose);
    addTearDown(reducedSum.dispose);
    addTearDown(reducedMean.dispose);

    expect(sum.scalarFloat64(), 5.0);
    expect(reducedSum.scalarFloat64(), 10.0);
    expect(reducedMean.scalarFloat64(), 2.5);
  });

  test('reshape changes dimensions but preserves data', () {
    if (Platform.isWindows) {
      return;
    }

    final arr = mlx.fromFloat64List([1.0, 2.0, 3.0, 4.0], shape: [2, 2]);
    final reshaped = arr.reshape([4, 1]);

    addTearDown(arr.dispose);
    addTearDown(reshaped.dispose);

    expect(reshaped.shape, [4, 1]);
    expect(reshaped.toFloat64List(), [1.0, 2.0, 3.0, 4.0]);
  });

  test('native non-zero return code throws MlxException', () {
    if (Platform.isWindows) {
      return;
    }

    final a = mlx.scalarFloat64(2.0);
    final b = mlx.scalarFloat64(3.0);
    addTearDown(a.dispose);
    addTearDown(b.dispose);

    expect(
      () => mlx.divide(a, b),
      throwsA(
        isA<MlxException>()
            .having((e) => e.operation, 'operation', 'mlx_divide')
            .having((e) => e.code, 'code', 7),
      ),
    );
  });

  test('using disposed array throws StateError', () {
    if (Platform.isWindows) {
      return;
    }

    final array = mlx.scalarFloat64(42.0);
    array.dispose();

    expect(() => array.size, throwsStateError);
  });
}

Future<String> _buildFakeMlxLibrary(Directory tempDir) async {
  final cPath = '${tempDir.path}/fake_mlx.c';
  final outPath = Platform.isMacOS
      ? '${tempDir.path}/libfake_mlx.dylib'
      : '${tempDir.path}/libfake_mlx.so';

  await File(cPath).writeAsString(_fakeMlxCSource);

  final args = Platform.isMacOS
      ? <String>['-dynamiclib', '-fPIC', cPath, '-o', outPath]
      : <String>['-shared', '-fPIC', cPath, '-o', outPath];

  final result = await Process.run('cc', args);
  if (result.exitCode != 0) {
    throw StateError(
      'Failed to compile fake mlx library.\n'
      'stdout:\n${result.stdout}\n'
      'stderr:\n${result.stderr}',
    );
  }

  return outPath;
}

const _fakeMlxCSource = r'''
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { void* ctx; } mlx_string;
typedef struct { void* ctx; } mlx_array;
typedef struct { void* ctx; } mlx_stream;

enum { MLX_FLOAT64 = 11 };

typedef struct {
  char* data;
} string_payload;

typedef struct {
  uint32_t dtype;
  int ndim;
  int* shape;
  size_t size;
  double* data;
} array_payload;

static size_t shape_size(const int* shape, int ndim) {
  size_t n = 1;
  for (int i = 0; i < ndim; i++) {
    n *= (size_t)shape[i];
  }
  return n;
}

static mlx_array make_float64_array_copy(const double* src, const int* shape, int ndim) {
  mlx_array out;
  out.ctx = NULL;

  array_payload* payload = (array_payload*)calloc(1, sizeof(array_payload));
  if (!payload) {
    return out;
  }

  payload->dtype = MLX_FLOAT64;
  payload->ndim = ndim;
  payload->shape = (int*)calloc((size_t)ndim, sizeof(int));
  if (!payload->shape) {
    free(payload);
    return out;
  }
  for (int i = 0; i < ndim; i++) {
    payload->shape[i] = shape[i];
  }

  payload->size = shape_size(shape, ndim);
  payload->data = (double*)calloc(payload->size, sizeof(double));
  if (!payload->data) {
    free(payload->shape);
    free(payload);
    return out;
  }

  memcpy(payload->data, src, payload->size * sizeof(double));
  out.ctx = payload;
  return out;
}

static int set_string(mlx_string* out, const char* text) {
  string_payload* payload = (string_payload*)calloc(1, sizeof(string_payload));
  if (!payload) {
    return 1;
  }

  size_t len = strlen(text);
  payload->data = (char*)calloc(len + 1, sizeof(char));
  if (!payload->data) {
    free(payload);
    return 1;
  }

  memcpy(payload->data, text, len + 1);
  out->ctx = payload;
  return 0;
}

mlx_stream mlx_default_cpu_stream_new(void) {
  mlx_stream s;
  s.ctx = (void*)0x1;
  return s;
}

int mlx_synchronize(mlx_stream stream) {
  (void)stream;
  return 0;
}

int mlx_version(mlx_string* str_) {
  return set_string(str_, "fake-mlx-1.0");
}

char* mlx_string_data(mlx_string str) {
  string_payload* payload = (string_payload*)str.ctx;
  if (!payload) {
    return NULL;
  }
  return payload->data;
}

int mlx_string_free(mlx_string str) {
  string_payload* payload = (string_payload*)str.ctx;
  if (!payload) {
    return 0;
  }
  free(payload->data);
  free(payload);
  return 0;
}

mlx_array mlx_array_new_float64(double val) {
  int shape[1] = {1};
  return make_float64_array_copy(&val, shape, 1);
}

mlx_array mlx_array_new_data(void* data, int* shape, int dim, uint32_t dtype) {
  if (dtype != MLX_FLOAT64) {
    mlx_array empty;
    empty.ctx = NULL;
    return empty;
  }
  return make_float64_array_copy((const double*)data, shape, dim);
}

int mlx_array_free(mlx_array arr) {
  array_payload* payload = (array_payload*)arr.ctx;
  if (!payload) {
    return 0;
  }
  free(payload->shape);
  free(payload->data);
  free(payload);
  return 0;
}

size_t mlx_array_itemsize(mlx_array arr) {
  (void)arr;
  return sizeof(double);
}

size_t mlx_array_size(mlx_array arr) {
  array_payload* payload = (array_payload*)arr.ctx;
  if (!payload) {
    return 0;
  }
  return payload->size;
}

size_t mlx_array_nbytes(mlx_array arr) {
  array_payload* payload = (array_payload*)arr.ctx;
  if (!payload) {
    return 0;
  }
  return payload->size * sizeof(double);
}

size_t mlx_array_ndim(mlx_array arr) {
  array_payload* payload = (array_payload*)arr.ctx;
  if (!payload) {
    return 0;
  }
  return (size_t)payload->ndim;
}

int* mlx_array_shape(mlx_array arr) {
  array_payload* payload = (array_payload*)arr.ctx;
  if (!payload) {
    return NULL;
  }
  return payload->shape;
}

uint32_t mlx_array_dtype(mlx_array arr) {
  array_payload* payload = (array_payload*)arr.ctx;
  if (!payload) {
    return 0;
  }
  return payload->dtype;
}

int mlx_array_eval(mlx_array arr) {
  (void)arr;
  return 0;
}

int mlx_array_item_float64(double* out, mlx_array arr) {
  array_payload* payload = (array_payload*)arr.ctx;
  if (!payload || payload->size != 1) {
    return 1;
  }
  *out = payload->data[0];
  return 0;
}

double* mlx_array_data_float64(mlx_array arr) {
  array_payload* payload = (array_payload*)arr.ctx;
  if (!payload) {
    return NULL;
  }
  return payload->data;
}

int mlx_add(mlx_array* out, mlx_array a, mlx_array b, mlx_stream s) {
  (void)s;
  array_payload* pa = (array_payload*)a.ctx;
  array_payload* pb = (array_payload*)b.ctx;
  if (!pa || !pb || pa->size != pb->size || pa->ndim != pb->ndim) {
    return 1;
  }

  mlx_array result = make_float64_array_copy(pa->data, pa->shape, pa->ndim);
  if (!result.ctx) {
    return 1;
  }

  array_payload* pr = (array_payload*)result.ctx;
  for (size_t i = 0; i < pa->size; i++) {
    pr->data[i] = pa->data[i] + pb->data[i];
  }

  *out = result;
  return 0;
}

int mlx_divide(mlx_array* out, mlx_array a, mlx_array b, mlx_stream s) {
  (void)out;
  (void)a;
  (void)b;
  (void)s;
  return 7;
}

int mlx_reshape(mlx_array* out, mlx_array a, int* shape, size_t shape_num, mlx_stream s) {
  (void)s;
  array_payload* pa = (array_payload*)a.ctx;
  if (!pa) {
    return 1;
  }

  size_t n = shape_size(shape, (int)shape_num);
  if (n != pa->size) {
    return 2;
  }

  mlx_array result = make_float64_array_copy(pa->data, shape, (int)shape_num);
  if (!result.ctx) {
    return 1;
  }

  *out = result;
  return 0;
}

int mlx_sum(mlx_array* out, mlx_array a, bool keepdims, mlx_stream s) {
  (void)keepdims;
  (void)s;
  array_payload* pa = (array_payload*)a.ctx;
  if (!pa) {
    return 1;
  }

  double total = 0.0;
  for (size_t i = 0; i < pa->size; i++) {
    total += pa->data[i];
  }

  *out = mlx_array_new_float64(total);
  return out->ctx ? 0 : 1;
}

int mlx_mean(mlx_array* out, mlx_array a, bool keepdims, mlx_stream s) {
  (void)keepdims;
  (void)s;
  array_payload* pa = (array_payload*)a.ctx;
  if (!pa || pa->size == 0) {
    return 1;
  }

  double total = 0.0;
  for (size_t i = 0; i < pa->size; i++) {
    total += pa->data[i];
  }

  *out = mlx_array_new_float64(total / (double)pa->size);
  return out->ctx ? 0 : 1;
}

int mlx_array_tostring(mlx_string* str, mlx_array arr) {
  array_payload* payload = (array_payload*)arr.ctx;
  if (!payload) {
    return set_string(str, "mlx_array(null)");
  }

  char buf[128];
  snprintf(buf, sizeof(buf), "mlx_array(size=%zu, ndim=%d)", payload->size, payload->ndim);
  return set_string(str, buf);
}
''';
