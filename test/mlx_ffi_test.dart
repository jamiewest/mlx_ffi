import 'dart:io';

import 'package:mlx_ffi/mlx_ffi.dart';
import 'package:test/test.dart';

void main() {
  late Directory tempDir;
  late String libraryPath;
  late String modelDirPath;
  late Mlx mlx;
  late MlxLlm llm;

  setUpAll(() async {
    if (Platform.isWindows) {
      return;
    }

    tempDir = await Directory.systemTemp.createTemp('mlx_ffi_test_');
    libraryPath = await _buildFakeMlxLibrary(tempDir);
    modelDirPath = '${tempDir.path}/fake_model';
    await Directory(modelDirPath).create(recursive: true);
    await File('$modelDirPath/config.json').writeAsString('{}');
    await File('$modelDirPath/tokenizer.json').writeAsString('{}');
    await File('$modelDirPath/model.safetensors').writeAsBytes(<int>[0]);

    mlx = Mlx.open(libraryPath: libraryPath);
    llm = MlxLlm.open(
      libraryPath: libraryPath,
      modelDirectory: modelDirPath,
    );
  });

  tearDownAll(() async {
    if (Platform.isWindows) {
      return;
    }

    llm.dispose();
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

  test('llm tokenize and decode roundtrip', () {
    if (Platform.isWindows) {
      return;
    }

    final tokens = llm.tokenize('cake', addBos: true, addEos: true);
    expect(tokens, <int>[1, 99, 97, 107, 101, 2]);
    expect(llm.decodeTokens(tokens), 'cake');
  });

  test('llm generation stream yields pieces and completes', () async {
    if (Platform.isWindows) {
      return;
    }

    final pieces = await llm
        .generateStream(
          prompt: 'how do i make cake',
          options: const MlxLlmSamplingOptions(maxTokens: 24),
        )
        .toList();
    final combined = pieces.join();

    expect(combined, contains('cake'));
    expect(combined, contains('mix'));
  });

  test('llm open and dispose are safe', () {
    if (Platform.isWindows) {
      return;
    }

    final instance = MlxLlm.open(
      libraryPath: libraryPath,
      modelDirectory: modelDirPath,
    );
    expect(() => instance.dispose(), returnsNormally);
    expect(() => instance.dispose(), returnsNormally);
  });

  test('llm native non-zero return code throws MlxException', () async {
    if (Platform.isWindows) {
      return;
    }

    expect(
      () async => llm.generateStream(prompt: '__gen_error__').toList(),
      throwsA(
        isA<MlxException>()
            .having((e) => e.operation, 'operation', 'mlx_llm_generation_start')
            .having((e) => e.code, 'code', 55),
      ),
    );
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
typedef struct { void* ctx; } mlx_vector_int;
typedef struct { void* ctx; } mlx_llm_model;
typedef struct { void* ctx; } mlx_llm_generation;

typedef struct {
  float temperature;
  float top_p;
  int top_k;
  int max_tokens;
  float repetition_penalty;
  bool has_seed;
  unsigned int seed;
  const char** stop_sequences;
  size_t stop_sequence_count;
  unsigned int stop_handlingAsInt;
} mlx_llm_sampling_options;

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

typedef struct {
  int* data;
  size_t size;
} vector_int_payload;

typedef struct {
  char* model_dir;
  bool has_active_generation;
} llm_model_payload;

typedef struct {
  llm_model_payload* model;
  char** pieces;
  size_t piece_count;
  size_t index;
  bool cancelled;
} llm_generation_payload;

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
  if (!out) {
    return 1;
  }

  if (out->ctx) {
    string_payload* existing = (string_payload*)out->ctx;
    free(existing->data);
    free(existing);
    out->ctx = NULL;
  }

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

static int set_vector_int(mlx_vector_int* out, const int* data, size_t size) {
  if (!out) {
    return 1;
  }

  if (out->ctx) {
    vector_int_payload* existing = (vector_int_payload*)out->ctx;
    free(existing->data);
    free(existing);
    out->ctx = NULL;
  }

  vector_int_payload* payload = (vector_int_payload*)calloc(1, sizeof(vector_int_payload));
  if (!payload) {
    return 1;
  }

  if (size > 0) {
    payload->data = (int*)calloc(size, sizeof(int));
    if (!payload->data) {
      free(payload);
      return 1;
    }
    memcpy(payload->data, data, size * sizeof(int));
  }
  payload->size = size;
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

int mlx_vector_int_free(mlx_vector_int vec) {
  vector_int_payload* payload = (vector_int_payload*)vec.ctx;
  if (!payload) {
    return 0;
  }
  free(payload->data);
  free(payload);
  return 0;
}

size_t mlx_vector_int_size(mlx_vector_int vec) {
  vector_int_payload* payload = (vector_int_payload*)vec.ctx;
  if (!payload) {
    return 0;
  }
  return payload->size;
}

int mlx_vector_int_get(int* out, const mlx_vector_int vec, size_t idx) {
  vector_int_payload* payload = (vector_int_payload*)vec.ctx;
  if (!payload || !out || idx >= payload->size) {
    return 1;
  }
  *out = payload->data[idx];
  return 0;
}

int mlx_llm_model_load(mlx_llm_model* out, const char* model_dir, const mlx_stream s) {
  (void)s;
  if (!out || !model_dir || model_dir[0] == '\0') {
    return 2;
  }
  if (strstr(model_dir, "__missing__") != NULL) {
    return 4;
  }

  llm_model_payload* payload = (llm_model_payload*)calloc(1, sizeof(llm_model_payload));
  if (!payload) {
    return 1;
  }

  size_t len = strlen(model_dir);
  payload->model_dir = (char*)calloc(len + 1, sizeof(char));
  if (!payload->model_dir) {
    free(payload);
    return 1;
  }
  memcpy(payload->model_dir, model_dir, len + 1);
  payload->has_active_generation = false;
  out->ctx = payload;
  return 0;
}

int mlx_llm_model_free(mlx_llm_model model) {
  llm_model_payload* payload = (llm_model_payload*)model.ctx;
  if (!payload) {
    return 0;
  }
  free(payload->model_dir);
  free(payload);
  return 0;
}

int mlx_llm_tokenize(
    mlx_vector_int* out,
    mlx_llm_model model,
    const char* text,
    bool add_bos,
    bool add_eos) {
  llm_model_payload* payload = (llm_model_payload*)model.ctx;
  if (!payload || !out || !text) {
    return 2;
  }
  if (strcmp(text, "__tokenize_error__") == 0) {
    return 41;
  }

  size_t len = strlen(text);
  size_t size = len + (add_bos ? 1 : 0) + (add_eos ? 1 : 0);
  int* buf = (int*)calloc(size > 0 ? size : 1, sizeof(int));
  if (!buf) {
    return 1;
  }

  size_t j = 0;
  if (add_bos) {
    buf[j++] = 1;
  }
  for (size_t i = 0; i < len; i++) {
    buf[j++] = (unsigned char)text[i];
  }
  if (add_eos) {
    buf[j++] = 2;
  }

  int code = set_vector_int(out, buf, j);
  free(buf);
  return code;
}

int mlx_llm_decode(
    mlx_string* out,
    mlx_llm_model model,
    const int* tokens,
    size_t token_count) {
  llm_model_payload* payload = (llm_model_payload*)model.ctx;
  if (!payload || !out || (token_count > 0 && !tokens)) {
    return 2;
  }

  char* text = (char*)calloc(token_count + 1, sizeof(char));
  if (!text) {
    return 1;
  }

  size_t j = 0;
  for (size_t i = 0; i < token_count; i++) {
    int token = tokens[i];
    if (token == 1 || token == 2) {
      continue;
    }
    if (token < 0 || token > 255) {
      text[j++] = '?';
    } else {
      text[j++] = (char)token;
    }
  }
  text[j] = '\0';

  int code = set_string(out, text);
  free(text);
  return code;
}

static llm_generation_payload* create_generation_payload(
    llm_model_payload* model,
    const char* response,
    int max_tokens) {
  llm_generation_payload* payload =
      (llm_generation_payload*)calloc(1, sizeof(llm_generation_payload));
  if (!payload) {
    return NULL;
  }

  payload->model = model;
  payload->cancelled = false;
  payload->index = 0;

  size_t len = strlen(response);
  size_t piece_capacity = 0;
  for (size_t i = 0; i < len; i++) {
    if (response[i] == ' ') {
      piece_capacity++;
    }
  }
  piece_capacity += 1;

  payload->pieces = (char**)calloc(piece_capacity, sizeof(char*));
  if (!payload->pieces) {
    free(payload);
    return NULL;
  }

  size_t start = 0;
  while (start < len) {
    size_t end = start;
    while (end < len && response[end] != ' ') {
      end++;
    }
    if (end < len) {
      end++;
    }

    if (max_tokens > 0 && (int)payload->piece_count >= max_tokens) {
      break;
    }

    size_t piece_len = end - start;
    char* piece = (char*)calloc(piece_len + 1, sizeof(char));
    if (!piece) {
      for (size_t i = 0; i < payload->piece_count; i++) {
        free(payload->pieces[i]);
      }
      free(payload->pieces);
      free(payload);
      return NULL;
    }
    memcpy(piece, response + start, piece_len);
    piece[piece_len] = '\0';
    payload->pieces[payload->piece_count++] = piece;
    start = end;
  }

  return payload;
}

int mlx_llm_generation_start(
    mlx_llm_generation* out,
    mlx_llm_model model,
    const char* prompt,
    const mlx_llm_sampling_options* opts) {
  llm_model_payload* payload = (llm_model_payload*)model.ctx;
  if (!payload || !out || !prompt) {
    return 2;
  }
  if (payload->has_active_generation) {
    return 5;
  }
  if (strcmp(prompt, "__gen_error__") == 0) {
    return 55;
  }

  const char* response = strstr(prompt, "cake") != NULL
      ? "To make cake, mix dry ingredients, add eggs and butter, bake, then cool and frost."
      : "This is a streamed response from the fake native LLM.";

  int max_tokens = 32;
  if (opts && opts->max_tokens > 0) {
    max_tokens = opts->max_tokens;
  }

  llm_generation_payload* gen =
      create_generation_payload(payload, response, max_tokens);
  if (!gen) {
    return 1;
  }

  payload->has_active_generation = true;
  out->ctx = gen;
  return 0;
}

int mlx_llm_generation_next(mlx_string* token_text, bool* is_done, mlx_llm_generation gen) {
  llm_generation_payload* payload = (llm_generation_payload*)gen.ctx;
  if (!payload || !token_text || !is_done) {
    return 2;
  }

  if (payload->cancelled || payload->index >= payload->piece_count) {
    *is_done = true;
    return set_string(token_text, "");
  }

  const char* piece = payload->pieces[payload->index++];
  *is_done = payload->index >= payload->piece_count;
  return set_string(token_text, piece);
}

int mlx_llm_generation_cancel(mlx_llm_generation gen) {
  llm_generation_payload* payload = (llm_generation_payload*)gen.ctx;
  if (!payload) {
    return 0;
  }
  payload->cancelled = true;
  if (payload->model) {
    payload->model->has_active_generation = false;
  }
  return 0;
}

int mlx_llm_generation_free(mlx_llm_generation gen) {
  llm_generation_payload* payload = (llm_generation_payload*)gen.ctx;
  if (!payload) {
    return 0;
  }
  if (payload->model) {
    payload->model->has_active_generation = false;
  }
  for (size_t i = 0; i < payload->piece_count; i++) {
    free(payload->pieces[i]);
  }
  free(payload->pieces);
  free(payload);
  return 0;
}
''';
