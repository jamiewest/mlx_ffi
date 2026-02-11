# mlx_ffi

Dart FFI bindings for [`mlx-c`](https://github.com/ml-explore/mlx-c), generated with [`ffigen`](https://pub.dev/packages/ffigen).

## What is included

- `lib/src/generated/mlx_bindings.dart`: auto-generated low-level bindings.
- `lib/mlx_ffi.dart`: single-file high-level API (`Mlx`, `MlxArray`) including dynamic library loading.
- `tool/ffigen_include/`: parse-only header overlay so `ffigen` can generate APIs that use C complex/fp16/bf16 types.

The generated bindings include the full public `mlx_*` symbol surface from `mlx/c/*.h`.

## Generate bindings

```bash
dart pub get
dart run ffigen --config ffigen.yaml
```

Or:

```bash
./tool/generate_bindings.sh
```

Validate symbol coverage:

```bash
./tool/verify_symbol_coverage.sh
```

## Runtime library

The high-level wrapper expects `mlx-c` to be available as a dynamic library:

- macOS: `libmlx.dylib`
- Linux: `libmlx.so`
- Windows: `mlx.dll`

You can also provide an explicit path with:

```dart
final mlx = Mlx.open(libraryPath: '/path/to/libmlx.dylib');
```

## Pure Dart CLI example

Run the non-Flutter example directly:

```bash
dart run bin/mlx_cli_example.dart --library /absolute/path/to/libmlx.dylib
```

Optional:

```bash
dart run bin/mlx_cli_example.dart --gpu --library /absolute/path/to/libmlx.dylib
```

## Build Apple XCFramework (Dart script)

You can build a multi-slice Apple XCFramework (macOS + iOS + iOS simulator)
without Flutter, using Dart as the orchestration layer:

```bash
dart run tool/build_apple_xcframework.dart
```

Common variants:

```bash
# Only macOS slice
dart run tool/build_apple_xcframework.dart --platforms macos

# Reuse existing build tree and set deployment targets
dart run tool/build_apple_xcframework.dart --no-clean --ios-min 17.0 --macos-min 14.0

# Use a separate mlx-c checkout
dart run tool/build_apple_xcframework.dart --source /absolute/path/to/mlx-c
```

Output default:

```text
build/apple/MLXC.xcframework
```

## Vendored source

`mlx-c` is vendored at `third_party/mlx-c` and used as the header source for generation.
