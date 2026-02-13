import 'dart:io';

import 'package:mlx_ffi/mlx_ffi.dart';
import 'package:test/test.dart';

void main() {
  final runIntegration =
      Platform.environment['MLX_FFI_RUN_LLM_INTEGRATION'] == '1';
  final modelDir = Platform.environment['MLX_FFI_MODEL_DIR'];
  final libraryPath = Platform.environment['MLX_FFI_LIBRARY_PATH'];

  final skipReason = !runIntegration
      ? 'Set MLX_FFI_RUN_LLM_INTEGRATION=1 to enable.'
      : (!Platform.isMacOS
          ? 'Integration test currently targets macOS.'
          : (modelDir == null || modelDir.isEmpty
              ? 'Set MLX_FFI_MODEL_DIR to a local HF MLX model folder.'
              : null));

  test(
    'local model folder smoke generation',
    () async {
      final llm = MlxLlm.open(
        libraryPath: libraryPath,
        modelDirectory: modelDir!,
        useGpuDefaultStream: true,
      );
      addTearDown(llm.dispose);

      final output = await llm
          .generateStream(
            prompt: 'how do i make cake',
            options: const MlxLlmSamplingOptions(maxTokens: 96),
          )
          .join();

      expect(output, isNotEmpty);
    },
    skip: skipReason,
  );
}
