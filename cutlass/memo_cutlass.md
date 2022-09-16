# CUTLASS (CUDA Templates for Linear Algebra Subroutines)

## 1. 编译

```bash
#!/bin/bash
mkdir build && cd build
cmake ../ -DCUTLASS_NVCC_ARCHS=75
```

* DCUTLASS_NVCC_ARCHS
  * 75: compiles for NVIDIA's Tuning Architecture
  * 80: compiles for NVIDIA's Ampere Architecture

* 编译期间和代码生成相关的cmake片段

```cmake
file(GLOB_RECURSE GENERATOR_PYTHON_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/scripts/*.py)

#
# auto-instantiation of CUTLASS kernels
#

# set cutlass generator compiler version to filter kernels in the generator not supported by a specific toolkit. 
set(CUTLASS_GENERATOR_CUDA_COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION})

execute_process(
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generator.py
    --operations "${CUTLASS_LIBRARY_OPERATIONS}" 
    --build-dir ${PROJECT_BINARY_DIR}
    --curr-build-dir ${CMAKE_CURRENT_BINARY_DIR}
    --generator-target library
    --architectures "${CUTLASS_NVCC_ARCHS_ENABLED}"
    --kernels "${CUTLASS_LIBRARY_KERNELS}"
    --ignore-kernels "${CUTLASS_LIBRARY_IGNORE_KERNELS}"
    --cuda-version "${CUTLASS_GENERATOR_CUDA_COMPILER_VERSION}"
  RESULT_VARIABLE cutlass_lib_INSTANCE_GENERATION_RESULT
  OUTPUT_VARIABLE cutlass_lib_INSTANCE_GENERATION_OUTPUT
  OUTPUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
  ERROR_FILE ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log
)

if(NOT cutlass_lib_INSTANCE_GENERATION_RESULT EQUAL 0)
  message(FATAL_ERROR "Error generating library instances. See ${CMAKE_CURRENT_BINARY_DIR}/library_instance_generation.log")
endif()

```

### 1.1 SM75 硬件概要

实验环境上有一台RTX2080Ti(Arch Turning)，我们以此为平台进行实验。

* 架构

|GPU Features|per SOC|per GPC|per TPC|per SM|
|------------|------:|------:|------:|-----:|
|GPCs        |   6   |    1  |   NG  |  NG  |
|TPCs        |  34*  |    6  |    1  |  NG  |
|SMs         |  68   |   12  |    2  |   1  |
|C-Cores     | 4352  |  768  |  128  |  64  |
|T-Cores     |  544  |   96  |   16  |   8  |

* 算力

|Computing|Peak TFLOPS|
|---------|----------:|
|FP32     | 13.4      |
|INT32    | 13.4      |
|FP16     | 26.9      |
|FP16-FP32-TCore |  53.8|
|FP16-FP16-TCore | 107.6|
|INT8-TCore      | 215.2|
|INT4-TCore      | 430.3|

* 其它
  * GDDR6: 11.2G
  * Bandwith: 616GB/s
  * L2 Cache Size: 5.5M
  * TDP: 250W
  * Die Size: 754mm^2
  * Manufacturing Process: 12nm FFN

* Tensor Core

|Instruction|GPU Architecture|Input Matrix format|Output Accumulator format|Matrix Instruction Size (MxNxK)|
|-|-|-|-|-|
|HMMA (16bits)|Votal  |FP16|FP16 / FP32|8x8x4|
|             |Turning|FP16|FP16 / FP32|8x8x4 / 16x8x8 / 16x8x16|
|             |Ampere |FP16|FP16 / FP32|16x8x8 / 16x8x16|
|HMMA (19bits)|Votal  |TF32|TF32|16x8x4|
|             |Turning|N/A |N/A | N/A  |
|             |Ampere |N/A |N/A | N/A  |
|IMMA (8bits) |Votal  |N/A |N/A | N/A  |
|             |Turning|int8_t |int32_t| 8x8x16 |
|             |Ampere |int8_t |int32_t| 8x8x16 / 16x8x16 / 16x8x32 |
|IMMA (4bits) |Votal  |N/A |N/A | N/A |
|             |Turning|int4_t |int32_t| 8x8x16 |
|             |Ampere |int4_t |int32_t| 8x8x16 / 16x8x16 / 16x8x32 |
|BMMA (1bits) |Votal  |N/A    |N/A    | N/A    |
|             |Turning|int1_t |int32_t| 8x8x128|
|             |Ampere |int1_t |int32_t| 8x8x128 / 16x8x128 / 16x8x256 |
|DMMA (64bits)|Votal  |N/A    |N/A    | N/A    |
|             |Turning|N/A    |N/A    | N/A    |
|             |Ampere |FP64   |FP64   | 8x8x4  |


### 1.2 GenerateSM75()

```python
def GenerateSM75(manifest, cuda_version):
  GenerateSM75_TensorOp_1688(manifest, cuda_version)
  # GenerateSM75_PlanarComplexTensorOp_1688(manifest, cuda_version)
  # GenerateSM75_TensorOp_8816_TN(manifest, cuda_version)
  # GenerateSM75_TensorOp_8816_Interleaved(manifest, cuda_version)
  # GenerateSM75_TensorOp_8832_TN(manifest, cuda_version)
  # GenerateSM75_TensorOp_8832_Interleaved(manifest, cuda_version)
  # GenerateSM75_TensorOp_88128(manifest, cuda_version)
  # #GenerateSM75_WmmaTensorOp_161616(manifest, cuda_version)
  # GenerateSM75_Simt_complex(manifest, cuda_version)
```

我们看一下最常见的FP16输入的计算(16x8x8)

```python

def GenerateSM75_TensorOp_1688(manifest, cuda_version):
  # CUDA 10.2 及其以上的版本才能支持
  if not CudaToolkitVersionSatisfies(cuda_version, 10, 2):
    return

  # 支持四种Layout组合 { MatrixA, MatrixB, MatrixC/MatrixD }
  layouts = [
    (LayoutType.ColumnMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.ColumnMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.ColumnMajor, LayoutType.ColumnMajor),
    (LayoutType.RowMajor, LayoutType.RowMajor, LayoutType.ColumnMajor),
  ]

  # 支持两种Intrinsic：f16*f16=f32 and f16*f16=f16
  math_instructions = [
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f32,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
    MathInstruction(                                  \
      [16, 8, 8],                                     \
      DataType.f16, DataType.f16, DataType.f16,       \
      OpcodeClass.TensorOp,                           \
      MathOperation.multiply_add),
  ]

  # minimum_compute_capability
  min_cc = 75
  # minimum_compute_capability
  max_cc = 1024

  alignment_constraints = [8, 4, 2, 1]

  for math_inst in math_instructions:
    tile_descriptions = [
      TileDescription([256, 128, 32], 2, [4, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 256, 32], 2, [2, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([128, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 256, 32], 2, [1, 4, 1], math_inst, min_cc, max_cc),
      TileDescription([256,  64, 32], 2, [4, 1, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([128,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64,  64, 32], 2, [2, 2, 1], math_inst, min_cc, max_cc),
      TileDescription([ 64, 128, 64], 2, [1, 2, 2], math_inst, min_cc, max_cc),
    ]

    data_type = [
      math_inst.element_a,
      math_inst.element_b,
      math_inst.element_accumulator,
      math_inst.element_accumulator,
    ]
    
    CreateGemmOperator(manifest, layouts, tile_descriptions, \
      data_type, alignment_constraints)

    conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

    CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)

    # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
    if math_inst.element_a != math_inst.element_accumulator:

      data_type_mixed = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_a,
        math_inst.element_accumulator,
      ]

      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type_mixed, alignment_constraints)

      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)

    # Separate generator for 'few channels' specializations
    GenerateSM75_TensorOp_1688_FewChannels(manifest, cuda_version, math_inst)
```
