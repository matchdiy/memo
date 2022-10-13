# CUTLASS (CUDA Templates for Linear Algebra Subroutines)

## 1. 硬件

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

### 1.2 TensorCore

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

## 2. 编译期代码生成

### 1.1 编译

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
################################################################################
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
################################################################################
```

### 1.2 代码生成

* __GenerateSM75__

  ```python
  ############################### Python Script ################################
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
  ##############################################################################
  ```

* __GenerateSM75_TensorOp_1688__

  ```python
  ##############################################################################
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

    # 支持两种 Intrinsic：f16*f16=f32 and f16*f16=f16
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

    # TODO: 
    alignment_constraints = [8, 4, 2, 1]

    for math_inst in math_instructions:
      tile_descriptions = [
        #                             stages
        #                 threadblock   |  warp_count
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

      # MatrixC 和 MatrixD 必须是相同的数据类型
      data_type = [
        math_inst.element_a,
        math_inst.element_b,
        math_inst.element_accumulator,
        math_inst.element_accumulator,
      ]
      
      # math_inst[0]: DataType.f16 / DataType.f16 / DataType.f32 / DataType.f32
      # math_inst[1]: DataType.f16 / DataType.f16 / DataType.f16 / DataType.f16
      CreateGemmOperator(manifest, layouts, tile_descriptions, \
        data_type, alignment_constraints)

      conv_layout = (LayoutType.TensorNHWC, LayoutType.TensorNHWC, LayoutType.TensorNHWC)

      # TODO
      CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type, alignment_constraints)

      # Avoid emitting two kernels if the accumulator type does not differ from the input type (e.g. F16 accumulation)
      if math_inst.element_a != math_inst.element_accumulator:

        data_type_mixed = [
          math_inst.element_a,
          math_inst.element_b,
          math_inst.element_a,
          math_inst.element_accumulator,
        ]

        # math_inst[0]: DataType.f16 / DataType.f16 / DataType.f16 / DataType.f32
        CreateGemmOperator(manifest, layouts, tile_descriptions, \
          data_type_mixed, alignment_constraints)

        CreateConv2dOperator(manifest, conv_layout, tile_descriptions, data_type_mixed, alignment_constraints)

      # 生成第三部分：DataType.f16 / DataType.f16 / DataType.f16 / DataType.f32
      # Separate generator for 'few channels' specializations
      GenerateSM75_TensorOp_1688_FewChannels(manifest, cuda_version, math_inst)
  ##############################################################################
  ```

* __TileDescription__ (默认只会生成最大Tile)

  ```text
  ## architectures is 75
  CreateGemmOperator:
  --> hhss_nnn_256x128x32_2_4x2x1_16x8x8_alignment
  --> hhss_ntn_256x128x32_2_4x2x1_16x8x8_alignment
  --> hhss_tnn_256x128x32_2_4x2x1_16x8x8_alignment
  --> hhss_ttn_256x128x32_2_4x2x1_16x8x8_alignment
  CreateGemmOperator:
  --> hhhs_nnn_256x128x32_2_4x2x1_16x8x8_alignment
  --> hhhs_ntn_256x128x32_2_4x2x1_16x8x8_alignment
  --> hhhs_tnn_256x128x32_2_4x2x1_16x8x8_alignment
  --> hhhs_ttn_256x128x32_2_4x2x1_16x8x8_alignment
  CreateGemmOperator:
  --> hhhh_nnn_256x128x32_2_4x2x1_16x8x8_alignment
  --> hhhh_ntn_256x128x32_2_4x2x1_16x8x8_alignment
  --> hhhh_tnn_256x128x32_2_4x2x1_16x8x8_alignment
  --> hhhh_ttn_256x128x32_2_4x2x1_16x8x8_alignment
  ```

* __manifest.emit()__
  * generate file: _initialize_all.cpp_

    ```C++
    /***************************************************************************
    Generated by manifest.py - Do not edit.
    ***************************************************************************/
    #include "cutlass/library/library.h"
    #include "cutlass/library/manifest.h"

    namespace cutlass {
      namespace library {

        void initialize_all_gemm_operations(Manifest &manifest);
        void initialize_all_conv2d_operations(Manifest &manifest);

        void initialize_all(Manifest &manifest) {
          manifest.reserve(30);


          initialize_all_gemm_operations(manifest);
          initialize_all_conv2d_operations(manifest);
          }

      } // namespace library
    } // namespace cutlass

    ```

  * generate file: _all_gemm_operations.cu_

    ```C++
    /***************************************************************************
    Generated by manifest.py - Do not edit.
    ***************************************************************************/

    #include "cutlass/cutlass.h"
    #include "cutlass/library/library.h"
    #include "cutlass/library/manifest.h"

    namespace cutlass {
    namespace library {

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    void initialize_cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_s1688gemm_f16_256x128_32x2_tn_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_s1688gemm_f16_256x128_32x2_tt_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nt_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tt_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_h1688gemm_256x128_32x2_nt_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_h1688gemm_256x128_32x2_tn_align8(Manifest &manifest);
    void initialize_cutlass_tensorop_h1688gemm_256x128_32x2_tt_align8(Manifest &manifest);


    //
    // Entry point to construct operations
    //
    void initialize_all_gemm_operations(Manifest &manifest) {
      initialize_cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8(manifest);
      initialize_cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8(manifest);
      initialize_cutlass_tensorop_s1688gemm_f16_256x128_32x2_tn_align8(manifest);
      initialize_cutlass_tensorop_s1688gemm_f16_256x128_32x2_tt_align8(manifest);
      initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align8(manifest);
      initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nt_align8(manifest);
      initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8(manifest);
      initialize_cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tt_align8(manifest);
      initialize_cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8(manifest);
      initialize_cutlass_tensorop_h1688gemm_256x128_32x2_nt_align8(manifest);
      initialize_cutlass_tensorop_h1688gemm_256x128_32x2_tn_align8(manifest);
      initialize_cutlass_tensorop_h1688gemm_256x128_32x2_tt_align8(manifest);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    } // namespace library
    } // namespace cutlass

    ```

  * All generated files list:

    ```Text
    ./generated/initialize_all.cpp
    ./generated/gemm/all_gemm_operations.cu
    ./generated/gemm/cutlass_tensorop_s1688gemm_f16_256x128_32x2_nn_align8.cu
    ./generated/gemm/cutlass_tensorop_s1688gemm_f16_256x128_32x2_nt_align8.cu
    ./generated/gemm/cutlass_tensorop_s1688gemm_f16_256x128_32x2_tn_align8.cu
    ./generated/gemm/cutlass_tensorop_s1688gemm_f16_256x128_32x2_tt_align8.cu
    ./generated/gemm/cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align8.cu
    ./generated/gemm/cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nt_align8.cu
    ./generated/gemm/cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tn_align8.cu
    ./generated/gemm/cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_tt_align8.cu
    ./generated/gemm/cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8.cu
    ./generated/gemm/cutlass_tensorop_h1688gemm_256x128_32x2_nt_align8.cu
    ./generated/gemm/cutlass_tensorop_h1688gemm_256x128_32x2_tn_align8.cu
    ./generated/gemm/cutlass_tensorop_h1688gemm_256x128_32x2_tt_align8.cu
    ```

  * kernel 文件命名规则：
  
    ```Python
    cutlass_${opcode_class}_${extended_name}_${threadblock}_${layout}_align${alignment}
    ```

    * ___opcode_class___:  _'simt'_,  _'tensorop'_,  _'wmma_tensorop'_
    * ___extended_name___:
      * default: ```${element_c}_${core_name}_${element_a}```
      * matrixC_dtype == math_inst.element_accumulator: 省略 ```${element_c}```
      * matrixA_dtype == math_inst.element_accumulator: 省略 ```${element_a}```

  * _cutlass_tensorop_f16_s1688gemm_f16_256x128_32x2_nn_align8.cu_

    这部分代码相当于把Kernel的Configuration透过 ___cutlass::gemm::device::GemmUniversalAdapter___ 生成 ___GemmUniversalOperation___ 添加到_Manifest_(Singleton)中。

    ```C++
        
    /*
      Generated by gemm_operation.py - Do not edit.
    */

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    #include "cutlass/cutlass.h"
    #include "cutlass/library/library.h"
    #include "cutlass/library/manifest.h"
    #include "library_internal.h"
    #include "gemm_operation.h"
    #include "cutlass/arch/wmma.h"
    #include "cutlass/numeric_types.h"
    #include "cutlass/arch/arch.h"
    #include "cutlass/arch/mma.h"
    #include "cutlass/layout/matrix.h"
    #include "cutlass/gemm/device/gemm.h"
    #include "cutlass/gemm/device/gemm_universal_adapter.h"
    #include "cutlass/gemm/kernel/default_gemm_universal.h"

    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Gemm operator cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8
    using cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8_base = 
      typename cutlass::gemm::kernel::DefaultGemmUniversal<
        cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed B operand
        cutlass::half_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,    // transposed A operand
        cutlass::half_t, cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<256, 128, 32>,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        
        cutlass::epilogue::thread::LinearCombination<
          cutlass::half_t,
          8,
          cutlass::half_t,
          cutlass::half_t
        >,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        2,
        cutlass::arch::OpMultiplyAdd
      >::GemmKernel;

    // Define named type
    struct cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8 : 
      public cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8_base { };

    namespace cutlass {
    namespace library {

    void initialize_cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8(Manifest &manifest) {
      manifest.append(new GemmUniversalOperation<
          cutlass::gemm::device::GemmUniversalAdapter<cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8>>("cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8"));

    }
    } // namespace library
    } // namespace cutlass
    ```

  * level-2: __cutlass::gemm::kernel::DefaultGemmUniversal__: cutlass_tensorop_h1688gemm_256x128_32x2_nn_align8
  * level-1: cutlass::gemm::device::GemmUniversalAdapter<_level-2_>
  * level-0: cutlass::library::GemmUniversalOperation<_level-1_>

## 3. cutlass template library

### 3.1 Code Organization

CUTLASS Templates are implemented by header files in the following directory structure:

```Python
include/            # Top-level include directory. Client applications should target this path.

  cutlass/          # CUDA Templates for Linear Algebra Subroutines and Solvers - headers only

    arch/           # direct exposure of architecture features (including instruction-level GEMMs)
      *
    gemm/           # code specialized for general matrix product computations
      thread/       #   thread-level operators
      warp/         #   warp-level operators
      threadblock/  #   CTA-level operators
      kernel/       #   CUDA kernel entry points
      device/       #   launches kernel(s) over a full device
      *             # scope-agnostic components and basic vocabular type definitions for GEMM

    layout/         # layout definitions for matrices, tensors, and other mathematical objects in memory
      *

    reduction/      # bandwidth-limited reduction kernels that do not fit the "gemm" models
      thread/       #   thread-level operators
      warp/         #   warp-level operators
      threadblock/  #   CTA-level operators
      kernel/       #   CUDA kernel entry points
      device/       #   launches kernel(s) over a full device
      *             # scope-agnostic components and basic vocabular type definitions

    transform/      # code specialized for layout, type, and domain transformations
      thread/       #   thread-level operators
      warp/         #   warp-level operators
      threadblock/  #   CTA-level operators
      kernel/       #   CUDA kernel entry points
      device/       #   launches kernel(s) over a full device
      *             # scope-agnostic components and basic vocabulary type definitions

    util/           # miscellaneous CUTLASS components
      *
    *               # core vocabulary types and fundamental arithmetic operators
```

### 3.2 Class Partial Specialization

#### 3.2.1 cutlass::gemm::kernel::DefaultGemmUniversal Implement

* class DefaultGemmUniversal
  * cutlass/gemm/kernel/default_gemm_universal.h
  * Partial specialization
    * define COMPLEX_TRUE = typename platform::enable_if<cutlass::is_complex<_ElementAccumulator_>::value>::type

  | |DefaultGemmUniversal<>        | Real                        |Complex           |
  |-|------------------------------|-----------------------------|------------------|
  | |typename ElementA_            |ElementA                     |ElementA          |
  | |typename LayoutA_             |LayoutA                      |LayoutA           |
  |*|ComplexTransform TransformA   |___ComplexTransform::kNone___|___TransformA___  |
  | |int kAlignmentA               |kAlignmentA                  |kAlignmentA       |
  | |typename ElementB_            |ElementB                     |ElementB          |
  | |typename LayoutB_             |LayoutB                      |LayoutB           |
  |*|ComplexTransform TransformB   |___ComplexTransform::kNone___|___TransformB___  |
  | |int kAlignmentB               |kAlignmentB                  |kAlignmentB       |
  | |typename ElementC_            |ElementC                     |ElementC          |
  | |typename LayoutC_             |LayoutC                      |LayoutC           |
  | |typename ElementAccumulator   |ElementAccumulator           |ElementAccumulator|
  | |typename OperatorClass        |OperatorClass                |OperatorClass     |
  | |typename ArchTag              |ArchTag                      |ArchTag           |
  | |typename ThreadblockShape     |ThreadblockShape             |ThreadblockShape  |
  | |typename WarpShape            |WarpShape                    |WarpShape         |
  | |typename InstructionShape     |InstructionShape             |InstructionShape  |
  | |typename EpilogueOutputOp     |EpilogueOutputOp             |EpilogueOutputOp  |
  | |typename ThreadblockSwizzle   |ThreadblockSwizzle           |ThreadblockSwizzle|
  | |int Stages                    |Stages                       |Stages            |
  | |typename Operator             |Operator                     |Operator          |
  | |SharedMemoryClearOption::kNone|SharedMemoryClear            |SharedMemoryClear |
  | |bool GatherA = false          |GatherA                      |false             |
  | |bool GatherB = false          |GatherB                      |false             |
  | |bool ScatterD = false         |ScatterD                     |false             |
  |*|typename Enable = void        |!COMPLEX_TRUE                |COMPLEX_TRUE      |

#### 3.2.2 cutlass::gemm::kernel::DefaultGemm

* Partial Specialization
  * SM80_TENSOR_CORE
  * SM75_TENSOR_CORE
  * SM70_TENSOR_CORE
  * SM80_TENSOR_CORE_INTEGER
  * SM75_TENSOR_CORE_INTEGER
  * SM80_CUDA_CORE
  * notSM80_CUDA_CORE
  * DP4A
  * WMMA

* Tensor Core

  | |Define                        |Ampere Tensor Core         |Turning Tensor Core        |Votal Tensor Core          |
  |-|------------------------------|---------------------------|---------------------------|---------------------------|
  | |typename ElementA_            |ElementA                   |ElementA                   |ElementA                   |
  | |typename LayoutA_             |LayoutA                    |LayoutA                    |LayoutA                    |
  | |int kAlignmentA               |kAlignmentA                |kAlignmentA                |kAlignmentA                |
  | |typename ElementB_            |ElementB                   |ElementB                   |ElementB                   |
  | |typename LayoutB_             |LayoutB                    |LayoutB                    |LayoutB                    |
  | |int kAlignmentB               |kAlignmentB                |kAlignmentB                |kAlignmentB                |
  | |typename ElementC_            |ElementC                   |ElementC                   |ElementC                   |
  |*|typename LayoutC_             |LayoutC                    |___layout::RowMajor___     |___layout::RowMajor___     |
  | |typename ElementAccumulator   |ElementAccumulator         |ElementAccumulator         |ElementAccumulator         |
  |*|typename OperatorClass        |___arch::OpClassTensorOp___|___arch::OpClassTensorOp___|___arch::OpClassTensorOp___|
  |*|typename ArchTag              |___arch::Sm80___           |___arch::Sm75___           |___arch::Sm70___           |
  | |typename ThreadblockShape     |ThreadblockShape           |ThreadblockShape           |ThreadblockShape           |
  | |typename WarpShape            |WarpShape                  |WarpShape                  |WarpShape                  |
  |*|typename InstructionShape     |InstructionShape           |InstructionShape           |___GemmShape<8,8,4>___     |
  | |typename EpilogueOutputOp     |EpilogueOutputOp           |EpilogueOutputOp           |EpilogueOutputOp           |
  | |typename ThreadblockSwizzle   |ThreadblockSwizzle         |ThreadblockSwizzle         |ThreadblockSwizzle         |
  |*|int Stages                    |Stages                     |___2___                    |___2___                    |
  | |bool SplitKSerial             |SplitKSerial               |SplitKSerial               |SplitKSerial               |
  | |typename Operator             |Operator                   |Operator                   |Operator                   |
  | |SharedMemoryClearOption::kNone|SharedMemoryClear          |SharedMemoryClear          |SharedMemoryClear          |
  | |bool GatherA = false          |GatherA                    |GatherA                    |GatherA                    |
  | |bool GatherB = false          |GatherB                    |GatherB                    |GatherB                    |
  | |bool ScatterD = false         |ScatterD                   |ScatterD                   |ScatterD                   |
  | |typename Enable = void        |                           |                           |                           |

* Tensor Core Integer
  * define layout::CMIK = layout::ColumnMajorInterleaved<_InterleavedK_>
  * define layout::RMIK = layout::RowMajorInterleaved<_InterleavedK_>

  | |Define                        |Ampere Tensor Core Integer |Turning Tensor Core Integer|
  |-|------------------------------|---------------------------|---------------------------|
  | |typename ElementA_            |ElementA                   |ElementA                   |
  |*|typename LayoutA_             |___CMIK___                 |___CMIK___                 |
  | |int kAlignmentA               |kAlignmentA                |kAlignmentA                |
  | |typename ElementB_            |ElementB                   |ElementB                   |
  |*|typename LayoutB_             |___RMIK___                 |___RMIK___                 |
  | |int kAlignmentB               |kAlignmentB                |kAlignmentB                |
  | |typename ElementC_            |ElementC                   |ElementC                   |
  |*|typename LayoutC_             |___CMIK___                 |___CMIK___                 |
  |*|typename ElementAccumulator   |___int32_t___              |___int32_t___              |
  |*|typename OperatorClass        |___arch::OpClassTensorOp___|___arch::OpClassTensorOp___|
  |*|typename ArchTag              |___arch::Sm80___           |___arch::Sm75___           |
  | |typename ThreadblockShape     |ThreadblockShape           |ThreadblockShape           |
  | |typename WarpShape            |WarpShape                  |WarpShape                  |
  | |typename InstructionShape     |InstructionShape           |InstructionShape           |
  | |typename EpilogueOutputOp     |EpilogueOutputOp           |EpilogueOutputOp           |
  | |typename ThreadblockSwizzle   |ThreadblockSwizzle         |ThreadblockSwizzle         |
  |*|int Stages                    |Stages                     |___2___                    |
  | |bool SplitKSerial             |SplitKSerial               |SplitKSerial               |
  | |typename Operator             |Operator                   |Operator                   |
  | |SharedMemoryClearOption::kNone|SharedMemoryClear          |SharedMemoryClear          |
  | |bool GatherA = false          |false                      |false                      |
  | |bool GatherB = false          |false                      |false                      |
  | |bool ScatterD = false         |false                      |false                      |
  | |typename Enable = void        |                           |                           |

* SIMT
  * define NOT_SM80 = ! platform::is_same<ArchTag, arch::Sm80>::value >::type

  | |Define                        |NOT_SM80 Cuda Core      |SM80 Cuda Core         |  DP4A                 |WMMA|
  |-|------------------------------|------------------------|-----------------------|-----------------------|----|
  | |typename ElementA_            |ElementA                |ElementA               |___int8_t___           |    |
  |*|typename LayoutA_             |LayoutA                 |LayoutA                |LayoutA                |    |
  | |int kAlignmentA               |kAlignmentA             |kAlignmentA            |kAlignmentA            |    |
  | |typename ElementB_            |ElementB                |ElementB               |___int8_t___           |    |
  |*|typename LayoutB_             |LayoutB                 |LayoutB                |LayoutB                |    |
  | |int kAlignmentB               |kAlignmentB             |kAlignmentB            |kAlignmentB            |    |
  | |typename ElementC_            |ElementC                |ElementC               |ElementC               |    |
  |*|typename LayoutC_             |LayoutC                 |LayoutC                |LayoutC                |    |
  |*|typename ElementAccumulator   |ElementAccumulator      |ElementAccumulator     |ElementAccumulator     |    |
  |*|typename OperatorClass        |___arch::OpClassSimt___ |___arch::OpClassSimt___|___arch::OpClassSimt___|    |
  |*|typename ArchTag              |ArchTag                 |___arch::Sm80___       |ArchTag                |    |
  | |typename ThreadblockShape     |ThreadblockShape        |ThreadblockShape       |ThreadblockShape       |    |
  | |typename WarpShape            |WarpShape               |WarpShape              |WarpShape              |    |
  |*|typename InstructionShape     |___GemmShape<1,1,1>___  |___GemmShape<1,1,1>___ |___GemmShape<1,1,4>___ |    |
  | |typename EpilogueOutputOp     |EpilogueOutputOp        |EpilogueOutputOp       |EpilogueOutputOp       |    |
  | |typename ThreadblockSwizzle   |ThreadblockSwizzle      |ThreadblockSwizzle     |ThreadblockSwizzle     |    |
  |*|int Stages                    |___2___                 |Stages                 |___2___                |    |
  | |bool SplitKSerial             |SplitKSerial            |SplitKSerial           |SplitKSerial           |    |
  | |typename Operator             |Operator                |Operator               |Operator               |    |
  | |SharedMemoryClearOption::kNone|SharedMemoryClear       |SharedMemoryClear      |SharedMemoryClear      |    |
  | |bool GatherA = false          |GatherA                 |GatherA                |false                  |    |
  | |bool GatherB = false          |GatherB                 |GatherB                |false                  |    |
  | |bool ScatterD = false         |ScatterD                |ScatterD               |false                  |    |
  | |typename Enable = void        |NOT_SM80                |                       |                       |    |

#### 3.2.3 cutlass::gemm::threadblock::DefaultMma

* partial specialization
  * CudaCore_CRowMajorWith2Stage
  * CudaCore_CRowMajorMultiStage
  * CudaCore_CRowMajorDP4A
  * TensorCore_CRowMajor
  * TensorCore_CRowMajorFloat32
  * TensorCore_CRowMajorMultiStage
  * CMIK_With2Stage
  * CMIK_MultiStage
  * WMMA_2Stage
  * WMMA_1Stage

* define
  * define NONE_CLEAR=SharedMemoryClearOption::kNone
  * define CMIK=layout::ColumnMajorInterleaved<_InterleavedK_>

* Cuda Core(SIMT)

  ```C++
    static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
              || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
              "simt epilogue must be row major");
  ```

  | |DefaultMma<>                     | Cuda-Core-2-Stage     | Cuda-Core-MultiStage  |           DP4A         |
  |-|---------------------------------|-----------------------|-----------------------|------------------------|
  | |typename ElementA_               |ElementA               |ElementA               |int8_t                  |
  | |typename LayoutA_                |LayoutA                |LayoutA                |LayoutA                 |
  | |int kAlignmentA                  |kAlignmentA            |kAlignmentA            |kAlignmentA             |
  | |typename ElementB_               |ElementB               |ElementB               |int8_t                  |
  | |typename LayoutB_                |LayoutB                |LayoutB                |LayoutB                 |
  | |int kAlignmentB                  |kAlignmentB            |kAlignmentB            |kAlignmentB             |
  | |typename ElementAccumulator_     |ElementAccumulator     |ElementAccumulator     |ElementAccumulator      |
  | |typename LayoutC_                |LayoutC                |LayoutC                |layout::RowMajor        |
  |*|typename OperatorClass_          |___arch::OpClassSimt___|___arch::OpClassSimt___|___arch::OpClassSimt___ |
  | |typename ArchTag_                |ArchTag                |ArchTag                |ArchTag                 |
  | |typename ThreadblockShape_       |ThreadblockShape       |ThreadblockShape       |ThreadblockShape        |
  | |typename WarpShape_              |WarpShape              |WarpShape              |WarpShape               |
  | |typename InstructionShape_       |InstructionShape       |InstructionShape       |___GemmShape<1 1 4>___  |
  |*|int Stages                       |___2___                |Stages                 |___2___                 |
  | |typename Operator                |Operator               |Operator               |Operator                |
  | |bool AccumulatorsInRowMajor=false|false                  |false                  |false                   |
  | |SharedMemoryClearOption::kNone   |NONE_CLEAR               |NONE_CLEAR               |NONE_CLEAR                |
  | |bool GatherA = false             |GatherA                |GatherA                |false                   |
  | |bool GatherB = false             |GatherB                |GatherB                |false                   |

* Tensor Core

  | |DefaultMma<>                     | TensorCore RowMajor       | TensorCore RowMajor FP32  | TensorCore RowMajor MultiStage|
  |-|---------------------------------|---------------------------|---------------------------|---------------------------|
  |*|typename ElementA_               |ElementA                   |___float___                |ElementA                   |
  | |typename LayoutA_                |LayoutA                    |LayoutA                    |LayoutA                    |
  | |int kAlignmentA                  |kAlignmentA                |kAlignmentA                |kAlignmentA                |
  |*|typename ElementB_               |ElementB                   |___float___                |ElementB                   |
  | |typename LayoutB_                |LayoutB                    |LayoutB                    |LayoutB                    |
  | |int kAlignmentB                  |kAlignmentB                |kAlignmentB                |kAlignmentB                |
  |*|typename ElementAccumulator_     |ElementAccumulator         |___float___                |ElementAccumulator         |
  |*|typename LayoutC_                |___layout::RowMajor___     |___layout::RowMajor___     |___LayoutC___              |
  |*|typename OperatorClass_          |___arch::OpClassTensorOp___|___arch::OpClassTensorOp___|___arch::OpClassTensorOp___|
  | |typename ArchTag_                |ArchTag                    |ArchTag                    |ArchTag                    |
  | |typename ThreadblockShape_       |ThreadblockShape           |ThreadblockShape           |ThreadblockShape           |
  | |typename WarpShape_              |WarpShape                  |WarpShape                  |WarpShape                  |
  | |typename InstructionShape_       |InstructionShape           |InstructionShape           |InstructionShape           |
  |*|int Stages                       |___2___                    |___2___                    |___Stages___               |
  | |typename Operator                |Operator                   |Operator                   |Operator                   |
  |*|bool AccumulatorsInRowMajor=false|false                      |false                      |false                      |
  | |SharedMemoryClearOption::kNone   |NONE_CLEAR                   |NONE_CLEAR                   |___SharedMemoryClear___    |
  | |bool GatherA = false             |GatherA                    |GatherA                    |GatherA                    |
  | |bool GatherB = false             |GatherB                    |GatherB                    |GatherB                    |

* CMIK(layout::ColumnMajorInterleaved<_InterleavedK_>)

  | |DefaultMma<>                     | 2 Stage                   | MultiStage                |
  |-|---------------------------------|---------------------------|---------------------------|
  |*|typename ElementA_               |ElementA                   |ElementA                   |
  | |typename LayoutA_                |LayoutA                    |LayoutA                    |
  | |int kAlignmentA                  |kAlignmentA                |kAlignmentA                |
  |*|typename ElementB_               |ElementB                   |ElementB                   |
  | |typename LayoutB_                |LayoutB                    |LayoutB                    |
  | |int kAlignmentB                  |kAlignmentB                |kAlignmentB                |
  |*|typename ElementAccumulator_     |ElementAccumulator         |ElementAccumulator         |
  |*|typename LayoutC_                |___CMIK___                 |___CMIK___                 |
  |*|typename OperatorClass_          |___OperatorClass___        |___OperatorClass___        |
  | |typename ArchTag_                |ArchTag                    |ArchTag                    |
  | |typename ThreadblockShape_       |ThreadblockShape           |ThreadblockShape           |
  | |typename WarpShape_              |WarpShape                  |WarpShape                  |
  | |typename InstructionShape_       |InstructionShape           |InstructionShape           |
  |*|int Stages                       |___2___                    |___Stages___               |
  | |typename Operator                |Operator                   |Operator                   |
  |*|bool AccumulatorsInRowMajor=false|___true___                 |___true___                 |
  | |SharedMemoryClearOption::kNone   |NONE_CLEAR                 |NONE_CLEAR                 |
  | |bool GatherA = false             |false                      |false                      |
  | |bool GatherB = false             |false                      |false                      |

* WMMA

  | |DefaultMma<>                     |   2 stage        | 1 stage           |
  |-|---------------------------------|------------------|-------------------|
  |*|typename ElementA_               |ElementA          |ElementA           |
  | |typename LayoutA_                |LayoutA           |LayoutA            |
  | |int kAlignmentA                  |kAlignmentA       |kAlignmentA        |
  |*|typename ElementB_               |ElementB          |ElementB           |
  | |typename LayoutB_                |LayoutB           |LayoutB            |
  | |int kAlignmentB                  |kAlignmentB       |kAlignmentB        |
  |*|typename ElementAccumulator_     |ElementAccumulator|ElementAccumulator |
  |*|typename LayoutC_                |LayoutC           |LayoutC            |
  |*|typename OperatorClass_          |___arch::OpClassWmmaTensorOp___|___arch::OpClassWmmaTensorOp___|
  | |typename ArchTag_                |ArchTag           |ArchTag            |
  | |typename ThreadblockShape_       |ThreadblockShape  |ThreadblockShape   |
  | |typename WarpShape_              |WarpShape         |WarpShape          |
  | |typename InstructionShape_       |InstructionShape  |InstructionShape   |
  |*|int Stages                       |___2___           |___1___            |
  | |typename Operator                |Operator          |Operator           |
  |*|bool AccumulatorsInRowMajor=false|false             |false              |
  | |SharedMemoryClearOption::kNone   |NONE_CLEAR        |NONE_CLEAR         |
  | |bool GatherA = false             |false             |false              |
  | |bool GatherB = false             |false             |false              |

#### 3.2.4 cutlass::gemm::threadblock::DefaultMmaCore

* File

  ```Text
  ./cutlass/include/cutlass/gemm/threadblock/default_mma_core.h
  ```

* Partial specialization
  
  ```Text
  ./cutlass/include/cutlass/gemm/threadblock/default_mma_core_simt.h
  ./cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm70.h
  ./cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm75.h
  ./cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h
  ./cutlass/include/cutlass/gemm/threadblock/default_mma_core_sparse_sm80.h
  ./cutlass/include/cutlass/gemm/threadblock/default_mma_core_with_access_size.h
  ./cutlass/include/cutlass/gemm/threadblock/default_mma_core_with_reducrtion.h
  ./cutlass/include/cutlass/gemm/threadblock/default_mma_core_wmma.h
  ```

* Define
  * MAC::Type:

    ```C++
    typename Operator = typename platform::conditional<
    (platform::is_same<_OperatorClass_, _cutlass::arch::OpClassTensorOp_>::value) &&
        ( platform::is_same<_ElementA_, int8_t>::value ||
          platform::is_same<_ElementA_, int4b_t>::value ||
          platform::is_same<_ElementA_, uint8_t>::value ||
          platform::is_same<_ElementA_, uint4b_t>::value ),
    cutlass::arch::OpMultiplyAddSaturate,
    cutlass::arch::OpMultiplyAdd>::type,
    ```

  | |DefaultMmaCore<>                   |
  |-|-----------------------------------|
  | |typename Shape (blockthread scope) |
  | |typename WarpShape                 |
  | |typename InstructionShape          |
  | |typename ElementA                  |
  | |typename LayoutA                   |
  | |typename ElementB                  |
  | |typename LayoutB                   |
  | |typename ElementC                  |
  | |typename LayoutC                   |
  | |typename OperatorClass             |
  | |int Stages = 2                     |
  | |___typename Operator = MAC::Type___|
  | |___bool AccumulatorsInRowMajor = false___|
  | |___cutlass::arch::CacheOperation::Kind CacheOpA = cutlass::arch::CacheOperation::Global___|
  | |___cutlass::arch::CacheOperation::Kind CacheOpB = cutlass::arch::CacheOperation::Global___|
  | |___bool IsComplex = false___|
  | |___ComplexTransform TransformA = ComplexTransform::kNone___|
  | |___ComplexTransform TransformB = ComplexTransform::kNone___|

* SIMT partial specialization

  | | TT | NN | NT | TN |TT_A|NN_A|NT_A|TN_A|TT_IDP4A|NN_IDP4A|NT_IDP4A|TN_IDP4A|
  |-|----|----|----|----|----|----|----|----|--------|--------|--------|--------|
  | |Shape      |Shape      |Shape    |Shape      |Shape|Shape|Shape|Shape|Shape|Shape|Shape|Shape|
  | |WarpShape  |WarpShape  |WarpShape|WarpShape  |WarpShape|WarpShape|WarpShape|WarpShape|WarpShape|WarpShape|WarpShape|WarpShape|
  |*|GemmShape<1 1 1>|GemmShape<1 1 1>|GemmShape<1 1 1>|GemmShape<1 1 1>|GemmShape<1 1 1>|GemmShape<1 1 1>|GemmShape<1 1 1>|GemmShape<1 1 1>|GemmShape<1 1 4>|GemmShape<1 1 4>|GemmShape<1 1 4>|GemmShape<1 1 4>|
  |*|ElementA   |ElementA   |ElementA |ElementA   |ElementA|ElementA|ElementA|ElementA|int8t|int8t|int8t|int8t|
  |*|ColumnMajor|RowMajor   |RowMajor |ColumnMajor|AffineRank2ColumnMajor|AffineRank2RowMajor|AffineRank2RowMajor|AffineRank2ColumnMajor|ColumnMajor|RowMajor|RowMajor|ColumnMajor|
  |*|ElementB   |ElementB   |ElementB |ElementB   |ElementB|ElementB|ElementB|ElementB|int8t|int8t|int8t|int8t|
  |*|RowMajor   |ColumnMajor|RowMajor |ColumnMajor|AffineRank2RowMajor|AffineRank2ColumnMajor|AffineRank2RowMajor|AffineRank2ColumnMajor|RowMajor|ColumnMajor|RowMajor|ColumnMajor|
  | |ElementC   |ElementC   |ElementC |ElementC   |ElementC|ElementC|ElementC|ElementC|ElementC|ElementC|ElementC|ElementC|
  | |LayoutC    |LayoutC    |LayoutC  |LayoutC    |LayoutC|LayoutC|LayoutC|LayoutC|LayoutC|LayoutC|LayoutC|LayoutC|
  | |OpClassSimt|OpClassSimt|OpClassSimt|OpClassSimt|OpClassSimt|OpClassSimt|OpClassSimt|OpClassSimt|OpClassSimt|OpClassSimt|OpClassSimt|OpClassSimt|
  | |2|2|2|2|2|2|2|2|2|2|2|2|
  | |Operator|Operator|Operator|Operator|Operator|Operator|Operator|Operator|Operator|Operator|Operator|Operator|

* SM70

  | |       TT        |       NN        |       NT       |       TN        |
  |-|-----------------|-----------------|----------------|-----------------|
  | |Shape            |Shape            |Shape           |Shape            |
  | |WarpShape        |WarpShape        |WarpShape       |WarpShape        |
  | |GemmShape<8 8 4> |GemmShape<8 8 4> |GemmShape<8 8 4>|GemmShape<8 8 4> |
  | |ElementA         |ElementA         |ElementA        |ElementA         |
  |*|___ColumnMajor___|___RowMajor___   |___RowMajor___  |___ColumnMajor___|
  | |ElementB         |ElementB         |ElementB        |ElementB         |
  |*|___RowMajor___   |___ColumnMajor___|___RowMajor___  |___ColumnMajor___|
  | |ElementC         |ElementC         |ElementC        |ElementC         |
  | |LayoutC          |LayoutC          |LayoutC         |LayoutC          |
  | |OpClassTensorOp  |OpClassTensorOp  |OpClassTensorOp |OpClassTensorOp  |
  | |2                |2                |2               |2                |
  | |Operator         |Operator         |Operator        |Operator         |

* SM75

  | | TT             | NN             | NT             | TN             | TT_FP32            | NN_FP32            | NT_FP32            | TT_CMIK              |
  |-|----------------|----------------|----------------|----------------|--------------------|--------------------|--------------------|----------------------|
  | |Shape           |Shape           |Shape           |Shape           |Shape               |Shape               |Shape               |Shape                 |
  | |WarpShape       |WarpShape       |WarpShape       |WarpShape       |WarpShape           |WarpShape           |WarpShape           |WarpShape             |
  | |InstructionShape|InstructionShape|InstructionShape|InstructionShape|InstructionShape    |InstructionShape    |InstructionShape    |InstructionShape      |
  | |ElementA        |ElementA        |ElementA        |ElementA        |float               |float               |float               |ElementA              |
  |*|ColumnMajor     |RowMajor        |RowMajor        |ColumnMajor     |ColumnMajor         |RowMajor            |RowMajor            |ColumnMajorInterleaved<_InterleavedK_>|
  | |ElementB        |ElementB        |ElementB        |ElementB        |float               |float               |float               |ElementB              |
  |*|RowMajor        |ColumnMajor     |RowMajor        |ColumnMajor     |RowMajor            |ColumnMajor         |RowMajor            |RowMajorInterleaved<_InterleavedK_>   |
  | |ElementC        |ElementC        |ElementC        |ElementC        |float               |float               |float               |ElementC              |
  | |LayoutC         |LayoutC         |LayoutC         |LayoutC         |LayoutC             |LayoutC             |LayoutC             |LayoutC               |
  | |OpClassTensorOp |OpClassTensorOp |OpClassTensorOp |OpClassTensorOp |OpClassTensorOp     |OpClassTensorOp     |OpClassTensorOp     |arch::OpClassTensorOp |
  | |2               |2               |2               |2               |2                   |2                   |2                   |2                     |
  | |Operator        |Operator        |Operator        |Operator        |OpMultiplyAddFastF16|OpMultiplyAddFastF16|OpMultiplyAddFastF16|Operator              |
  | |false           |false           |false           |false           |false               |false               |false               |AccumulatorsInRowMajor|

* SM80
  * SM80-FP64-TensorCore
    | |TN_FP64    |TT_FP64    |NN_FP64    |NT_FP64    |TN_AFF_FP64           |TT_AFF_FP64           |NN_AFF_FP64           |NT_AFF_FP64       |
    |-|-----------|-----------|-----------|-----------|----------------------|----------------------|----------------------|------------------|
    | |Shape|Shape|Shape|Shape|Shape|Shape|Shape|Shape|
    | |WarpShape|WarpShape|WarpShape|WarpShape|WarpShape|WarpShape|WarpShape|WarpShape|
    | |InstructionShape|InstructionShape|InstructionShape|InstructionShape|InstructionShape|InstructionShape|InstructionShape|InstructionShape |
    | |double     |double     |double     |double     |double                |double                |double                |double             |
    | |ColumnMajor|ColumnMajor|RowMajor   |RowMajor   |AffineRank2ColumnMajor|AffineRank2ColumnMajor|AffineRank2RowMajor   |AffineRank2RowMajor|
    | |double     |double     |double     |double     |double                |double                |double                |double             |
    | |ColumnMajor|RowMajor   |ColumnMajor|RowMajor   |AffineRank2ColumnMajor|AffineRank2RowMajor   |AffineRank2ColumnMajor|AffineRank2RowMajor|
    | |double     |double     |double     |double     |double                |double                |double                |double             |
    | |LayoutC|LayoutC|LayoutC|LayoutC|LayoutC|LayoutC|LayoutC|LayoutC|
    | |OpClassTensorOp|OpClassTensorOp|OpClassTensorOp|OpClassTensorOp|OpClassTensorOp|OpClassTensorOp|OpClassTensorOp|OpClassTensorOp|
    | |Stages|Stages|Stages|Stages|Stages|Stages|Stages|Stages|
    | |Operator|Operator|Operator|Operator|Operator|Operator|Operator|Operator|
    | |false|false|false|false|false|false|false|false|
    | |CacheOpA|CacheOpA|CacheOpA|CacheOpA|CacheOpA|CacheOpA|CacheOpA|CacheOpA|
    | |CacheOpB|CacheOpB|CacheOpB|CacheOpB|CacheOpB|CacheOpB|CacheOpB|CacheOpB|

  * SM80-Complex-TensorCore
    | |  Complex<_float_>     |   Complex<_double_>   |
    |-|-----------------------|-----------------------|
    | |Shape                  |Shape                  |
    | |WarpShape              |WarpShape              |
    |*|___GemmShape<16 8 8>___|___GemmShape<8 8 4>___ |
    |*|___complex<_float_>___ |___complex<_double_>___|
    | |LayoutA                |LayoutA                |
    |*|___complex<_float_>___ |___complex<_double_>___|
    | |LayoutB                |LayoutB                |
    |*|___complex<_float_>___ |___complex<_double_>___|
    | |LayoutC                |LayoutC                |
    | |arch::OpClassTensorOp  |arch::OpClassTensorOp  |
    | |Stages                 |Stages                 |
    | |Operator               |Operator               |
    | |false                  |false                  |
    | |CacheOpA               |CacheOpA               |
    | |CacheOpB               |CacheOpB               |
    | |TransformA             |TransformA             |
    | |TransformB             |TransformB             |
    | |true                   |true                   |

  * SM80-Common
    * TensorCore: NN/NT/TN/TT/CMIK
    * CudaCore: NN/NT/TN/TT/NN_AFF/NT_AFF/TN_AFF/TT_AFF

#### 3.2.5 cutlass::transform::threadblock::PredicatedTileIterator

略

#### 3.2.6 cutlass::gemm::threadblock::MmaPipelined

核心计算：

```C++
/// Construct from tensor references
  CUTLASS_DEVICE
  MmaPipelined(
    typename Base::SharedStorage &shared_storage,       ///< Shared storage needed for internal use by threadblock-scoped GEMM
    int thread_idx,                                     ///< ID within the threadblock
    int warp_idx,                                       ///< ID of warp
    int lane_idx                                        ///< ID of each thread within a warp
  ):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
    smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx) {
    DUMP printf("MmaPipelined Enter: Base::WarpCount<%d, %d, %d> Warp%d\n", 
        Base::WarpCount::kM, Base::WarpCount::kN, Base::WarpCount::kK, warp_idx);

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;
    
    DUMP printf("  warp_idx_m=%d\n", warp_idx_m);
    DUMP printf("  warp_idx_n=%d\n", warp_idx_n);
    DUMP printf("  warp_idx_k=%d\n", warp_idx_k);

    DUMP printf("  Base::kWarpGemmIterations=%d\n", Base::kWarpGemmIterations);
    DUMP printf("  warp_tile_iterator_A_=%d, %d\n", warp_idx_m, Base::kWarpGemmIterations * warp_idx_k);
    DUMP printf("  warp_tile_iterator_B_=%d, %d\n", Base::kWarpGemmIterations * warp_idx_k, warp_idx_n);


    // 
    // warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx)
    // warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx)

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    DUMP printf("MmaPipelined Leave\n");
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &accum,                                 ///< destination accumulator tile
    IteratorA iterator_A,                             ///< iterator over A operand in global memory
    IteratorB iterator_B,                             ///< iterator over B operand in global memory
    FragmentC const &src_accum,                       ///< source accumulator tile
    TransformA transform_A = TransformA(),            ///< transformation applied to A fragment
    TransformB transform_B = TransformB()) {          ///< transformation applied to B fragment

    //
    // Prologue
    //

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    tb_frag_A.clear();
    tb_frag_B.clear();

    // The last kblock is loaded in the prolog
    iterator_A.load(tb_frag_A);
    iterator_B.load(tb_frag_B);

    ++iterator_A;
    ++iterator_B;

    this->smem_iterator_A_.store(transform_A(tb_frag_A));
    this->smem_iterator_B_.store(transform_B(tb_frag_B));

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    // Avoid reading out of bounds
    iterator_A.clear_mask(gemm_k_iterations <= 1);
    iterator_B.clear_mask(gemm_k_iterations <= 1);

    // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing 
    // shared memory loads (which have the tighest latency requirement).

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {

          // Write fragments to shared memory
          this->smem_iterator_A_.store(transform_A(tb_frag_A));
          this->smem_iterator_B_.store(transform_B(tb_frag_B));

          __syncthreads();
          
          ++this->smem_iterator_A_;
          ++this->smem_iterator_B_;

          // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          }
          else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        
        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {

          iterator_A.load(tb_frag_A);
          iterator_B.load(tb_frag_B);

          ++iterator_A;
          ++iterator_B;

          // Avoid reading out of bounds if this was the last loop iteration
          iterator_A.clear_mask(gemm_k_iterations <= 2);
          iterator_B.clear_mask(gemm_k_iterations <= 2);
        }

        warp_mma(accum, warp_frag_A[warp_mma_k % 2],
                 warp_frag_B[warp_mma_k % 2], accum);
      }
    }
  }
```
