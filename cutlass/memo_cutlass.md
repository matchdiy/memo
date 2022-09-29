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

### 3.2 cutlass::gemm::kernel::DefaultGemmUniversal

* class DefaultGemmUniversal<>
  * cutlass/gemm/kernel/default_gemm_universal.h
  * Partial specialization
    * define Complex = typename platform::enable_if<cutlass::is_complex<_ElementAccumulator_>::value>::type
    * define Real = !Complex

  | |Define                        |Real                   |Complex           |
  |-|------------------------------|-----------------------|------------------|
  | |typename ElementA_            |ElementA               |ElementA          |
  | |typename LayoutA_             |LayoutA                |LayoutA           |
  | |ComplexTransform TransformA   |ComplexTransform::kNone|TransformA        |
  | |int kAlignmentA               |kAlignmentA            |kAlignmentA       |
  | |typename ElementB_            |ElementB               |ElementB          |
  | |typename LayoutB_             |LayoutB                |LayoutB           |
  | |ComplexTransform TransformB   |ComplexTransform::kNone|TransformB        |
  | |int kAlignmentB               |kAlignmentB            |kAlignmentB       |
  | |typename ElementC_            |ElementC               |ElementC          |
  | |typename LayoutC_             |LayoutC                |LayoutC           |
  | |typename ElementAccumulator   |ElementAccumulator     |ElementAccumulator|
  | |typename OperatorClass        |OperatorClass          |OperatorClass     |
  | |typename ArchTag              |ArchTag                |ArchTag           |
  | |typename ThreadblockShape     |ThreadblockShape       |ThreadblockShape  |
  | |typename WarpShape            |WarpShape              |WarpShape         |
  | |typename InstructionShape     |InstructionShape       |InstructionShape  |
  | |typename EpilogueOutputOp     |EpilogueOutputOp       |EpilogueOutputOp  |
  | |typename ThreadblockSwizzle   |ThreadblockSwizzle     |ThreadblockSwizzle|
  | |int Stages                    |Stages                 |Stages            |
  | |typename Operator             |Operator               |Operator          |
  | |SharedMemoryClearOption::kNone|SharedMemoryClear      |SharedMemoryClear |
  | |bool GatherA = false          |GatherA                |false             |
  | |bool GatherB = false          |GatherB                |false             |
  | |bool ScatterD = false         |ScatterD               |false             |
  |*|typename Enable = void        |false                  |true              |

#### 3.2.1 cutlass::gemm::kernel::DefaultGemm

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

#### 3.2.2 cutlass::gemm::kernel::DefaultGemmComplex

略
