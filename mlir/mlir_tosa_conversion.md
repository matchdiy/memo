# MLIR Conversion 源码解读

## 1. 代码结构

- Sources: mlir/lib/Conversion/**
- Headers: /mlir/include/mlir/Conversion/[passes.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Conversion/Passes.td): 定义了所有的 Conversion Pass，默认是 mlir::Pass类型：
  - ConvertAffineToStandard
  - ConvertAffineForToGPU (mlir::FunctionPass)
  - ConvertAsyncToLLVM
  - ConvertComplexToLLVM
  - ConvertComplexToStandard (mlir::FunctionPass)
  - GpuToLLVMConversionPass
  - LowerHostCodeToLLVM
  - ConvertGpuOpsToNVVMOps
  - ConvertGpuOpsToROCDLOps
  - ConvertGPUToSPIRV
  - ConvertGpuLaunchFuncToVulkanLaunchFunc
  - ConvertVulkanLaunchFuncToVulkanCalls
  - ConvertLinalgToLLVM
  - ConvertLinalgToStandard
  - ConvertLinalgToSPIRV
  - ConvertMathToLibm
  - ConvertMathToLLVM (mlir::FunctionPass)
  - ConvertMathToSPIRV
  - ConvertMemRefToLLVM
  - ConvertMemRefToSPIRV
  - ConvertOpenACCToSCF
  - ConvertOpenACCToLLVM
  - ConvertOpenMPToLLVM
  - ConvertPDLToPDLInterp
  - SCFToSPIRV
  - SCFToStandard
  - ConvertSCFToOpenMP (mlir::FunctionPass)
  - ConvertParallelLoopToGpu
  - ConvertShapeToStandard
  - ConvertShapeConstraints
  - ConvertSPIRVToLLVM
  - ConvertStandardToLLVM
  - ConvertStandardToSPIRV
  - TosaToSCF
  - TosaToStandard
  - TosaToLinalgOnTensors (mlir::FunctionPass)
  - ConvertVectorToLLVM
  - ConvertVectorToROCDL
  - ConvertVectorToSPIRV
  - ConvertVectorToGPU (mlir::FunctionPass)
  - ConvertVectorToSCF (mlir::FunctionPass)
  - ConvertArmNeon2dToIntr

## 2. 源码分析

### 2.1 TosaToLinalg

- ___外部接口___
  - _std::unique_ptr\<Pass> createTosaToLinalgOnTensors()_
    - TableGen里边会调用这个接口，完成Pass的创建。
  - _void addTosaToLinalgOnTensorsPasses(OpPassManager &pm)_
    - pm.addNestedPass\<FuncOp>(createTosaMakeBroadcastablePass());
    - pm.addNestedPass\<FuncOp>(__createTosaToLinalgOnTensors__()); // 意图不明
    - Populates passes to convert from TOSA to Linalg on buffers. At the end of the pass, the function will only contain linalg ops or standard ops if the pipeline succeeds. 没找到调用的地方，应该是个预留。
- ___内部接口___
  - _void populateTosaToLinalgOnTensorsConversionPatterns(RewritePatternSet *patterns)_
    - 本接口会在创建TosaToLinalg的Pass中调用，完成TOSA方言到Linalg方言的转换，基于Pattern。
    - 以下OP不会转换成Linalg方言：
      - target.addLegalOp\<tosa::ApplyScaleOp>
      - target.addLegalOp\<tosa::IfOp>
      - target.addLegalOp\<tosa::ConstOp>
      - target.addLegalOp\<tosa::WhileOp>
    - 以下OP会转换成Linalg方言：
      - PointwiseConverter\<tosa::AddOp>
      - PointwiseConverter\<tosa::SubOp>
      - PointwiseConverter\<tosa::MulOp>
      - PointwiseConverter\<tosa::DivOp>
      - PointwiseConverter\<tosa::NegateOp>
      - PointwiseConverter\<tosa::PowOp>
      - PointwiseConverter\<tosa::ReciprocalOp>
      - PointwiseConverter\<tosa::RsqrtOp>
      - PointwiseConverter\<tosa::LogOp>
      - PointwiseConverter\<tosa::ExpOp>
      - PointwiseConverter\<tosa::AbsOp>
      - PointwiseConverter\<tosa::TanhOp>
      - PointwiseConverter\<tosa::BitwiseAndOp>
      - PointwiseConverter\<tosa::BitwiseOrOp>
      - PointwiseConverter\<tosa::BitwiseNotOp>
      - PointwiseConverter\<tosa::BitwiseXorOp>
      - PointwiseConverter\<tosa::LogicalAndOp>
      - PointwiseConverter\<tosa::LogicalNotOp>
      - PointwiseConverter\<tosa::LogicalOrOp>
      - PointwiseConverter\<tosa::LogicalXorOp>
      - PointwiseConverter\<tosa::CastOp>
      - PointwiseConverter\<tosa::LogicalLeftShiftOp>
      - PointwiseConverter\<tosa::LogicalRightShiftOp>
      - PointwiseConverter\<tosa::ArithmeticRightShiftOp>
      - PointwiseConverter\<tosa::SelectOp>
      - PointwiseConverter\<tosa::GreaterOp>
      - PointwiseConverter\<tosa::GreaterEqualOp>
      - PointwiseConverter\<tosa::EqualOp>
      - PointwiseConverter\<tosa::MaximumOp>
      - PointwiseConverter\<tosa::MinimumOp>
      - PointwiseConverter\<tosa::CeilOp>
      - PointwiseConverter\<tosa::FloorOp>
      - PointwiseConverter\<tosa::ClampOp>
      - PointwiseConverter\<tosa::ReluNOp>
      - PointwiseConverter\<tosa::SigmoidOp>
      - IdentityNConverter\<tosa::IdentityOp>
      - ReduceConverter\<tosa::ReduceAllOp>
      - ReduceConverter\<tosa::ReduceAnyOp>
      - ReduceConverter\<tosa::ReduceMinOp>
      - ReduceConverter\<tosa::ReduceMaxOp>
      - ReduceConverter\<tosa::ReduceSumOp>
      - ReduceConverter\<tosa::ReduceProdOp>
      - ArgMaxConverter
      - ConcatConverter
      - ConvConverter\<tosa::Conv2DOp>
      - ConvConverter\<tosa::DepthwiseConv2DOp>
      - TransposeConvConverter
      - GatherConverter
      - PadConverter
      - ReshapeConverter
      - RescaleConverter
      - ResizeConverter
      - ReverseConverter
      - TableConverter
      - TileConverter
      - TransposeConverter
      - MatMulConverter
      - Pool2dConverter\<tosa::AvgPool2dOp>
      - Pool2dConverter\<tosa::MaxPool2dOp>
      - FullyConnectedConverter

## 2.2 TosaToLinalg::PointwiseConverter\<SrcOp>

### 2.2.1 限制条件

- assert(operation->getNumResults() == 1)：只允许一个输出
- 输出必须是[ShapedType](https://mlir.llvm.org/doxygen/classmlir_1_1ShapedType.html) 类型，并且是静态Shape (hasStaticShape==True)。
- 支持Broadcast功能：
  - Scalar 输入的时候可以使用不同Rank的Tensor
  - 不是Scalar的话， 需要各个Tensor的Rank相同，不同的时候也需要填加Size=1的维度。
- 所有的输入和输出的数据类型必须相同（有量化功能的接口除外）。

### 2.2.2 主要逻辑

- 先给 Result 创建 linalg::InitTensorOp。
- 遍历所有的输入：
  - 如果输入和输出的 Shape 相同，那么根据rank创建Operand对应的 _indexingMap(AffineMap)_列表。
  - 否者会插入一个 tosa.Reshape 算子到 _operands_ 列表。
- 将输出插入到 _indexingMap_ 和 _operands_ 列表的最后。
- 创建 _linalg::GenericOp_。
- 调用 _createLinalgBodyCalculationForElementwiseOp()_,根据 tosa 算子的类型对应到 linalg 算子。
  - 对于Integer数据类型的 _tosa::MulOp_ 会考虑其量化需求, 将输入扩展成32位，然后使用 _tosa::ApplyScaleOp_ 来替换  _tosa::MulOp_ 。
  - 对于 i64 数据类型仍旧会转成 i32 ，导致错误 ```%4 = sexti %arg3 : i64 to i32```
  - 对于Integer数据类型的 _tosa::NegateOp_，如果带有 __量化__ 信息也会单独处理。
  
### 2.2.3 示例（tosa.mul）

- tosa.mul:

  ```tosa.mul
  //// mlir-opt %s --debug --tosa-to-linalg-on-tensors ////
  !type_mul_in0 = type tensor<1x256xi32>
  !type_mul_in1 = type tensor<128x1xi16>
  !type_mul_out = type tensor<128x256xi32>

  func @test_broadcast_mul(%arg0: !type_mul_in0, %arg1: !type_mul_in1) -> !type_mul_out {
    %0 = "tosa.mul"(%arg0, %arg1) {shift = 8 : i32}: (!type_mul_in0, !type_mul_in1) -> !type_mul_out
    return %0 : !type_mul_out
  }
  ```

- Summary:
  - Trying to match "{anonymous}::PointwiseConverter\<mlir::tosa::MulOp>"
    - Insert  : 'linalg.init_tensor'(0x562136e337b0)
    - Insert  : 'tosa.reshape'(0x562136e316b0)
    - Insert  : 'tosa.reshape'(0x562136e44290)
    - Insert  : 'std.constant'(0x562136e30b20)
    - Insert  : 'std.sexti'(0x562136e4f820)
    - Insert  : 'tosa.apply_scale'(0x562136e4f900)
    - Insert  : 'linalg.yield'(0x562136e4f9c0)
    - Insert  : 'linalg.generic'(0x562136e2da60)
    - Replace : 'tosa.mul'(0x562136e0bab0)

- Pattern Apply (1)
  - 给输出创建初始化Tensor：
    - ```%0 = "linalg.init_tensor"() {static_sizes = [128, 256]} : () -> tensor<128x256xi32>```
  - 插入一个tosa.reshape算子将第一个输入的Shpae从[1x256]变成[256]
    - ```%1 = "tosa.reshape"(%arg0) {new_shape = [256]} : (tensor<1x256xi32>) -> tensor<256xi32>```
    - tosa.reshape 触发 tosa-to-linalg Pattern，会插入 linalg.tensor_collapse_shape算子
    - ```%1 = "linalg.tensor_collapse_shape"(%arg0) {reassociation = [[0, 1]]} : (tensor<1x256xi32>) -> tensor<256xi32>```
  - Result:

    ```IR Dump After Pattern Apply 1
    builtin.func @test_broadcast_mul(%arg0: tensor<1x256xi32>, %arg1: tensor<128x1xi16>) -> tensor<128x256xi32> {
      %0 = linalg.init_tensor [128, 256] : tensor<128x256xi32>
      %1 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<1x256xi32> into tensor<256xi32>
      %2 = "tosa.reshape"(%arg0) {new_shape = [256]} : (tensor<1x256xi32>) -> tensor<256xi32>
      %3 = "tosa.reshape"(%arg1) {new_shape = [128]} : (tensor<128x1xi16>) -> tensor<128xi16>
      %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<256xi32>, tensor<128xi16>) outs(%0 : tensor<128x256xi32>) {
      ^bb0(%arg2: i32, %arg3: i16, %arg4: i32):  // no predecessors
        %c8_i8 = constant 8 : i8
        %6 = sexti %arg3 : i16 to i32
        %7 = "tosa.apply_scale"(%arg2, %6, %c8_i8) {double_round = false} : (i32, i32, i8) -> i32
        linalg.yield %7 : i32
      } -> tensor<128x256xi32>
      %5 = "tosa.mul"(%arg0, %arg1) {shift = 8 : i32} : (tensor<1x256xi32>, tensor<128x1xi16>) -> tensor<128x256xi32>
      return %5 : tensor<128x256xi32>
    }
    ```

- Pattern Apply (2)
  - 插入一个tosa.reshape算子将第二个输入的Shpae从[128x1]变成[128]
    - ```%3 = "tosa.reshape"(%arg1) {new_shape = [128]} : (tensor<128x1xi16>) -> tensor<128xi16>```
    - 触发 tosa-to-linalg Pattern，会插入 linalg.tensor_collapse_shape算子
    - ```%3 = "linalg.tensor_collapse_shape"(%arg1) {reassociation = [[0, 1]]} : (tensor<128x1xi16>) -> tensor<128xi16>```
  - Result:

    ```IR Dump After Pattern Apply 2
    builtin.func @test_broadcast_mul(%arg0: tensor<1x256xi32>, %arg1: tensor<128x1xi16>) -> tensor<128x256xi32> {
      %0 = linalg.init_tensor [128, 256] : tensor<128x256xi32>
      %1 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<1x256xi32> into tensor<256xi32>
      %2 = "tosa.reshape"(%arg0) {new_shape = [256]} : (tensor<1x256xi32>) -> tensor<256xi32>
      %3 = linalg.tensor_collapse_shape %arg1 [[0, 1]] : tensor<128x1xi16> into tensor<128xi16>
      %4 = "tosa.reshape"(%arg1) {new_shape = [128]} : (tensor<128x1xi16>) -> tensor<128xi16>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %4 : tensor<256xi32>, tensor<128xi16>) outs(%0 : tensor<128x256xi32>) {
      ^bb0(%arg2: i32, %arg3: i16, %arg4: i32):  // no predecessors
        %c8_i8 = constant 8 : i8
        %7 = sexti %arg3 : i16 to i32
        %8 = "tosa.apply_scale"(%arg2, %7, %c8_i8) {double_round = false} : (i32, i32, i8) -> i32
        linalg.yield %8 : i32
      } -> tensor<128x256xi32>
      %6 = "tosa.mul"(%arg0, %arg1) {shift = 8 : i32} : (tensor<1x256xi32>, tensor<128x1xi16>) -> tensor<128x256xi32>
      return %6 : tensor<128x256xi32>
    }
    ```

- Pattern Apply (3)
  - shift = 8:
    - ```%7 = "std.constant"() {value = 8 : i8} : () -> i8```
  - 将第二个输入从 i16 转换成 i32:
    - ```%8 = "std.sexti"(%arg3) : (i16) -> i32```
  - Integer的 tosa.mul 会使用支持量化功能的ApplyScale完成：
    - ```%9 = "tosa.apply_scale"(%arg2, %8, %7) {double_round = false} : (i32, i32, i8) -> i32```
  - 创建YieldOp：nestedBuilder.create<linalg::YieldOp>(loc, opResult);
    - ```"linalg.yield"(%9) : (i32) -> ()```
    - ```"std.return"(%6) : (tensor<128x256xi32>) -> ()```
  - Result

    ```IR Dump After Pattern Apply 3
    builtin.func @test_broadcast_mul(%arg0: tensor<1x256xi32>, %arg1: tensor<128x1xi16>) -> tensor<128x256xi32> {
      %0 = linalg.init_tensor [128, 256] : tensor<128x256xi32>
      %1 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<1x256xi32> into tensor<256xi32>
      %2 = "tosa.reshape"(%arg0) {new_shape = [256]} : (tensor<1x256xi32>) -> tensor<256xi32>
      %3 = linalg.tensor_collapse_shape %arg1 [[0, 1]] : tensor<128x1xi16> into tensor<128xi16>
      %4 = "tosa.reshape"(%arg1) {new_shape = [128]} : (tensor<128x1xi16>) -> tensor<128xi16>
      %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%2, %4 : tensor<256xi32>, tensor<128xi16>) outs(%0 : tensor<128x256xi32>) {
      ^bb0(%arg2: i32, %arg3: i16, %arg4: i32):  // no predecessors
        %c8_i8 = constant 8 : i8
        %7 = sexti %arg3 : i16 to i32
        %8 = "tosa.apply_scale"(%arg2, %7, %c8_i8) {double_round = false} : (i32, i32, i8) -> i32
        linalg.yield %8 : i32
      } -> tensor<128x256xi32>
      %6 = "tosa.mul"(%arg0, %arg1) {shift = 8 : i32} : (tensor<1x256xi32>, tensor<128x1xi16>) -> tensor<128x256xi32>
      return %6 : tensor<128x256xi32>
    }
    ```

- 最终输出结果：
  - Replace tosa算子：_rewriter.replaceOp(operation, linalgOp->getResults())_

    ```Convert
    #map0 = affine_map<(d0, d1) -> (d1)>
    #map1 = affine_map<(d0, d1) -> (d0)>
    #map2 = affine_map<(d0, d1) -> (d0, d1)>
    builtin.module  {
      builtin.func @test_broadcast_mul(%arg0: tensor<1x256xi32>, %arg1: tensor<128x1xi16>) -> tensor<128x256xi32> {
        %0 = linalg.init_tensor [128, 256] : tensor<128x256xi32>
        %1 = linalg.tensor_collapse_shape %arg0 [[0, 1]] : tensor<1x256xi32> into tensor<256xi32>
        %2 = linalg.tensor_collapse_shape %arg1 [[0, 1]] : tensor<128x1xi16> into tensor<128xi16>
        %3 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%1, %2 : tensor<256xi32>, tensor<128xi16>) outs(%0 : tensor<128x256xi32>) {
        ^bb0(%arg2: i32, %arg3: i16, %arg4: i32):  // no predecessors
          %c8_i8 = constant 8 : i8
          %4 = sexti %arg3 : i16 to i32
          %5 = "tosa.apply_scale"(%arg2, %4, %c8_i8) {double_round = false} : (i32, i32, i8) -> i32
          linalg.yield %5 : i32
        } -> tensor<128x256xi32>
        return %3 : tensor<128x256xi32>
      }
    }
    ```

## 2.3 TosaToLinalg::IdentityNConverter\<SrcOp>

- 返回一个和Input一样的Tensor。
- 输入：

  ```tosa
  func @test_identity(%arg0: tensor<1xf32>, %arg1: tensor<1xi32>) -> (tensor<1xf32>, tensor<1xi32>) {
    %0 = "tosa.identity"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
    %1 = "tosa.identity"(%arg1) : (tensor<1xi32>) -> tensor<1xi32>

    // CHECK: return %arg0, %arg1
    return %0, %1 : tensor<1xf32>, tensor<1xi32>
  }
  ```

- 输出

  ``` linalg
  builtin.module  {
    builtin.func @test_identity(%arg0: tensor<1xf32>, %arg1: tensor<1xi32>) ->   (tensor<1xf32>, tensor<1xi32>) {
      return %arg0, %arg1 : tensor<1xf32>, tensor<1xi32>
    }
  }
  ```

## 2.4 TosaToLinalg::ReduceConverter\<SrcOp>

__全部的Reduce算子：__

- tosa.reduce_all：ReduceConverter\<tosa::ReduceAllOp>：AND
- tosa.reduce_any：ReduceConverter\<tosa::ReduceAnyOp>: OR
- tosa.reduce_min：ReduceConverter\<tosa::ReduceMinOp>: MIN
- tosa.reduce_max：ReduceConverter\<tosa::ReduceMaxOp>: MAX
- tosa.reduce_sum: ReduceConverter\<tosa::ReduceSumOp>: SUM
- tosa.reduce_prod: ReduceConverter\<tosa::ReduceProdOp>: Product(点积)

__Attrbuite:__

- axis：mlir::IntegerAttr (64-bit signless integer attribute)
- 这个属性是一个Integer，不是Array，也就是说目前只支持某一个维度的Reduce，__这是什么原因？__

__Sample 1:__

```%0 = "tosa.reduce_sum"(%arg0) {axis = 3 : i64} : (tensor<5x4x3x2xf32>) -> tensor<5x4x3x1xf32>```

Tosa-to-Linaly

 ```IR
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
builtin.module  {
  builtin.func @test_reduce(%arg0: tensor<5x4x3x2xf32>) {
    %0 = linalg.init_tensor [5, 4, 3] : tensor<5x4x3xf32>
    %cst = constant 0.000000e+00 : f32
    %1 = linalg.fill(%cst, %0) : f32, tensor<5x4x3xf32> -> tensor<5x4x3xf32> 
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<5x4x3x2xf32>) outs(%1 : tensor<5x4x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %3 = addf %arg1, %arg2 : f32
      linalg.yield %3 : f32
    } -> tensor<5x4x3xf32>
    return
  }
}
```

__Sample 2:__

```MLIR
func @test_reduce(%arg0: tensor<5x4x3x2xf32>) -> () {
  %0 = "tosa.reduce_sum"(%arg0) {axis = 3 : i64} : (tensor<5x4x3x2xf32>) -> tensor<5x4x3xf32>
  %1 = "tosa.reduce_sum"(%0) {axis = 2 : i64} : (tensor<5x4x3xf32>) -> tensor<5x4xf32>
  return
}
```

Tosa-To-Linalg:

```
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
builtin.module  {
  builtin.func @test_reduce(%arg0: tensor<5x4x3x2xf32>) {
    %0 = linalg.init_tensor [5, 4, 3] : tensor<5x4x3xf32>
    %cst = constant 0.000000e+00 : f32
    %1 = linalg.fill(%cst, %0) : f32, tensor<5x4x3xf32> -> tensor<5x4x3xf32> 
    %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%arg0 : tensor<5x4x3x2xf32>) outs(%1 : tensor<5x4x3xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %6 = addf %arg1, %arg2 : f32
      linalg.yield %6 : f32
    } -> tensor<5x4x3xf32>
    %3 = linalg.init_tensor [5, 4] : tensor<5x4xf32>
    %cst_0 = constant 0.000000e+00 : f32
    %4 = linalg.fill(%cst_0, %3) : f32, tensor<5x4xf32> -> tensor<5x4xf32> 
    %5 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<5x4x3xf32>) outs(%4 : tensor<5x4xf32>) {
    ^bb0(%arg1: f32, %arg2: f32):  // no predecessors
      %6 = addf %arg1, %arg2 : f32
      linalg.yield %6 : f32
    } -> tensor<5x4xf32>
    return
  }
}
```

作业：后续需要如何变换才能生成支持多维度的 Reduction？？？

## 2.5 TosaToLinalg::ConvConverter\<SrcOp>

### 2.5.1 Conv2D

__限制条件__：

- 输入是三个Teansor：
  - input: 必须是 4D， 并且必须是静态形状
  - weight：必须是 4D，并且必须是静态形状
  - bias: 必须是 1D，并且必须是静态形状
- 默认Layout是 ___linalg.conv_2d_input_nhwc_filter_ohwi_poly___，其他还有：
  - linalg.conv_2d
  - linalg.conv_2d_nchw
  - linalg.conv_2d_nhwc
  - linalg.conv_2d_input_nhwc_filter_ohwi_poly
  - linalg.conv_2d_input_nhwc_filter_ohwi_poly_q
  - linalg.conv_2d_input_nchw_filter_hwcf
  - linalg.conv_2d_input_nhwc_filter_hwcf
  - linalg.depthwise_conv_2d_input_nhwc_filter_hwc_poly
  - linalg.depthwise_conv_2d_input_nhwc_filter_hwcf
  - linalg.depthwise_conv_2d_input_nhwc_filter_hwc

__三个分支：__

- 非量化 tosa::Conv2DOp
  - linalg.conv_2d_input_nhwc_filter_ohwi_poly
- 量化的 tosa::Conv2DOp
  - linalg.conv_2d_input_nhwc_filter_ohwi_poly_q
- 非量化 tosa::DepthwiseConv2DOp
  - linalg.depthwise_conv_2d_input_nhwc_filter_hwcf

__Samples:__

- Pattern Apply:

  ```Tosa
  !type_f = type tensor<1x47x40x28xf32>
  !type_w = type tensor<28x3x3x28xf32>
  !type_b = type tensor<28xf32>
  !type_o = type tensor<1x45x40x28xf32>
  func @conv2d_padded_f32(%input: !type_f, %weights: !type_w, %bias: !type_b) -> () {
    %0 = "tosa.conv2d"(%input, %weights, %bias) {pad = [1, 1, 1, 1], stride = [1, 1], dilation = [2, 1]} : (!type_f, !type_w, !type_b) -> (!type_o)
    return
  }
  ```

- Tosa-to-Linalg:

  ```Linalg
  #map0 = affine_map<(d0, d1, d2, d3) -> (d3)>
  #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
  builtin.module  {
    builtin.func @conv2d_padded_f32(%arg0: tensor<1x47x40x28xf32>, %arg1: tensor<28x3x3x28xf32>, %arg2: tensor<28xf32>) {
      %cst = constant 0.000000e+00 : f32
      %0 = linalg.pad_tensor %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]  {
      ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):  // no predecessors
        linalg.yield %cst : f32
      } : tensor<1x47x40x28xf32> to tensor<1x49x42x28xf32> 
      %1 = linalg.init_tensor [1, 45, 40, 28] : tensor<1x45x40x28xf32>
      %2 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<28xf32>) outs(%1 : tensor<1x45x40x28xf32>) {
      ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
        linalg.yield %arg3 : f32
      } -> tensor<1x45x40x28xf32>
      %3 = linalg.conv_2d_input_nhwc_filter_ohwi_poly {dilations = dense<[2, 1]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%0, %arg1 : tensor<1x49x42x28xf32>, tensor<28x3x3x28xf32>) outs(%2 : tensor<1x45x40x28xf32>) -> tensor<1x45x40x28xf32>
      return
    }
  }
  ```

__还很简陋:__

- Tosa层没有数据Format的描述，默认是 NHWC+CoRSCi 的格式。
- Linalg上支持更多数据格式的卷积。

### 2.5.2 Conv3D

_Tosa上目前还没有Conv3D，但是Linalg上是有一些的，目测后面应该会有支持。_

- linalg.conv_3d
- linalg.conv_3d_input_ncdhw_filter_dhwcf
- linalg.conv_3d_input_ndhwc_filter_dhwcf
- linalg.conv_3d_ncdhw
- linalg.conv_3d_ndhwc

## 2.6 TosaToLinalg::ArgMaxConverter

计算最大值以及最大值所在的索引
```%0 = "tosa.argmax"(%arg0) { axis = 0 : i64} : (tensor<3x2xf32>)  -> (tensor<2xi32>)```

Tosa-to_Linalg:

```IR
#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
builtin.module  {
  builtin.func @argmax(%arg0: tensor<3x2xf32>) {
    %0 = linalg.init_tensor [2] : tensor<2xi32>
    %c0_i32 = constant 0 : i32
    %1 = linalg.fill(%c0_i32, %0) : i32, tensor<2xi32> -> tensor<2xi32> 
    %2 = linalg.init_tensor [2] : tensor<2xf32>
    %cst = constant -3.40282347E+38 : f32
    %3 = linalg.fill(%cst, %2) : f32, tensor<2xf32> -> tensor<2xf32> 
    %4:2 = linalg.generic {indexing_maps = [#map0, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<3x2xf32>) outs(%1, %3 : tensor<2xi32>, tensor<2xf32>) {
    ^bb0(%arg1: f32, %arg2: i32, %arg3: f32):  // no predecessors
      %5 = linalg.index 0 : index
      %6 = index_cast %5 : index to i32
      %7 = cmpf ogt, %arg1, %arg3 : f32
      %8 = select %7, %arg1, %arg3 : f32
      %9 = select %7, %6, %arg2 : i32
      linalg.yield %9, %8 : i32, f32
    } -> (tensor<2xi32>, tensor<2xf32>)
    return
  }
}
```

## 2.7 TosaToLinalg::ConcatConverter

在指定的维度上拼接两个Tensor，使用的是 __tensor.insert_slice(DeSlice)__ 进行转换。

```TOSA
%0 = "tosa.concat"(%arg0, %arg1) { axis = 0 : i64} : (tensor<5x32xf32>, tensor<6x32xf32>)  -> (tensor<11x32xf32>)
```

Tosa-to-Linalg：

```IR
builtin.module  {
  builtin.func @concat(%arg0: tensor<5x32xf32>, %arg1: tensor<6x32xf32>) {
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c0_0 = constant 0 : index
    %c0_1 = constant 0 : index
    %0 = tensor.dim %arg0, %c0_1 : tensor<5x32xf32>
    %c1_2 = constant 1 : index
    %1 = tensor.dim %arg0, %c1_2 : tensor<5x32xf32>
    %2 = tensor.dim %arg1, %c0 : tensor<6x32xf32>
    %3 = addi %0, %2 : index
    %4 = linalg.init_tensor [11, 32] : tensor<11x32xf32>
    %cst = constant 0.000000e+00 : f32
    %5 = linalg.fill(%cst, %4) : f32, tensor<11x32xf32> -> tensor<11x32xf32> 
    %6 = tensor.dim %arg0, %c0 : tensor<5x32xf32>
    %7 = tensor.insert_slice %arg0 into %5[%c0_0, %c0_0] [%6, %1] [%c1, %c1] : tensor<5x32xf32> into tensor<11x32xf32>
    %8 = addi %c0_0, %6 : index
    %9 = tensor.dim %arg1, %c0 : tensor<6x32xf32>
    %10 = tensor.insert_slice %arg1 into %7[%8, %c0_0] [%9, %1] [%c1, %c1] : tensor<6x32xf32> into tensor<11x32xf32>
    %11 = addi %8, %9 : index
    return
  }
}
```

## 2.8 TosaToLinalg::TransposeConvConverter

Tosa的思路和HLO不同，HLO会把Transpose放到卷积外部，从而使卷积的语义更加单纯。而Tosa把BPI当成不同的卷积，转换成Linalg的时候再做Reverse。这样对比的话Tosa的粒度比HLO更粗一些。

__TransposeConv（BPI）转置卷积目前有很多限制：__

```Limit
// We have not solved for stride / dilation yet. Dilation should be
// straight forward but stride is more complicated. Linalg work is likely
// required for efficient implementation.
if (llvm::any_of(stride, [](int64_t v) { return v != 1; }))
  return failure();
if (llvm::any_of(dilation, [](int64_t v) { return v != 1; }))
  return failure();
```

__Sample:__

```Tosa
%0 = "tosa.transpose_conv2d"(%arg0, %arg1, %arg2) {dilation = [1, 1], out_pad = [0, 0], out_shape = [1, 14, 14, 4], stride = [1, 1]}
 : (tensor<1x12x12x2xf32>, tensor<4x3x3x2xf32>, tensor<4xf32>) -> tensor<1x14x14x4xf32>
```

- Rotation180:
  - 对Weight的 _H_ 维度进行Revert
  - 对Weight的 _W_ 维度进行Revert
- 对Input的 _H_ 和 _W_ 进行Padding:
  - Padding[0] = kernelHeight - 1 - out_pad[0]
  - Padding[1] = (outputHeight + kernelHeight d- 1) - Padding[0] - inputHeight
  - Padding[2] = kernelWidth - 1 - out_pad[1]
  - Padding[3] = (outputWidth + kernelWidth - 1) - Padding[2] - inputWidth
- 对Bias进行广播
- 调用linalg.conv_2d_input_nhwc_filter_ohwi_poly

```IR
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, -d1 + 2, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, -d2 + 2, d3)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d3)>
builtin.module  {
  builtin.func @transpose_conv(%arg0: tensor<1x12x12x2xf32>, %arg1: tensor<4x3x3x2xf32>, %arg2: tensor<4xf32>) {
    %0 = linalg.init_tensor [4, 3, 3, 2] : tensor<4x3x3x2xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<4x3x3x2xf32>) outs(%0 : tensor<4x3x3x2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      linalg.yield %arg3 : f32
    } -> tensor<4x3x3x2xf32>
    %2 = linalg.init_tensor [4, 3, 3, 2] : tensor<4x3x3x2xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%1 : tensor<4x3x3x2xf32>) outs(%2 : tensor<4x3x3x2xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      linalg.yield %arg3 : f32
    } -> tensor<4x3x3x2xf32>
    %cst = constant 0.000000e+00 : f32
    %4 = linalg.pad_tensor %arg0 low[0, 2, 2, 0] high[0, 2, 2, 0]  {
    ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<1x12x12x2xf32> to tensor<1x16x16x2xf32> 
    %5 = linalg.init_tensor [1, 14, 14, 4] : tensor<1x14x14x4xf32>
    %6 = linalg.generic {indexing_maps = [#map3, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<4xf32>) outs(%5 : tensor<1x14x14x4xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      linalg.yield %arg3 : f32
    } -> tensor<1x14x14x4xf32>
    %7 = linalg.conv_2d_input_nhwc_filter_ohwi_poly {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%4, %3 : tensor<1x16x16x2xf32>, tensor<4x3x3x2xf32>) outs(%6 : tensor<1x14x14x4xf32>) -> tensor<1x14x14x4xf32>
    return
  }
}
```

## 2.9 TosaToLinalg::GatherConverter

__当前的限制:__

- Input必须是 _3D_ tensor， 或者是 _unranked_ tensor。
- Indexes必须是 _2D_ tensor。
- 当前代码中的处理是HardCode的方式：

  ```Code
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, ArrayRef<Type>({resultTy}), ValueRange{indices},
        ValueRange{initTensor}, affineMaps,
        getNParallelLoopsAttrs(resultTy.getRank()),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto indexValue = args[0];
          auto index0 = rewriter.create<linalg::IndexOp>(loc, 0);  // OutIndex0 = InputIndex0
          Value index1 = rewriter.create<IndexCastOp>(
              loc, rewriter.getIndexType(), indexValue);           // OutIndex1 = indices[0]
          auto index2 = rewriter.create<linalg::IndexOp>(loc, 2);  // OutIndex2 = InputIndex2
          Value extract = rewriter.create<tensor::ExtractOp>(
              loc, input, ValueRange{index0, index1, index2});
          rewriter.create<linalg::YieldOp>(loc, extract);
        });
  ```

__Sample:__

```Gather
%0 = "tosa.gather"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<3x6xi32>)  -> (tensor<3x6x5xi32>)
```

```Linalg
#map0 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
builtin.module  {
  builtin.func @gather_static(%arg0: tensor<3x4x5xi32>, %arg1: tensor<3x6xi32>) {
    %0 = linalg.init_tensor [3, 6, 5] : tensor<3x6x5xi32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg1 : tensor<3x6xi32>) outs(%0 : tensor<3x6x5xi32>) {
    ^bb0(%arg2: i32, %arg3: i32):  // no predecessors
      %2 = linalg.index 0 : index
      %3 = index_cast %arg2 : i32 to index
      %4 = linalg.index 2 : index
      %5 = tensor.extract %arg0[%2, %3, %4] : tensor<3x4x5xi32>
      linalg.yield %5 : i32
    } -> tensor<3x6x5xi32>
    return
  }
} 
```

## 2.10 TosaToLinalg::PadConverter

__限制条件__:

- tosa.pad 和 linalg.pad 对于参数的描述是不同的
  - tosa.pad: Nx2 tensor
  - linalg.pad: low[N] and hight[N]

__Sample:__

```Tosa
!type_i = type tensor<1x2x128xf32>
!type_p = type tensor<3x2xi32>
!type_o = type tensor<4x9x128xf32>
func @pad_float(%arg0 : !type_i) -> (!type_o) {
  %0 = constant dense<[[1, 2], [3, 4], [0, 0]]> : !type_p
  %1 = "tosa.pad"(%arg0, %0)  : (!type_i, !type_p)  -> (!type_o)
  return %1 : !type_o
}
```

```Linalg
builtin.module  {
  builtin.func @pad_float(%arg0: tensor<1x2x128xf32>) -> tensor<4x9x128xf32> {
    %cst = constant dense<[[1, 2], [3, 4], [0, 0]]> : tensor<3x2xi32>
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    %c0_0 = constant 0 : index
    %c1_i32 = constant 1 : i32
    %c2_i32 = constant 2 : i32
    %c1_1 = constant 1 : index
    %c2 = constant 2 : index
    %c1_2 = constant 1 : index
    %c3_i32 = constant 3 : i32
    %c4_i32 = constant 4 : i32
    %c3 = constant 3 : index
    %c4 = constant 4 : index
    %c2_3 = constant 2 : index
    %c0_i32 = constant 0 : i32
    %c0_i32_4 = constant 0 : i32
    %c0_5 = constant 0 : index
    %c0_6 = constant 0 : index
    %cst_7 = constant 0.000000e+00 : f32
    %0 = linalg.pad_tensor %arg0 low[%c1_1, %c3, %c0_5] high[%c2, %c4, %c0_6]  {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):  // no predecessors
      linalg.yield %cst_7 : f32
    } : tensor<1x2x128xf32> to tensor<4x9x128xf32> 
    return %0 : tensor<4x9x128xf32>
  }
}
```

## 2.11 TosaToLinalg::ReshapeConverter

__语义:__

只是改变Shape，没有内存的copy，不是Transpose（燧原的DMA使用了Reshape很不专业）。

__变换:__

- linalg.tensor_collapse_shape：已输入维度信息为基础
- linalg.tensor_expand_shape：已输出维度信息为基础

```Tosa
// CHECK-LABEL: @test_reshape_downrank_6D
func @test_reshape_downrank_6D(%arg0: tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32> {
  %0 = "tosa.reshape"(%arg0) {new_shape = [6, 5, 77]} : (tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32>
  return %0 : tensor<6x5x77xf32>
}

// CHECK-LABEL: @test_reshape_uprank
func @test_reshape_uprank(%arg0: tensor<6x5x77xf32>) -> tensor<1x2x3x5x7x11xf32> {
  %0 = "tosa.reshape"(%arg0) {new_shape = [1, 2, 3, 5, 7, 11]} : (tensor<6x5x77xf32>) -> tensor<1x2x3x5x7x11xf32>
  return %0 : tensor<1x2x3x5x7x11xf32>
}

// CHECK-LABEL: @test_reshape_samerank
func @test_reshape_samerank(%arg0: tensor<1x2x3x4x5x6x7xf32>) -> tensor<7x5x3x1x2x4x6xf32> {
  %0 = "tosa.reshape"(%arg0) {new_shape = [7, 5, 3, 1, 2, 4 , 6]} : (tensor<1x2x3x4x5x6x7xf32>) -> tensor<7x5x3x1x2x4x6xf32>
  return %0 : tensor<7x5x3x1x2x4x6xf32>
}

```

```Linalg
builtin.module  {
  builtin.func @test_reshape_downrank_6D(%arg0: tensor<1x2x3x5x7x11xf32>) -> tensor<6x5x77xf32> {
    %0 = linalg.tensor_collapse_shape %arg0 [[0, 1, 2], [3], [4, 5]] : tensor<1x2x3x5x7x11xf32> into tensor<6x5x77xf32>
    return %0 : tensor<6x5x77xf32>
  }
  builtin.func @test_reshape_uprank(%arg0: tensor<6x5x77xf32>) -> tensor<1x2x3x5x7x11xf32> {
    %0 = linalg.tensor_expand_shape %arg0 [[0, 1, 2], [3], [4, 5]] : tensor<6x5x77xf32> into tensor<1x2x3x5x7x11xf32>
    return %0 : tensor<1x2x3x5x7x11xf32>
  }
  builtin.func @test_reshape_samerank(%arg0: tensor<1x2x3x4x5x6x7xf32>) -> tensor<7x5x3x1x2x4x6xf32> {
    %0 = linalg.tensor_collapse_shape %arg0 [[0, 1, 2, 3, 4, 5, 6]] : tensor<1x2x3x4x5x6x7xf32> into tensor<5040xf32>
    %1 = linalg.tensor_expand_shape %0 [[0, 1, 2, 3, 4, 5, 6]] : tensor<5040xf32> into tensor<7x5x3x1x2x4x6xf32>
    return %1 : tensor<7x5x3x1x2x4x6xf32>
  }
}
```

## 2.12 TosaToLinalg::RescaleConverter

对已经量化过的数据重新缩放到一个新的范围。

__支持范围：__
|Mode|Input|Output|
|----|-----|------|
|    |int8 |int8  |
|    |int8 |int16 |
|    |int8 |int32 |
|    |int16|int8  |
|    |int16|int16 |
|    |int16|int32 |
|    |int32|int8  |
|    |int32|int16 |
|    |int32|int32 |
|    |int48|int8  |
|    |int48|int16 |
|    |int48|int32 |
|    |uint8|int8  |
|    |int8 |uint8 |

__支持属性：__
|Attribute   |MLIR Type          |Description|
|------------|-------------------|-----------|
|input_zp    |::mlir::IntegerAttr|32bits Integer, 输入数据零点|
|output_zp   |::mlir::IntegerAttr|32bits Integer, 输出数据零点|
|multiplier  |::mlir::ArrayAttr  |32bits Integer Array， per_channel=true，那么需要每个维度都提供|
|shift       |::mlir::ArrayAttr  |32bits Integer Array， per_channel=true，那么需要每个维度都提供|
|scale32     |::mlir::BoolAttr   |代码中没有使用，有个条件检查：tosa.rescale requires scale32 for double_round to be true|
|double_round|::mlir::BoolAttr   |Double round only occurs if shift is greater than 31|
|per_channel |::mlir::BoolAttr   |代码上看起来这个参数没用，完全由multi和shift的Size决定，Size必须和Input的inner-dim-size相同|

__代码逻辑:__

- 检查输入数据位宽，小于32位使用32bis，否者使用48bits
  - input_zp 转换成32bits或者48bits
  - output_zp 转换成32bits或者48bits
- 输入数据(value)位宽如果小于32bits转成32bits
- value = sub(value, input_zp）
- value = tosa::ApplyScaleOp(value, multi, shift)
- value = add(value, output_zp)
- 根据Output位宽计算出其范围(IntMin, IntMax)
- 饱和处理： value = clamp(value, IntMin, IntMax, SignedLessThan)
- 如果输出小于32bits，value = TruncateIOp(value)

__Sample:__ int8-to-int8

```Tosa
func @rescale_per_channel(%arg0 : tensor<2x3x4x2xi8>) -> (tensor<2x3x4x2xi8>) {
  %0 = "tosa.rescale"(%arg0) {
    input_zp = 243 : i32, output_zp = 252 : i32, 
    multiplier = [42 : i32, 43 : i32], 
    shift = [14 : i32, 33 : i32], 
    scale32 = true, double_round = true, per_channel = true
  } : (tensor<2x3x4x2xi8>)  -> (tensor<2x3x4x2xi8>)
  return %0 : tensor<2x3x4x2xi8>
}
```

为什么变换后仍旧还有 ___tosa.apply_scale___ ?????

```Linalg
#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3)>
builtin.module  {
  builtin.func @rescale_per_channel(%arg0: tensor<2x3x4x2xi8>) -> tensor<2x3x4x2xi8> {
    %cst = constant dense<[42, 43]> : tensor<2xi32>
    %cst_0 = constant dense<[14, 33]> : tensor<2xi8>
    %0 = linalg.init_tensor [2, 3, 4, 2] : tensor<2x3x4x2xi8>
    %1 = linalg.generic {indexing_maps = [#map0, #map1, #map1, #map0], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %cst, %cst_0 : tensor<2x3x4x2xi8>, tensor<2xi32>, tensor<2xi8>) outs(%0 : tensor<2x3x4x2xi8>) {
    ^bb0(%arg1: i8, %arg2: i32, %arg3: i8, %arg4: i8):  // no predecessors
      %c243_i32 = constant 243 : i32
      %c252_i32 = constant 252 : i32
      %2 = sexti %arg1 : i8 to i32
      %3 = subi %2, %c243_i32 : i32
      %4 = "tosa.apply_scale"(%3, %arg2, %arg3) {double_round = true} : (i32, i32, i8) -> i32 // ????
      %5 = addi %4, %c252_i32 : i32
      %c-128_i32 = constant -128 : i32
      %c127_i32 = constant 127 : i32
      %6 = cmpi slt, %5, %c-128_i32 : i32
      %7 = select %6, %c-128_i32, %5 : i32
      %8 = cmpi slt, %c127_i32, %5 : i32
      %9 = select %8, %c127_i32, %7 : i32
      %10 = trunci %9 : i32 to i8
      linalg.yield %10 : i8
    } -> tensor<2x3x4x2xi8>
    return %1 : tensor<2x3x4x2xi8>
  }
}
```

## 2.13 TosaToLinalg::ResizeConverter

__语义:__
tosa.resize()可以做 resize/upsample 缩放处理，支持 "NEAREST_NEIGHBOR" 和 "BILINEAR" 两种算法。

|Attribute|MLIR Type|Description|
|---------|---------|-----------|
|output_size|::mlir::ArrayAttr|64-bit integer array attribute with exactly 2 elements|
|stride|::mlir::ArrayAttr|64-bit integer array attribute with exactly 2 elements|
|offset|::mlir::ArrayAttr|64-bit integer array attribute with exactly 2 elements|
|shift|::mlir::IntegerAttr|32-bit signless integer attribute|
|stride_fp|::mlir::ArrayAttr|32-bit float array attribute with exactly 2 elements|
|offset_fp|::mlir::ArrayAttr|32-bit float array attribute with exactly 2 elements|
|mode|::mlir::StringAttr|Supported resize/upsampling strategies|

- shift == 0：使用 _stride_fp_和_offset_fp_，否则使用 _stride_和_offset_。
- mode：resize(NEAREST_NEIGHBOR) 和 upsample(BILINEAR)

__限制:__

- 输入是4D的Tensor，并且只能在HW维度上进行Resize。
- 固定输入Format： {Batch, H, W, C}
- 模式有："NEAREST_NEIGHBOR" 和 "BILINEAR"。

__逻辑:__

- 从InputShape获得 H 和 W
- shift == 0：使用 _stride_fp_和_offset_fp_，否则使用 _stride_ 和 _offset_
- ompute the the integer index and partial offset:
  - x = x * stride + offset;
  - ix = floor(x)
  - dx = x - ix
  - 同理计算 y/iy/dy
- mode == NEAREST_NEIGHBOR:
  - 选择最近邻的像素
  - 处理边界
- mode == BILINEAR:
  - 计算 y0x0/y0x1/y1x0/y1x1 四个点的坐标。
  - 根据中心到四个点的距离获得权重，计算出插值点的像素值。

__NEAREST_NEIGHBOR Sample:__

```Tosa
func @resize_nearest(%input: tensor<1x2x2x1xf32>) -> () {
  %output = "tosa.resize"(%input) { 
    output_size = [4, 4], stride = [0, 0], offset = [0, 0], 
    stride_fp = [0.5 : f32, 0.5 : f32], 
    offset_fp = [0.1 : f32, 0.2 : f32], 
    shift = 0 : i32, mode = "NEAREST_NEIGHBOR" 
  } : (tensor<1x2x2x1xf32>)  -> (tensor<1x4x4x1xf32>)
  return
}
```

```Linalg
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
builtin.module  {
  builtin.func @resize_nearest(%arg0: tensor<1x2x2x1xf32>) {
    %0 = linalg.init_tensor [1, 4, 4, 1] : tensor<1x4x4x1xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x4x4x1xf32>) {
    ^bb0(%arg1: f32):  // no predecessors
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 2 : index
      %5 = linalg.index 3 : index
      %c0_i32 = constant 0 : i32
      %c1_i32 = constant 1 : i32
      %c1_i32_0 = constant 1 : i32
      %6 = index_cast %3 : index to i32
      %7 = index_cast %4 : index to i32
      %cst = constant 5.000000e-01 : f32
      %cst_1 = constant 5.000000e-01 : f32
      %cst_2 = constant 1.000000e-01 : f32
      %cst_3 = constant 2.000000e-01 : f32
      %8 = uitofp %6 : i32 to f32
      %9 = uitofp %7 : i32 to f32
      %10 = mulf %8, %cst : f32
      %11 = mulf %9, %cst_1 : f32
      %12 = addf %10, %cst_2 : f32
      %13 = addf %11, %cst_3 : f32
      %14 = floorf %12 : f32
      %15 = floorf %13 : f32
      %16 = subf %12, %14 : f32
      %17 = subf %13, %15 : f32
      %18 = fptosi %14 : f32 to i32
      %19 = fptosi %15 : f32 to i32
      %cst_4 = constant 5.000000e-01 : f32
      %20 = cmpf oge, %16, %cst_4 : f32
      %21 = cmpf oge, %17, %cst_4 : f32
      %c0_i32_5 = constant 0 : i32
      %c1_i32_6 = constant 1 : i32
      %22 = select %20, %c1_i32_6, %c0_i32_5 : i32
      %23 = select %21, %c1_i32_6, %c0_i32_5 : i32
      %24 = addi %18, %22 : i32
      %25 = addi %19, %23 : i32
      %26 = cmpi slt, %24, %c0_i32 : i32
      %27 = select %26, %c0_i32, %24 : i32
      %28 = cmpi slt, %c1_i32, %24 : i32
      %29 = select %28, %c1_i32, %27 : i32
      %30 = cmpi slt, %25, %c0_i32 : i32
      %31 = select %30, %c0_i32, %25 : i32
      %32 = cmpi slt, %c1_i32_0, %25 : i32
      %33 = select %32, %c1_i32_0, %31 : i32
      %34 = index_cast %29 : i32 to index
      %35 = index_cast %33 : i32 to index
      %36 = tensor.extract %arg0[%2, %34, %35, %5] : tensor<1x2x2x1xf32>
      linalg.yield %36 : f32
    } -> tensor<1x4x4x1xf32>
    return
  }
}
```

__BILINEAR Sample:__

```Tosa
func @resize_nearest(%input: tensor<1x2x2x1xf32>) -> () {
  %output = "tosa.resize"(%input) { 
    output_size = [4, 4], stride = [0, 0], offset = [0, 0], 
    stride_fp = [0.5 : f32, 0.5 : f32], 
    offset_fp = [0.1 : f32, 0.2 : f32], 
    shift = 0 : i32, mode = "BILINEAR" 
  } : (tensor<1x2x2x1xf32>)  -> (tensor<1x4x4x1xf32>)

  return
}
```

```Linalg
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
builtin.module  {
  builtin.func @resize_nearest(%arg0: tensor<1x2x2x1xf32>) {
    %0 = linalg.init_tensor [1, 4, 4, 1] : tensor<1x4x4x1xf32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%0 : tensor<1x4x4x1xf32>) {
    ^bb0(%arg1: f32):  // no predecessors
      %2 = linalg.index 0 : index
      %3 = linalg.index 1 : index
      %4 = linalg.index 2 : index
      %5 = linalg.index 3 : index
      %c0_i32 = constant 0 : i32
      %c1_i32 = constant 1 : i32
      %c1_i32_0 = constant 1 : i32
      %6 = index_cast %3 : index to i32
      %7 = index_cast %4 : index to i32
      %cst = constant 5.000000e-01 : f32
      %cst_1 = constant 5.000000e-01 : f32
      %cst_2 = constant 1.000000e-01 : f32
      %cst_3 = constant 2.000000e-01 : f32
      %8 = uitofp %6 : i32 to f32
      %9 = uitofp %7 : i32 to f32
      %10 = mulf %8, %cst : f32
      %11 = mulf %9, %cst_1 : f32
      %12 = addf %10, %cst_2 : f32
      %13 = addf %11, %cst_3 : f32
      %14 = floorf %12 : f32
      %15 = floorf %13 : f32
      %16 = subf %12, %14 : f32
      %17 = subf %13, %15 : f32
      %18 = fptosi %14 : f32 to i32
      %19 = fptosi %15 : f32 to i32
      %c1_i32_4 = constant 1 : i32
      %20 = addi %18, %c1_i32_4 : i32
      %21 = addi %19, %c1_i32_4 : i32
      %22 = cmpi slt, %18, %c0_i32 : i32
      %23 = select %22, %c0_i32, %18 : i32
      %24 = cmpi slt, %c1_i32, %18 : i32
      %25 = select %24, %c1_i32, %23 : i32
      %26 = cmpi slt, %20, %c0_i32 : i32
      %27 = select %26, %c0_i32, %20 : i32
      %28 = cmpi slt, %c1_i32, %20 : i32
      %29 = select %28, %c1_i32, %27 : i32
      %30 = cmpi slt, %19, %c0_i32 : i32
      %31 = select %30, %c0_i32, %19 : i32
      %32 = cmpi slt, %c1_i32_0, %19 : i32
      %33 = select %32, %c1_i32_0, %31 : i32
      %34 = cmpi slt, %21, %c0_i32 : i32
      %35 = select %34, %c0_i32, %21 : i32
      %36 = cmpi slt, %c1_i32_0, %21 : i32
      %37 = select %36, %c1_i32_0, %35 : i32
      %38 = index_cast %25 : i32 to index
      %39 = index_cast %29 : i32 to index
      %40 = index_cast %33 : i32 to index
      %41 = index_cast %37 : i32 to index
      %42 = tensor.extract %arg0[%2, %38, %40, %5] : tensor<1x2x2x1xf32>
      %43 = tensor.extract %arg0[%2, %38, %41, %5] : tensor<1x2x2x1xf32>
      %44 = tensor.extract %arg0[%2, %39, %40, %5] : tensor<1x2x2x1xf32>
      %45 = tensor.extract %arg0[%2, %39, %41, %5] : tensor<1x2x2x1xf32>
      %cst_5 = constant 1.000000e+00 : f32
      %46 = subf %cst_5, %17 : f32
      %47 = mulf %42, %46 : f32
      %48 = mulf %43, %17 : f32
      %49 = addf %47, %48 : f32
      %50 = mulf %44, %46 : f32
      %51 = mulf %45, %17 : f32
      %52 = addf %50, %51 : f32
      %53 = subf %cst_5, %16 : f32
      %54 = mulf %49, %53 : f32
      %55 = mulf %52, %16 : f32
      %56 = addf %54, %55 : f32
      linalg.yield %56 : f32
    } -> tensor<1x4x4x1xf32>
    return
  }
}
```

## 2.14 TosaToLinalg::ReverseConverter

指定 axis 对Input tensor进行 reverse 处理， axis是整数，只能指定一个维度。

```Tosa
func @reverse(%arg0: tensor<5x4xi32>) -> () {
  %0 = "tosa.reverse"(%arg0) {axis = 0 : i64} : (tensor<5x4xi32>) -> tensor<5x4xi32>
  %1 = "tosa.reverse"(%arg0) {axis = 1 : i64} : (tensor<5x4xi32>) -> tensor<5x4xi32>
  return
}
```

这个比较简单，单纯地坐标变换即可：

```Linalg
#map0 = affine_map<(d0, d1) -> (-d0 + 4, d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, -d1 + 3)>
builtin.module  {
  builtin.func @reverse(%arg0: tensor<5x4xi32>) {
    %0 = linalg.init_tensor [5, 4] : tensor<5x4xi32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x4xi32>) outs(%0 : tensor<5x4xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
      linalg.yield %arg1 : i32
    } -> tensor<5x4xi32>
    %2 = linalg.init_tensor [5, 4] : tensor<5x4xi32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<5x4xi32>) outs(%2 : tensor<5x4xi32>) {
    ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
      linalg.yield %arg1 : i32
    } -> tensor<5x4xi32>
    return
  }
}
```

## 2.15 TosaToLinalg::TableConverter

插值查表操作。输入的值缩放至一个定点 9bits+7bits 的16bits值， 高位 9bits 作为索引用于查Table。接下来的7bits作为Weight，用于计算Table中Index和Index+1的插值。table的长度必须是513及其以上。

__代码逻辑:__

- _input(i8)_ 和 _table(i8)_ 以及 _output(i8)_ 的时候：
  - _input_ 的8bits就是Index
  - 不需要插值运算，直接使用Index查询 _table_
- _input(i16)_ 和 _table(i16)_ 以及 _output(i32)_ 的时候：
  - value = _input(i16)_ + 0x8000, 将i16的值挪到正数范围
  - index = value >> 7，取高9bits作为index
  - fraction = value & 0x3F, 取低7bits作为分数部分
  - 取 base=_table[index]_ 和 next=_table[index+1]_, _table(i16)_ 的长度大于等于513的原因在于此。
  - result = (base << 7) + (next - base) * fraction
- 目前只支持这两种情况。

__Sample:__

```Tosa
!type_input = type tensor<6xi16>
!type_table = type tensor<513xi16>
!type_output = type tensor<6xi32>
func @table(%arg0: !type_input, %arg1: !type_table) -> () {
  %0 = "tosa.table"(%arg0, %arg1)  : (!type_input, !type_table)  -> (!type_output)
  return
}
```

```Linalg
#map = affine_map<(d0) -> (d0)>
builtin.module  {
  builtin.func @table(%arg0: tensor<6xi16>, %arg1: tensor<513xi16>) {
    %0 = linalg.init_tensor [6] : tensor<6xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<6xi16>) outs(%0 : tensor<6xi32>) {
    ^bb0(%arg2: i16, %arg3: i32):  // no predecessors
      %2 = sexti %arg2 : i16 to i32
      %c32768_i32 = constant 32768 : i32
      %c7_i32 = constant 7 : i32
      %c1_i32 = constant 1 : i32
      %c127_i32 = constant 127 : i32
      %3 = addi %2, %c32768_i32 : i32
      %4 = shift_right_unsigned %3, %c7_i32 : i32
      %5 = and %3, %c127_i32 : i32
      %6 = addi %4, %c1_i32 : i32
      %7 = index_cast %4 : i32 to index
      %8 = index_cast %6 : i32 to index
      %9 = tensor.extract %arg1[%7] : tensor<513xi16>
      %10 = tensor.extract %arg1[%8] : tensor<513xi16>
      %11 = sexti %9 : i16 to i32
      %12 = sexti %10 : i16 to i32
      %13 = shift_left %11, %c7_i32 : i32
      %14 = subi %12, %11 : i32
      %15 = muli %14, %5 : i32
      %16 = addi %13, %15 : i32
      linalg.yield %16 : i32
    } -> tensor<6xi32>
    return
  }
}
```

## 2.16 TosaToLinalg::TileConverter

输入的Tensor沿着 _multiples{}_ 指定的值，在对应的维度进行复制。

__逻辑：__

- 输入Tensor的Rank扩展为原本的两倍，在各个维度前面插入 _multiples[ i ]_，构建输出Tensor
- 插入tosa.reshape 将输出的维度两两合并
- 用linalg.tensor_collapse_shape 替换 tosa.reshape

__例子：__

```Tosa
func @tile(%arg0 : tensor<2x3x4xi8>) -> () {
  %0 = "tosa.tile"(%arg0) {multiples = [3, 5, 7]} : (tensor<2x3x4xi8>)  -> (tensor<6x15x28xi8>)
  return
}
```

```Linalg
// SmallVector<AffineExpr, 4> dimExprs;
// dimExprs.reserve(rank);
// for (unsigned i = 0; i < rank; ++i)
//   dimExprs.push_back(rewriter.getAffineDimExpr(i * 2 + 1));
// auto readAffineMap =
//    AffineMap::get(/*dimCount=*/rank * 2, /*symbolCount=*/0, dimExprs,
//                    rewriter.getContext());
// SmallVector<AffineMap, 2> affineMaps = {
// // #map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>
//    readAffineMap,
// // #map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
//    rewriter.getMultiDimIdentityMap(genericShape.size())};

#map0 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
builtin.module  {
  builtin.func @tile(%arg0: tensor<2x3x4xi8>) {
    %0 = linalg.init_tensor [3, 2, 5, 3, 7, 4] : tensor<3x2x5x3x7x4xi8>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x3x4xi8>) outs(%0 : tensor<3x2x5x3x7x4xi8>) {
    ^bb0(%arg1: i8, %arg2: i8):  // no predecessors
      linalg.yield %arg1 : i8
    } -> tensor<3x2x5x3x7x4xi8>
    %2 = linalg.tensor_collapse_shape %1 [[0, 1], [2, 3], [4, 5]] : tensor<3x2x5x3x7x4xi8> into tensor<6x15x28xi8>
    return
  }
}
```

## 2.17 TosaToLinalg::TransposeConverter

- 语义：_Param[1, 2, 0]是指_:
  - output_dim0=input_dim1
  - output_dim1=input_dim2
  - output_dim2=input_dim0

- Tosa
  
  ```TOSA
  func @test_transpose(%arg0: tensor<1x2x3xi32>) -> () {
    %0 = constant dense<[1, 2, 0]> : tensor<3xi32>
    %1 = "tosa.transpose"(%arg0, %0) : (tensor<1x2x3xi32>, tensor<3xi32>) -> (tensor<2x3x1xi32>)
    return
  }
  ```

- Tosa-to-Linalg:
  - map0: input在空间(d0, d1, d2)按照(d2, d0, d1)顺序读取数据
  - map1: output在空间(d0, d1, d2)按照(d0, d1, d2)顺序写入数据

  ```Linalg
  #map0 = affine_map<(d0, d1, d2) -> (d2, d0, d1)>
  #map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  builtin.module  {
    builtin.func @test_transpose(%arg0: tensor<1x2x3xi32>) {
      %cst = constant dense<[1, 2, 0]> : tensor<3xi32>
      %0 = linalg.init_tensor [2, 3, 1] : tensor<2x3x1xi32>
      %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x2x3xi32>) outs(%0 : tensor<2x3x1xi32>) {
      ^bb0(%arg1: i32, %arg2: i32):  // no predecessors
        linalg.yield %arg1 : i32
      } -> tensor<2x3x1xi32>
      return
    }
  }
  ```

## 2.18 TosaToLinalg::MatMulConverter

Tosa.matmul在语义上和Xla.DotGeneral相同，接口描述上是可以带有Bias的，__但实际上还没有支持__。目前会转成：

- _linalg.batch_matmul_
- _linalg.quantized_batch_matmul_

```Tosa
func @matmul(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x6xf32>, %arg2: tensor<1x6xf32>) -> (tensor<1x5x6xf32>) {
  %0 = "tosa.matmul"(%arg0, %arg1) : (tensor<1x5x3xf32>, tensor<1x3x6xf32>)  -> (tensor<1x5x6xf32>)
  return %0 : tensor<1x5x6xf32>
}
```

```Linalg
builtin.module  {
  builtin.func @matmul(%arg0: tensor<1x5x3xf32>, %arg1: tensor<1x3x6xf32>, %arg2: tensor<1x6xf32>) -> tensor<1x5x6xf32> {
    %cst = constant 0.000000e+00 : f32
    %0 = linalg.init_tensor [1, 5, 6] : tensor<1x5x6xf32>
    %1 = linalg.fill(%cst, %0) : f32, tensor<1x5x6xf32> -> tensor<1x5x6xf32> 
    %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x5x3xf32>, tensor<1x3x6xf32>) outs(%1 : tensor<1x5x6xf32>) -> tensor<1x5x6xf32>
    return %2 : tensor<1x5x6xf32>
  }
}
```

## 2.19 TosaToLinalg::FullyConnectedConverter

2D的矩阵乘计算，Bias是必须项，所以看起来这是网络中的全连接层的计算描述。
|Operand|Description|
|-------|--------------------------------------------------------------|
|input  |unranked.tensor of number values or 2D tensor of number values|
|weight |unranked.tensor of number values or 2D tensor of number values|
|bias   |unranked.tensor of number values or 1D tensor of number values|

整数输入时支持量化：

- _linalg.quantized_matmul_
- _linalg.matmul_

```Tosa
func @fully_connected(%arg0: tensor<5x3xf32>, %arg1: tensor<6x3xf32>, %arg2: tensor<6xf32>) -> (tensor<5x6xf32>) {
  %0 = "tosa.fully_connected"(%arg0, %arg1, %arg2) : (tensor<5x3xf32>, tensor<6x3xf32>, tensor<6xf32>)  -> (tensor<5x6xf32>)
  return %0 : tensor<5x6xf32>
}
```

```Linalg
#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
builtin.module  {
  builtin.func @fully_connected(%arg0: tensor<5x3xf32>, %arg1: tensor<6x3xf32>, %arg2: tensor<6xf32>) -> tensor<5x6xf32> {
    %0 = linalg.init_tensor [5, 6] : tensor<5x6xf32>
    %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg2 : tensor<6xf32>) outs(%0 : tensor<5x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      linalg.yield %arg3 : f32
    } -> tensor<5x6xf32>
    %cst = constant dense<[1, 0]> : tensor<2xi64>
    %2 = linalg.init_tensor [3, 6] : tensor<3x6xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1 : tensor<6x3xf32>) outs(%2 : tensor<3x6xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):  // no predecessors
      linalg.yield %arg3 : f32
    } -> tensor<3x6xf32>
    %4 = linalg.matmul ins(%arg0, %3 : tensor<5x3xf32>, tensor<3x6xf32>) outs(%1 : tensor<5x6xf32>) -> tensor<5x6xf32>
    return %4 : tensor<5x6xf32>
  }
}
```

目前的处理方式是Bias会在Output的初始化时赋值，但这样的性能不会太好，为什么要这样实现？

## 2.20 TosaToLinalg::Pool2dConverter\<SrcOp>

支持两种 pool 算子的转换：

- tosa.avg_pool2d
- tosa.max_pool2d

|type|Attribute|MLIR Type|Description|
|----|---------|---------|-----------|
|avg, max|kernel|mlir::ArrayAttr|64-bit integer array attribute with exactly 2 elements|
|avg, max|stride|mlir::ArrayAttr|64-bit integer array attribute with exactly 2 elements|
|avg, max|pad|mlir::ArrayAttr|64-bit integer array attribute with exactly 4 elements|
|avg|quantization_info|mlir::tosa::UnaryOpQuantizationAttr|Attribute for UnaryOp quantization information.|

__代码逻辑：__

- 创建Fake tensor[kernel.y, kernel.x]
- 创建Daliation{1, 1}
- 有 _Pad>0_ 的时候需要插入 _linalg.pad_tensor_
- 有 _Stride_ 的时候计算输出为：
- ```output_size = ((input_size + pad_low + pad_high) - kernel_size + 1) + (stride_size - 1) /stride_size```
- 如果是MaxPool：
  - 创建初始化变量：取对应有符号数据类型的最小值。
  - _linalg.pooling_nhwc_max_(kernel): { attr.strides, attr.dilation }
  - 默认是 ___NHWC___ 的格式
- 如果是AvgPool：
  - 创建初始化变量：Zero。
  - linalg.pooling_nhwc_sum(kernel): { attr.strides, attr.dilation }
  - 默认是 ___NHWC___ 的格式，注意Linalg只能计算pool-sum，还需要计算均值。
  - 根据 _Pad_ 的值以及 _Kernel_ 计算每一个Output元素对应于输入的位置上 _Kernel_ 窗口内实际有效的元素个数，去掉PadZero对均值的影响。

__Sample：__

```Tosa
func @avg_pool(%arg0: tensor<1x6x34x62xf32>) -> (tensor<1x3x11x62xf32>) {
  %0 = "tosa.avg_pool2d"(%arg0) {
    pad = [1, 1, 1, 1], kernel = [4, 4], stride = [2, 3]
  } : (tensor<1x6x34x62xf32>)  -> (tensor<1x3x11x62xf32>) 
  return %0 : tensor<1x3x11x62xf32>
}
```

```Linalg
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
builtin.module  {
  builtin.func @avg_pool(%arg0: tensor<1x6x34x62xf32>) -> tensor<1x3x11x62xf32> {
    %cst = constant 0.000000e+00 : f32
    %0 = linalg.pad_tensor %arg0 low[0, 1, 1, 0] high[0, 1, 1, 0]  {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):  // no predecessors
      linalg.yield %cst : f32
    } : tensor<1x6x34x62xf32> to tensor<1x8x36x62xf32> 
    %cst_0 = constant 0.000000e+00 : f32
    %1 = linalg.init_tensor [1, 3, 11, 62] : tensor<1x3x11x62xf32>
    %2 = linalg.fill(%cst_0, %1) : f32, tensor<1x3x11x62xf32> -> tensor<1x3x11x62xf32> 
    %3 = linalg.init_tensor [4, 4] : tensor<4x4xf32>
    %4 = linalg.pooling_nhwc_sum {dilations = dense<1> : vector<2xi64>, strides = dense<[2, 3]> : vector<2xi64>} ins(%0, %3 : tensor<1x8x36x62xf32>, tensor<4x4xf32>) outs(%2 : tensor<1x3x11x62xf32>) -> tensor<1x3x11x62xf32>
    %5 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%4 : tensor<1x3x11x62xf32>) {
    ^bb0(%arg1: f32):  // no predecessors
      %c0 = constant 0 : index
      %c1 = constant 1 : index
      %c2 = constant 2 : index
      %c10 = constant 10 : index
      %6 = linalg.index 1 : index
      %7 = linalg.index 2 : index
      %8 = subi %c2, %6 : index
      %9 = subi %c10, %7 : index
      %c4 = constant 4 : index
      %c1_1 = constant 1 : index
      %10 = subi %6, %c1_1 : index
      %11 = cmpi slt, %10, %c0 : index
      %12 = select %11, %10, %c0 : index
      %13 = addi %c4, %12 : index
      %c1_2 = constant 1 : index
      %14 = subi %8, %c1_2 : index
      %15 = cmpi slt, %14, %c0 : index
      %16 = select %15, %14, %c0 : index
      %17 = addi %13, %16 : index
      %18 = cmpi slt, %17, %c1 : index
      %19 = select %18, %c1, %17 : index
      %c4_3 = constant 4 : index
      %c1_4 = constant 1 : index
      %20 = subi %7, %c1_4 : index
      %21 = cmpi slt, %20, %c0 : index
      %22 = select %21, %20, %c0 : index
      %23 = addi %c4_3, %22 : index
      %c1_5 = constant 1 : index
      %24 = subi %9, %c1_5 : index
      %25 = cmpi slt, %24, %c0 : index
      %26 = select %25, %24, %c0 : index
      %27 = addi %23, %26 : index
      %28 = cmpi slt, %27, %c1 : index
      %29 = select %28, %c1, %27 : index
      %30 = muli %19, %29 : index
      %31 = index_cast %30 : index to i32
      %32 = sitofp %31 : i32 to f32
      %33 = divf %arg1, %32 : f32
      linalg.yield %33 : f32
    } -> tensor<1x3x11x62xf32>
    return %5 : tensor<1x3x11x62xf32>
  }
}
```
