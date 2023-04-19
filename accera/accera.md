# Accera

[Accera](https://microsoft.github.io/Accera/) 采用定义循环计划和算法的 Python 代码，同时将其转换为MLIR中间表示 (IR)。Accera 的编译器然后通过一系列 MLIR 管道获取此 IR 以执行转换。结果是一个带有 C 头文件的二进制库。该库实现了在 Python 中定义的算法，并且与目标兼容。

## 1. accera/acc-gpu-runner

略

## 2. accera/accc

一个端到端的编译工具链，这个可以参考一下。

* 输入：
  * Accera DSL C++ file
  * csv file of domain sizes
  * Library name
  * Runner main file
* 行为：
  * Create a generator for the Accera DSL
  * Run the generator and lower the result, emitting an object file and header for the current host machine
  * Create a runner project for the runner main file
  * Build and run the runner program

## 3. accera/ir

* 生成 _libMLIRAcceraIntrinsics.a_: 他是一个基于MLIR的 accera dialect
* 生成 _libir.a_: 包括是本目录的全部功能，定义了 accera ir

### 3.1 accera/ir/accera

FIXME: 看起来这个 Accera 是应用层类的一些计算描述，类似 `tosa` ?

* Accera table-gen 文件
  * AcceraOps.td: 定义了 `accera` dialect，以及  `accera.gemm` operation
* Accera cpp 文件：
  * AcceraOps.cpp：初始化 `accera` dialect， 登录 `accera.gemm` operation

### 3.2 accera/ir/argo

FIXME: 和linagl对应的一层吗？

* Argo table-gen 文件：
  * ArgoBase.td:
    * 定义 `argo` dialect
  * ArgoOps.td:
    * 定义 `argo.yield` operation: 是 `argo` 中区域内块的不透明操作的特殊终止符操作， 它将值返回到即将结束的 `argo` 不透明操作。
    * 定义 `argo.entry_point` operation: 是一个 _function-declaration-like operation_，为 kernel 函数声明一个 host 端入口函数，起到了在 _kernel_ 和 _entry_ 之间的连接加入了一层抽象，方便更加灵活的支持各种后端。
  * ArgoStructuredOps.td
    * 定义 `argo.copy` structed op: 拷贝数据从input view 到 output view。
    * 定义 `argo.fill` structed op: 填充Output view。
    * 定义 `argo.matmul` structed op: build-in instruction。
    * 定义 `argo.acc` structed op: build-in instruction。
  * ArgoStructuredOpsInterface.td
    * 定义 _ArgoStructuredInterface_ : 提供访问 _ArgoOp_ 的接口。
* Argo cpp 文件：
  * ArgoOps.cpp:
    * 实现 ArgoOp 的 _print_, _parse_, _verify_ 等基础操作
    * 定义 ArgoOp 的 _CanonicalizationPatterns pass_，支持了 _EraseDeadArgoOp_
  * ArgoTypes.cpp:
    * 重载 _inline_ interface，定制 _inline_ 规则
    * 初始化 dialect `argo`，添加两个  `yield`和`entry_point` op，添加 一个 _inline_ interface。
  * Utils.cpp:
    * 针对 ArgoOp 的一些常用操作。

### 3.3 accera/ir/exec

* Execution table-gen 文件
  * ExecutionPlanOps.td:
    * 定义 `accxp` dialect
    * 定义 _VectorizationInfoAttr_ dialect attribute
    * 定义 _ParallelizationInfoAttr_ dialect attribute
    * 定义 _TensorizationInfoAttr_ dialect attribute
    * 定义 _InPlaceUnrollInfoAttr_ dialect attribute
    * 定义 `accxp.terminator` op
    * 定义 `accxp.make_cache` op: 推理出 cache shape 以及 viewing map.
    * 定义 `accxp.active_element_cache_copy` op: cache reshape and copy.
    * 定义 `accxp.active_block_cache_copy` op: cache reshape and block copy.
    * 定义 `accxp.multi_cache_copy` op: multi cache reshape and copy.
    * 定义 `accxp.cacheFillOp` op: fill cache with constant value.
    * 定义 `accxp.active_element_cache_reduce` op: cache write back to output
    * 定义 `active_block_cache_reduce` op: cache block write back to output
    * 定义 `accxp.begin_cache_mapping` op: 映射的开始位置，使用  _cache value_ 替换 _input value_，并且根据需要替换相应的 _operation_，需要在同一个 _block_ 内和 `accxp.end_cache_mapping` 成对出现。
    * 定义 `accxp.end_cache_mapping` op: 表示cache mapping结束的位置。
    * 定义 `accxp.begin_create_cache` op: 标记 cache active 的起始位置。
    * 定义 `accxp.begin_active_cache_region` op: This is different from `accxp.begin_create_cache` in that it does not compute a cache based on the contents of the region, but simply denotes where a previously-computed cache is active. ___FIXME: 要看看代码确认一下逻辑___
    * 定义 `accxp.begin_create_max_element_cache` op: 他会lower成 `accxp.begin_create_cache` op，表示 a max element cache is active.
    * 定义 `accxp.end_cache_region` op: 需要和上面三个begin成对出现，否则会报错。
    * 定义 `accxp.delayed_mapping_region_op` op:  operation will map one value to another for all the ops in its region. This op exists as a way to replace one value with another in ops that haven't been fully expanded to consume the "from" SSA value yet, as is sometimes the case with accesses into cache memrefs
  * ExecutionPlanAttrs.td:
    * 定义 _CacheMappingAttr_: 很奇怪的定义: Global/Physical/Logical
    * 定义 _CacheAllocationAttr_: automatic or none
    * 定义 _CacheCopyDimensionsAttr_：src or dst，指定 cache copy side.
  * ExecutionPlanInterfaces.td:
    * 定义 _BeginCacheRegion_ Interface
    * 定义 _EndCacheRegion_ Interface
* Execution cpp 文件
  * ExecutionPlanOps.cpp
  * ExecutionPlanAttributes.cpp
  * CacheAccessMaps.cpp

### 3.4 accera/ir/intrinsics

### 3.5 accera/ir/nest

### 3.6 accera/ir/value
