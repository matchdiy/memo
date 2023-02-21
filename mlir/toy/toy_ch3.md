# Chapter 3: High-level Language-Specific Analysis and Transformation

创建一种精确代表输入语言语义的方言，可以在 MLIR 中进行分析、转换和优化，这些分析、转换和优化需要高级语言信息，并且通常在语言 AST 上执行。例如，clang 有一个相当繁重的机制来在 C 中执行模板实例化。

我们将编译器转换分为两类：局部和全局。 在本章中，我们将重点介绍如何利用 Toy 方言及其高级语义来执行在 LLVM 中很难进行的本地模式匹配转换。为此，我们使用 MLIR 的通用 DAG 重写器。

有两种方法可用于实现模式匹配转换：

- ___命令式：___ C 模式匹配和重写
- ___声明式：___ 基于规则的模式匹配和使用表驱动的声明式重写规则 (Declarative Rewrite Rules:[DRR](https://mlir.llvm.org/docs/DeclarativeRewrites/)) 重写。 请注意，使用 DRR 要求使用 ODS 定义操作，如第 2 章所述。
