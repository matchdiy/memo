# Chapter 3: High-level Language-Specific Analysis and Transformation

创建一种精确代表输入语言语义的方言，可以在 MLIR 中进行分析、转换和优化，这些分析、转换和优化需要高级语言信息，并且通常在语言 AST 上执行。例如，clang 有一个相当繁重的机制来在 C++ 中执行模板实例化。

我们将编译器转换分为两类：_局部_ 和 _全局_。 在本章中，我们将重点介绍如何利用 Toy 方言及其高级语义来执行在 LLVM 中很难进行的局部模式匹配转换（local pattern-match transformations）。为此，我们使用 MLIR 的 [Generic DAG Rewriter](https://mlir.llvm.org/docs/PatternRewriter/)。

有两种方法可用于实现模式匹配转换：

* ___命令式：___ C++模式匹配和重写
* ___声明式：___ 基于规则的模式匹配，和使用table-driven重写的 Declarative Rewrite Rules:[DRR](https://mlir.llvm.org/docs/DeclarativeRewrites/)。 请注意，使用 DRR 要求使用 ODS 定义 operation，如第 2 章所述。

## 1. Optimize Transpose using C++ style pattern-match and rewrite

让我们从一个简单的模式开始，尝试消除两个相互抵消的转置：transpose(transpose(X)) -> X：

Toy Example

  ```Toy
  def transpose_transpose(x) {
    return transpose(transpose(x));
  }
  ```

IR Express

  ```IR
  toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
    toy.return %1 : tensor<*xf64>
  }
  ```

这是一个很好的转换示例，在 Toy IR 上匹配起来很简单，但 LLVM 做起来却很难。今天的 clang 无法优化掉临时数组，这个计算将被表达为这样的循环：

  ```C++
  #define N 100
  #define M 100
  void sink(void *);
  void double_transpose(int A[N][M]) {
    int B[M][N];
    for(int i = 0; i < N; ++i) {
      for(int j = 0; j < M; ++j) {
        B[j][i] = A[i][j];
      }
    }
    for(int i = 0; i < N; ++i) {
      for(int j = 0; j < M; ++j) {
        A[i][j] = B[j][i];
      }
    }
    sink(A);
  }
  ```

对于一种简单的 C++ 重写方法，涉及在 IR 中匹配树状模式并将其替换为一组不同的操作，我们可以通过实现 RewritePattern 插入 MLIR [canonicalizer pass](https://mlir.llvm.org/docs/Canonicalization/)：

  ```C++
  /// Fold transpose(transpose(x)) -> x
  struct SimplifyRedundantTranspose : public mlir::OpRewritePattern<TransposeOp> {
    /// We register this pattern to match every toy.transpose in the IR.
    /// The "benefit" is used by the framework to order the patterns and process
    /// them in order of profitability.
    SimplifyRedundantTranspose(mlir::MLIRContext *context)
        : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}

    /// This method is attempting to match a pattern and rewrite it. The rewriter
    /// argument is the orchestrator of the sequence of rewrites. It is expected
    /// to interact with it to perform any changes to the IR from here.
    mlir::LogicalResult
    matchAndRewrite(TransposeOp op,
                    mlir::PatternRewriter &rewriter) const override {
      // Look through the input of the current transpose.
      mlir::Value transposeInput = op.getOperand();
      TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

      // Input defined by another transpose? If not, no match.
      if (!transposeInputOp)
        return failure();

      // Otherwise, we have a redundant transpose. Use the rewriter.
      rewriter.replaceOp(op, {transposeInputOp.getOperand()});
      return success();
    }
  };
  ```

这个 rewriter 主要定义在 ToyCombine.cpp 文件中，_canonicalization pass_ 以贪婪、迭代的方式实现了变换。然后将这个pass 注册到 _TransposeOp::getCanonicalizationPatterns()_ 中：

  ```C++
  // Register our patterns for rewrite by the Canonicalization framework.
  void TransposeOp::getCanonicalizationPatterns(
      RewritePatternSet &results, MLIRContext *context) {
    results.add<SimplifyRedundantTranspose>(context);
  }
  ```

接下来还需要添加一个用于优化的pipeline，MLIR以类似于LLVM的方式，使用 _PassManager_ 来管理这些优化：

  ```C++
  mlir::PassManager pm(module->getName());
  pm.addNestedPass<mlir::toy::FuncOp>(mlir::createCanonicalizerPass());
  ```

进行一下测试```toyc-ch3 test/Examples/Toy/Ch3/transpose_transpose.toy -emit=mlir -opt```，可以看到如下结果：

```MLIR
toy.func @transpose_transpose(%arg0: tensor<*xf64>) -> tensor<*xf64> {
  %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
  toy.return %arg0 : tensor<*xf64>
}
```

结果不出所料，直接返回了 arg0 作为结果。然而结果并不完美，还残留了一个冗余的 transpose 没有被消除。我们的模式用函数输入替换了最后一个变换，并留下了现在的 dead 转置输入。Canonicalizer 知道删掉死代码，然而MLIR保守的认为这个操作会有 _side-effects_。我们通过增加一个特征 ___Pure___ 到 _TransposeOp_ 来解决这个问题：

  ```MLIR
  def TransposeOp : Toy_Op<"transpose", [Pure]> {...}
  ```

再次测试会发现这回的结果就是我们期望的！

## 2. Optimize Reshapes using DRR

声明式、基于规则的模式匹配和重写 (_DRR_) 是一种基于 DAG 的声明式重写器操作，它为模式匹配和重写规则提供基于表的语法。
  
  ```TB
  class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    dag benefitsAdded = (addBenefit 0)>;
  ```

类似于 SimplifyRedundantTranspose 的冗余 _ReshapeOp_ 优化可以使用 _DRR_ 更简单地表示如下。连续进行多个 _ReshapeOp_ 的时候，只保留最后一个即可：

  ```TB
  // Reshape(Reshape(x)) = Reshape(x)
  def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)), (ReshapeOp $arg)>;
  ```

可以在 path/to/BUILD/tools/mlir/examples/toy/Ch3/ToyCombine.inc 下找到对应于每个 _DRR_ 模式的自动生成的 C++ 代码。当转换以参数和结果的某些属性为条件时，DRR 还提供了一种添加参数约束的方法，一个例子是在冗余时消除 _reshape_ 的转换时，设置输入和输出形状相同时。下面的例子是在 _ReshapeOp_ 的输入和输出 _Type_ 相同的时候消除掉 _ReshapeOp_ 的优化：

  ```TB
  def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
  def RedundantReshapeOptPattern : Pat<(ReshapeOp:$res $arg), (replaceWithValue $arg),
        [(TypesAreIdentical $res, $arg)]>;
  ```

一些优化可能需要对指令参数进行额外的转换。这是使用 ___NativeCodeCall___ 实现的，它允许通过调用 C++ 辅助函数或使用内联 C++ 进行更复杂的转换。这种优化的一个例子是 FoldConstantReshape，我们直接通过 inline 生成一个指定 _Type_ 的 _ConstantOp_ 操作来优化常量值的 Reshape， 使其在运行期不发生。

  ```TB
  def ReshapeConstant : NativeCodeCall<"$0.reshape(($1.getType()).cast<ShapedType>())">;
  def FoldConstantReshapeOptPattern : Pat<
    (ReshapeOp:$res (ConstantOp $arg)),
    (ConstantOp (ReshapeConstant $arg, $res))>;
  ```

我们使用以下 trivial_reshape.toy 程序演示这些 reshape 优化：

```Toy
def main() {
  var a<2,1> = [1, 2];
  var b<2,1> = a;
  var c<2,1> = b;
  print(c);
}
```

```MLIR
module {
  toy.func @main() {
    %0 = toy.constant dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf64>
    %1 = toy.reshape(%0 : tensor<2xf64>) to tensor<2x1xf64>
    %2 = toy.reshape(%1 : tensor<2x1xf64>) to tensor<2x1xf64>
    %3 = toy.reshape(%2 : tensor<2x1xf64>) to tensor<2x1xf64>
    toy.print %3 : tensor<2x1xf64>
    toy.return
  }
}
```

运行```toyc-ch3 test/Examples/Toy/Ch3/trivial_reshape.toy -emit=mlir -opt```

```MLIR
module {
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf64>
    toy.print %0 : tensor<2x1xf64>
    toy.return
  }
}
```

完美！
