# Chapter 4: Enabling Generic Transformation with Interfaces

## 1. Background: Grappling with an Extensible IR

通过方言，MLIR 允许表示许多不同的抽象层次，我们之前定义的 Toy 方言就是这样一个例子。 尽管这些不同的方言可能代表不同的抽象，但通常有一组我们想要执行的通用转换和分析。 如果为每一种方言都实现各自的转换会导致大量代码重复，因为内部算法通常非常相似。 我们希望提供转换功能，以不透明的方式挂钩到像 Toy 这样的方言中，以获取他们需要的信息。

MLIR 为某些核心转换提供了一组始终可用的挂钩，如前一章所示，我们通过操作挂钩 (_getCanonicalizationPatterns_) 注册了一些规范化。 然而，这些类型的钩子并不能很好地扩展。 因此，以 [Interface](https://mlir.llvm.org/docs/Interfaces/) 的形式设计了一个更通用的解决方案，使 MLIR 基础设施像表示（representation）一样可扩展。 接口为方言和操作提供了一种通用机制，以便为转换或分析提供信息。

## 2. Shape Inference: Preparing for Code Generation

我们的 Toy IR 目前在通用张量上运行，这意味着我们不知道张量的形状除非使用常量初始化这些张量。 这使优化和代码生成变得复杂。幸运的是，我们可以通过计算简单地传播形状，直到它们都已知为止。 问题是如何处理对用户定义的通用函数的调用：每个调用点都可以推断出不同的形状。__一种__ 可能性是根据参数类型执行符号推理，但如果我们要在语言中引入更多控制流，这将很难通用。__另一种__ 方法是函数特化，其中每个具有新参数形状的调用点都复制被调用函数并对其进行特化。我们对 Toy 采取的方法是内联所有函数调用，然后执行过程内形状传播。

### 2.1 Inlining

在这里，我们可以编写专门为 Toy 方言设计的内联算法，但这可能会变得相当复杂，具体取决于我们想要的复杂程度。 撇开成本建模不谈，从头开始实施纯结构转型已经很复杂了。值得庆幸的是，MLIR 提供通用内联算法可以插入到方言中。在 Toy 中，我们需要做的就是为内联器提供 [Interface](https://mlir.llvm.org/docs/Interfaces/) 。

我们需要做的第一件事是在 Toy 方言中定义对内联操作的约束。 此信息通过方言界面提供。这本质上是一个包含一组虚拟挂钩的类，方言可以override这些virtual hooks。在这种情况下，接口是 DialectInlinerInterface。

  ```C++
  /// This class defines the interface for handling inlining with Toy operations.
  /// We simplify inherit from the base interface class and override
  /// the necessary methods.
  struct ToyInlinerInterface : public DialectInlinerInterface {
    using DialectInlinerInterface::DialectInlinerInterface;

    /// This hook checks to see if the given callable operation is legal to inline
    /// into the given call. For Toy this hook can simply return true, as the Toy
    /// Call operation is always inlinable.
    bool isLegalToInline(Operation *call, Operation *callable,
                        bool wouldBeCloned) const final {
      return true;
    }

    /// This hook checks to see if the given operation is legal to inline into the
    /// given region. For Toy this hook can simply return true, as all Toy
    /// operations are inlinable.
    bool isLegalToInline(Operation *, Region *, bool,
                        IRMapping &) const final {
      return true;
    }

    /// This hook cheks if the given 'src' region can be inlined into the 'dest'
    /// region. The regions here are the bodies of the callable functions. For
    /// Toy, any function can be inlined, so we simply return true.
    bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                        IRMapping &valueMapping) const final {
      return true;
    }

    /// This hook is called when a terminator operation has been inlined. The only
    /// terminator that we have in the Toy dialect is the return
    /// operation(toy.return). We handle the return by replacing the values
    /// previously returned by the call operation with the operands of the
    /// return.
    void handleTerminator(Operation *op,
                          ArrayRef<Value> valuesToRepl) const final {
      // Only "toy.return" needs to be handled here.
      auto returnOp = cast<ReturnOp>(op);

      // Replace the values directly with the return operands.
      assert(returnOp.getNumOperands() == valuesToRepl.size());
      for (const auto &it : llvm::enumerate(returnOp.getOperands()))
        valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  };
  ```

我们需要重载 _isLegalToInline_ 提供在以上各种情况下 Toy Dialect是否支持 _inline_ 的信息。此外，内联器只会丢弃私有可见的未使用函数定义。我们还必须在 MLIR 生成器中设置函数（主函数除外）的可见性。

  ```C++
  /// Emit a new function and add it to the MLIR module.
  mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
    ...
    // If this function isn't main, then set the visibility to private.
    if (funcAST.getProto()->getName() != "main")
      function.setPrivate();

    return function;
  }
  ```

然后我们直接在 Toy Dialect 上注册我们的方言接口，和 _Operation_ 的注册时相似的。

  ```C++
  void ToyDialect::initialize() {
    addInterfaces<ToyInlinerInterface>();
  }
  ```

接下来，我们需要提供一种方法让内联器知道 toy.generic_call 代表一个调用（_call_），而 toy.func 代表一个函数（_function_）。MLIR 提供操作接口，可用于将操作标记为 ___call-like___ 或 ___callable-like___。与方言界面不同，操作界面提供更精细的信息粒度，这些信息是特定于单个操作的核心。 我们将在此处添加的接口是 _CallOpInterface_ 和 _CallableOpInterface_。要添加此接口，我们只需将定义包含到我们的操作规范文件 (Ops.td) 中：

  ```C++
  #include "mlir/Interfaces/CallInterfaces.td"
  ```

并将其添加到 ___GenericCallOp___ 的特征列表中：

  ```MLIR
  def FuncOp : Toy_Op<"func",
      [DeclareOpInterfaceMethods<CallableOpInterface>]> {
    ...
  }

  def GenericCallOp : Toy_Op<"generic_call",
      [DeclareOpInterfaceMethods<CallOpInterface>]> {
    ...
  }
  ```

在上面我们还使用 ___DeclareOpInterfaceMethods___ 指令自动声明 ___GenericCallOp___ 类声明中的所有接口方法。这意味着我们只需要提供一个定义：

  ```C++
  /// Returns the region on the function operation that is callable.
  Region *FuncOp::getCallableRegion() { return &getBody(); }

  /// Returns the results types that the callable region produces when
  /// executed.
  ArrayRef<Type> FuncOp::getCallableResults() { return getType().getResults(); }

  // ....

  /// Return the callee of the generic call operation, this is required by the
  /// call interface.
  CallInterfaceCallable GenericCallOp::getCallableForCallee() {
    return getAttrOfType<SymbolRefAttr>("callee");
  }

  /// Get the argument operands to the called function, this is required by the
  /// call interface.
  Operation::operand_range GenericCallOp::getArgOperands() { return inputs(); }
  ```

现在内联器已获知 Toy Dialect的信息，我们可以将 _inliner pass_ 添加到玩具的通行证管理器中：

  ```C++
  pm.addPass(mlir::createInlinerPass());
  ```

现在让我们看一个工作示例：

  ```MLIR
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
    %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
    %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
    %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
  ```

我们有两个对 multiply_transpose 的调用，我们希望将其内联到 main 中，但如果我们查看输出，则没有任何变化。 我们遗漏了最后一个微妙的部分：在调用的边缘有一个隐藏的类型转换。如果我们看上面的内容，generic_call 的操作数是 tensor<2x3xf64> 类型，而函数的输入期望是 tensor<*xf64>。为了解决这个差异，内联器需要插入一个显式转换操作。 为此，我们需要向 Toy Dialect 添加一个新操作 _ToyCastOp(toy.cast)_，以表示两个不同形状之间的转换。

  ```TB
  def CastOp : Toy_Op<"cast", [
      DeclareOpInterfaceMethods<CastOpInterface>,
      Pure,
      SameOperandsAndResultShape]
    > {
    let summary = "shape cast operation";
    let description = [{
      The "cast" operation converts a tensor from one type to an equivalent type
      without changing any data elements. The source and destination types
      must both be tensor types with the same element type. If both are ranked,
      then shape is required to match. The operation is invalid if converting
      to a mismatching constant dimension.
    }];

    let arguments = (ins F64Tensor:$input);
    let results = (outs F64Tensor:$output);
    let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  }
  ```

请注意，此转换操作的定义将 ___CastOpInterface___ 添加到特征列表中。 该接口为类转换操作提供了几个实用程序，例如折叠身份转换和验证。 我们通过为 _areCastCompatible_ 方法提供定义来连接到这个接口：

  ```C++
  /// Returns true if the given set of input and result types are compatible with
  /// this cast operation. This is required by the `CastOpInterface` to verify
  /// this operation and provide other additional utilities.
  bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
    if (inputs.size() != 1 || outputs.size() != 1)
      return false;
    // The inputs must be Tensors with the same element type.
    TensorType input = inputs.front().dyn_cast<TensorType>();
    TensorType output = outputs.front().dyn_cast<TensorType>();
    if (!input || !output || input.getElementType() != output.getElementType())
      return false;
    // The shape is required to match if both types are ranked.
    return !input.hasRank() || !output.hasRank() || input == output;
  }
  ```

通过适当的转换操作，我们现在可以覆盖 _ToyInlinerInterface_ 上必要的挂钩，以便在必要时为我们插入它。重新跑一下示例，这次我们得到了想要的结果：_multiply_transpose_ 的两次调用已经完全被 inline 到了 main 函数里边，并且因为使用了通用内联器，其中也会执行简化，因此输出可能比预期的要干净一些。

```MLIR
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.cast %1 : tensor<2x3xf64> to tensor<*xf64>
  %3 = toy.cast %0 : tensor<2x3xf64> to tensor<*xf64>
  %4 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
  %5 = toy.transpose(%3 : tensor<*xf64>) to tensor<*xf64>
  %6 = toy.mul %4, %5 : tensor<*xf64>
  toy.print %6 : tensor<*xf64>
  toy.return
}
```

### 2.2 Intraprocedural Shape Inference

现在我们已经内联了所有函数，剩下的是一个包含静态和动态形状操作混合的 main 函数。我们现在可以编写一个简单的形状推理 _pass_ 来（在单个函数内）传播形状。我们可以将其写成直接对 Toy Dialect 中 _operation_ 约束进行编码的 _pass_，但看起来这更应该以通用的方式来写这个 _transformation_。 作为一个好的经验法则，最好尽可能通用地表达 _transformation_，以便将来可以扩展到其他方言。不知道有多少其他方言可能有类似的需求或遇到相同的问题。

对于形状推断，如果我们将问题进行核心分解，我们实际上只希望 _operation_ 告诉我们给定一组静态已知输入的预期输出。（我们肯定可以变得比这更复杂，但为了我们的需要，我们可以保持简单。）鉴于此属性是特定操作的核心，我们可以定义一个操作接口，可以在需要其结果的操作上指定形状推断。

与操作类似，我们也可以使用操作定义规范 (ODS) 框架来定义操作接口。 该接口是通过从 OpInterface 继承来定义的，它将要提供给生成的 C 接口类的名称作为模板参数。 出于我们的目的，我们将简单地将生成的类命名为 ShapeInference。 我们还提供了接口的描述。

  ```TB
  def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
    let description = [{
      Interface to access a registered method to infer the return types for an
      operation that can be used during type inference.
    }];
  }
  ```

接下来，我们定义操作需要提供的接口方法。 接口方法包括：

* description
* C++ return type in string form
* method name in string form
* a few optional components (根据需要)

参考[ODS documentation](https://mlir.llvm.org/docs/Interfaces/#attributeoperationtype-interfaces)

  ```TB
  def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
    ...

    let methods = [
      InterfaceMethod<"Infer and set the output shape for the current operation.",
                      "void", "inferShapes">
    ];
  }
  ```

现在接口已定义，我们可以将其添加到必要的 Toy Operations中，其方式与我们将 _CallOpInterface_ 添加到 _GenericCallOp_ 的方式类似：

  ```TB
  def MulOp : Toy_Op<"mul",
      [..., DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    ...
  }
  ```

这些操作中的每一个都需要为 inferShapes() 方法提供定义。例如，对于乘法操作，结果形状被推断为输入的形状。

  ```C++
  /// Infer the output shape of the MulOp, this is required by the shape inference
  /// interface.
  void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
  ```

此时，每个必要的 Toy Operation 都提供了一种机制来推断其输出形状。 _ShapeInferencePass_ 将对函数进行操作：它将在每个函数上独立运行。MLIR 还支持在任何孤立操作上运行的通用 _OperationPass_，但这里我们的模块仅包含函数，因此无需泛化到所有操作。通过创建一个继承自 _mlir::OperationPass\<FuncOp>_ 的类并覆盖 _runOnOperation()_ 方法来实现这样的 _pass_。

  ```C++
  class ShapeInferencePass
      : public mlir::PassWrapper<ShapeInferencePass, OperationPass<FuncOp>> {
    void runOnOperation() override {
      FuncOp function = getOperation();
      ...
    }
  };
  ```

同时，我们还创建一个辅助方法来实例化 _pass_：

  ```C++
  std::unique_ptr<mlir::Pass> mlir::toy::createShapeInferencePass() {
    return std::make_unique<ShapeInferencePass>();
  }
  ```

形状推断算法的操作如下：

* 构建一个工作列表，其中包含返回动态形状张量的所有操作：这些是需要形状推断的操作。
* 迭代工作列表：
  * 找到一个要处理的操作：工作列表中的下一个就绪操作的所有参数都是非泛型的
  * 如果没有找到操作，则跳出循环
  * 从工作列表中删除操作
  * 根据参数类型推断输出的形状。
* 如果工作列表为空，则算法成功。

在处理描述的操作时，我们使用以下代码片段查询它是否注册了 _ShapeInference_ 接口：

  ```C++
  // Ask the operation to infer its output shapes.
    LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");

    /// We check if an operation has a particular interface by casting.
    if (ShapeInference shapeOp = dyn_cast<ShapeInference>(op)) {
      shapeOp.inferShapes();
    } else {
      op->emitError("unable to infer shape of operation without shape "
                    "inference interface");
      return signalPassFailure();
    }
  ```

添加 _pass_ 到 _pipeline_ 中：

  ```C++
  pm.addPass(mlir::createShapeInferencePass());
  ```

如果重新运行我们原来的例子，现在得到以下内容：

```MLIR
toy.func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %2 = toy.mul %1, %1 : tensor<3x2xf64>
  toy.print %2 : tensor<3x2xf64>
  toy.return
}
```

测试： ```toyc-ch4 test/Examples/Toy/Ch4/codegen.toy -emit=mlir -opt```
