# Chapter 2: Emitting Basic MLIR

遍历 AST 以产生 MLIR 中的一个方言，介绍基本 MLIR 概念。在这里，我们展示了如何开始将语义附加到我们在 MLIR 中的自定义操作中。

## 1. Introduction: Multi-Level Intermediate Representation

LLVM和其它的Compiler一样，会预先提供一组固定的类型和指令集定义。在产生 LLVM IR 之前，由给定语言的前端执行任何特定于语言的类型检查、分析或转换。比如Clang将使用AST做静态分析和执行转换。多个不同的前端都会做各自的分析和转换，虽然算法需求上是基本相同的，但每个前端都无法利用别人的基础设施，重复造轮子。MLIR 通过其可扩展性解决这个问题，很少有预定义的类型和指令。

## 2. Interfacing with MLIR

语言参考（略）：

- High-Level Structure
- Notation
  - Common syntax
  - Identifiers and keywords
- ___Dialects___
  - Target specific operations
- ___Operations___
  - Builtin Operations
- Blocks
- Regions
  - Definition
  - Value Scoping
  - Control Flow and SSACFG Regions
  - Graph Regions
  - Arguments and Results
- Type System
  - Type Aliases
  - Dialect Types
  - Builtin Types
- Attributes
  - Attribute Value Aliases
  - Dialect Attribute Values
  - Builtin Attribute Values

MLIR 被设计成一个完全可扩展的基础设施； 没有封闭的属性集（想想：常量元数据）、操作或类型。 MLIR 通过方言的概念支持这种可扩展性。 方言在唯一的命名空间下为抽象提供了分组机制。

### _Operation_

在MLIR中 _Operation_ 是计算和抽象的核心单元，类似于LLVM的Instruction。操作可以具有特定于应用程序的语义，可用于表示 LLVM 中的所有核心 IR 结构：指令、全局变量（如函数）、模块等。MLIR 中的操作集是可扩展的。操作使用一小组概念建模，使操作能够被一般地推理和操作。这些概念是：

- A name for the operation.
- A list of SSA operand values.
- A list of attributes .
- A list of types for result values.
- A source location for debugging purposes.
- A list of successors blocks (for ___branches___, mostly).
- A list of regions (for structural operations like ___functions___).

在 MLIR 中，每个 _Operator_ 都有一个与之关联的强制源位置（source location）这是核心要求，很多APIs是基于Location进行工作的。位置信息默认不会打印，需要加 -mlir-print-debuginfo。

## 3. Toy Dialect

Toy语言为了方便的和MLIR进行交互，我们添加一个Dialect来支持。添加新的方言可以使用C++代码：

```C++
/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types. It can
/// also override virtual methods to change some general behavior, which will be
/// demonstrated in later chapters of the tutorial.
class ToyDialect : public mlir::Dialect {
public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace.
  static llvm::StringRef getDialectNamespace() { return "toy"; }

  /// An initializer called from the constructor of ToyDialect that is used to
  /// register attributes, operations, types, and more within the Toy dialect.
  void initialize();
};
```

MLIR也提供了TableGen的方式，这样可以节省大量的基础工作同时也能让文档和代码生成在一起：

```TableGenDialect
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "toy";

  // A short one-line summary of our dialect.
  let summary = "A high-level dialect for analyzing and optimizing the "
                "Toy language";

  // A much longer description of our dialect.
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "toy";
}
```

我们看一下它将生成什么样的结果：

```C++
namespace mlir {
namespace toy {

class ToyDialect : public ::mlir::Dialect {
  explicit ToyDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<ToyDialect>()) {
    
    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~ToyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("toy");
  }
};
} // namespace toy
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::ToyDialect)
```

## 4. Toy Operations

有了Dialect，我们接下来看看如何添加Operation到Dialect。

```Operation
%4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

___toy.constant()___ 有0个输入，一个 “dense elements” 属性叫value，用于表示constant的值。
以及一个RankedTensorType的输出。从CRTP（奇异递归模板模式）类型的mlir::Op中继承：

```C++
class ConstantOp : public mlir::Op<
                     /// `mlir::Op` is a CRTP class, meaning that we provide the
                     /// derived class as a template parameter.
                     ConstantOp,
                     /// The ConstantOp takes zero input operands.
                     mlir::OpTrait::ZeroOperands,
                     /// The ConstantOp returns a single result.
                     mlir::OpTrait::OneResult,
                     /// We also provide a utility `getType` accessor that
                     /// returns the TensorType of the single result.
                     mlir::OpTraits::OneTypedResult<TensorType>::Impl> {

 public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// Provide the unique name for this operation. MLIR will use this to register
  /// the operation and uniquely identify it throughout the system. The name
  /// provided here must be prefixed by the parent dialect namespace followed
  /// by a `.`.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Return the value of the constant by fetching it from the attribute.
  mlir::DenseElementsAttr getValue();

  /// Operations may provide additional verification beyond what the attached
  /// traits provide.  Here we will ensure that the specific invariants of the
  /// constant operation are upheld, for example the result type must be
  /// of TensorType and matches the type of the constant `value`.
  LogicalResult verify();

  /// Provide an interface to build this operation from a set of input values.
  /// This interface is used by the `builder` classes to allow for easily
  /// generating instances of this operation:
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// This method populates the given `state` that MLIR uses to create
  /// operations. This state is a collection of all of the discrete elements
  /// that an operation may contain.
  /// Build a constant with the given return type and `value` attribute.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// Build a constant and reuse the type from the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// Build a constant by broadcasting the given 'value'.
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state,
                    double value);
};
```

将Operation注册到Dialect的初始化函数中：

```C++
void ToyDialect::initialize() {
  addOperations<ConstantOp>();
}
```

## 5.Op vs Operation: Using MLIR Operations

Operation是基类，对所有操作进行一般性的建模，不描述特定Op和Op类型的属性。每一个特定的类型都由一个Operation的一个派生类来表述。比如 ___ConstantOp___ 只是 ___mlir::Operation___ 的智能指针的Wrap，我们定义的一个一个的Op只是语义上有用的、干净的接口，用来构建和连接 ___mlir::Operation___。

```C++
void processConstantOp(mlir::Operation *operation) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(operation);

  // This operation is not an instance of `ConstantOp`.
  if (!op)
    return;

  // Get the internal operation instance wrapped by the smart pointer.
  mlir::Operation *internalOperation = op.getOperation();
  assert(internalOperation == operation &&
         "these operation instances are the same");
}
```

## 6. Using the Operation Definition Specification (ODS) Framework

除了使用C++模板来定义一个特定的Op以外，我们推荐使用MLIR的ODS框架来定义Op。ODS 中的操作是通过从 Op 类继承来定义的。为了简化我们的操作定义，我们将为 Toy 方言中的操作定义一个基类。

```C++
// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

使用TableGen

```TableGen
def ConstantOp : Toy_Op<"constant"> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // Add custom build methods for the constant operation. These methods populate
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      // Call into an autogenerated `build` method.
      build(builder, result, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value. This builder
    // creates a declaration for `ConstantOp::build` with the given parameters.
    OpBuilder<(ins "double":$value)>
  ];


  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Here we invoke
  // a static `verify` method in a C++ source file. This codeblock is executed
  // inside of ConstantOp::verify, so we can use `this` to refer to the current
  // operation instance.
  let verifier = [{ return ::verify(*this); }];
}
```

- Defining Arguments and Results
  - let ___arguments___
  - let ___results___
- Adding Documentation
  - let ___description___
- Attaching build Methods
  - let ___builders___
  - 通过一个Tensor Value创建
  - 通过一个浮点常量来创建
- Verifying Operation Semantics
  - let ___verifier___
  - ODS框架会根据我们给出的约束自动生成必要的验证逻辑，你可以不必检查输入类型的正确性等等。
  - 如果有额外的特别验证需求，也可以通过verifier中内嵌 C Blob 来完成。

## 7. Example

- mlir-tblgen 生成头文件
  - mlir_tablegen(Ops.h.inc -gen-op-decls)
  - mlir_tablegen(Ops.cpp.inc -gen-op-defs)
  - mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
  - mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
- Dialect.h 引用头文件

  ```C++
  #ifndef MLIR_TUTORIAL_TOY_DIALECT_H_
  #define MLIR_TUTORIAL_TOY_DIALECT_H_

  #include "mlir/IR/BuiltinOps.h"
  #include "mlir/IR/Dialect.h"
  #include "mlir/Interfaces/SideEffectInterfaces.h"

  /// Include the auto-generated header file containing the declaration of the toy
  /// dialect.
  #include "toy/Dialect.h.inc"

  /// Include the auto-generated header file containing the declarations of the
  /// toy operations.
  #define GET_OP_CLASSES
  #include "toy/Ops.h.inc"

  #endif // MLIR_TUTORIAL_TOY_DIALECT_H_
  ```

- 输入：(test/Examples/Toy/Ch2/codegen.toy)

  ```Toy
  # RUN: toyc-ch2 %s -emit=mlir 2>&1 | FileCheck %s

  # User defined generic function that operates on unknown shaped arguments
  def multiply_transpose(a, b) {
    return transpose(a) * transpose(b);
  }

  def main() {
    var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
    var b<2, 3> = [1, 2, 3, 4, 5, 6];
    var c = multiply_transpose(a, b);
    var d = multiply_transpose(b, a);
    print(d);
  }
  ```

- 转成 AST Module

  ```AST
  Module:
    Function 
      Proto 'multiply_transpose' @../mlir/test/Examples/Toy/Ch2/codegen.toy:4:1
      Params: [a, b]
      Block {
        Return
          BinOp: * @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:25
            Call 'transpose' [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:10
              var: a @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:20
            ]
            Call 'transpose' [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:25
              var: b @../mlir/test/Examples/Toy/Ch2/codegen.toy:5:35
            ]
      } // Block
    Function 
      Proto 'main' @../mlir/test/Examples/Toy/Ch2/codegen.toy:8:1
      Params: []
      Block {
        VarDecl a<2, 3> @../mlir/test/Examples/Toy/Ch2/codegen.toy:9:3
          Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @../mlir/test/Examples/Toy/Ch2/codegen.toy:9:17
        VarDecl b<2, 3> @../mlir/test/Examples/Toy/Ch2/codegen.toy:10:3
          Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @../mlir/test/Examples/Toy/Ch2/codegen.toy:10:17
        VarDecl c<> @../mlir/test/Examples/Toy/Ch2/codegen.toy:11:3
          Call 'multiply_transpose' [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:11:11
            var: a @../mlir/test/Examples/Toy/Ch2/codegen.toy:11:30
            var: b @../mlir/test/Examples/Toy/Ch2/codegen.toy:11:33
          ]
        VarDecl d<> @../mlir/test/Examples/Toy/Ch2/codegen.toy:12:3
          Call 'multiply_transpose' [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:12:11
            var: b @../mlir/test/Examples/Toy/Ch2/codegen.toy:12:30
            var: a @../mlir/test/Examples/Toy/Ch2/codegen.toy:12:33
          ]
        Print [ @../mlir/test/Examples/Toy/Ch2/codegen.toy:13:3
          var: d @../mlir/test/Examples/Toy/Ch2/codegen.toy:13:9
        ]
      } // Block
  ```

- Generator：将 AST 转化成 MLIR 的 Module Operation（mlir::ModuleOp）
- ```odin/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp```
  - _lexer_ = Lexer(codegen.toy)
  - _parser_ = Parser(lexer)
    - std::unique_ptr\<toy::ModuleAST> moduleAST = parser.parseModule()
  - mlirGen
    - mlir::ModuleOp mlirGen(___ModuleAST___ &moduleAST)
      - mlir::FuncOp mlirGen(___FunctionAST___ &funcAST)
        - mlir::FuncOp mlirGen(___PrototypeAST___ &proto) 使用 builder 创建 FuncOp
        - 建立 symbolTable
        - 在 Function Body 的开头插入 insertion point
        - mlir::LogicalResult mlirGen(___ExprASTList___ &blockAST) 产生 Function 的 Body
          - case: mlir::LogicalResult mlirGen(___ReturnExprAST___ &ret)
            - builder.create\<ReturnOp>(location, expr ? makeArrayRef(expr) : ArrayRef\<mlir::Value>())
          - case: mlir::LogicalResult mlirGen(___PrintExprAST___ &call)
            - builder.create\<PrintOp>(loc(call.loc()), arg)
          - case: mlir::Value mlirGen(___VarDeclExprAST___ &vardecl)
            - 使用 Var 的初始值创建一个 _mlir::Value_用于初始化。
            - 如果 Var 的 _Type_ 是有 _Shape_ 的，那么插入一个 _ReshapeOp_。
            - 将 Var 插入 _symbolTable_ 并且判断是否是已经存在的 Var。
          - default: mlir::Value mlirGen(___ExprAST___ &expr) 处理通用表达式：
            - case: mlirGen(cast\<___BinaryExprAST___>(expr))
              - 先产生 _lhs_ 和 _rhs_ 两个Operation，然后再产生一个BinaryOp。
              - case __'+'__: builder.create\<AddOp>(location, lhs, rhs)
              - case __'*'__: builder.create\<MulOp>(location, lhs, rhs)
            - case: mlirGen(cast\<___VariableExprAST___>(expr))
              - 变量表达式需要这个变量是已经声明了的，需要从 _symbolTable_ 中确认存在并查找。
            - case: mlirGen(cast\<___LiteralExprAST___>(expr))
              - builder.create\<ConstantOp>(loc(lit.loc()), type, dataAttribute)
              - 常量数据会被拍平放在 _dataAttribute_ 里边，然后和 _type_ 一起构建 _ConstantOp_
            - case: mlirGen(cast\<___CallExprAST___>(expr))
              - 由 _call.getArgs()_ 构建 Operand 列表
              - case _call.callee()_ == "transpose": builder.create\<TransposeOp>(location, operands[0])
              - default: builder.create\<GenericCallOp>(location, callee, operands)
            - case: mlirGen(cast\<___NumberExprAST___>(expr))
              - builder.create\<ConstantOp>(loc(num.loc()), num.getValue()) 数字常量
            - default: emitError()
      - theModule.push_back(func) 将生成的Func插入Module

- Dump 生成的 mlir::ModuleOp:

  ```ToyIR
  builtin.module  {
    builtin.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
      %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
      %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
      %2 = toy.mul %0, %1 : tensor<*xf64>
      toy.return %2 : tensor<*xf64>
    }
    builtin.func @main() {
      %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
      %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
      %2 = toy.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>
      %3 = toy.reshape(%2 : tensor<6xf64>) to tensor<2x3xf64>
      %4 = toy.generic_call @multiply_transpose(%1, %3) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
      %5 = toy.generic_call @multiply_transpose(%3, %1) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
      toy.print %5 : tensor<*xf64>
      toy.return
    }
  }
  ```
