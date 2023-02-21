# MLIR Toy Chapter 1: Toy Language and AST

本教程展示了如何在MLIR的框架下创建一门编程语言（Toy）以及创建编程语言的抽象语法树。

## 1. 基础知识

### 1.1 AST (Abstract Syntax Trees)

抽象语法树是用于描述编程语言 ___源码___ 的 ___抽象语法___ 的 ___树的表示___。树上的每一个结点描述了源码中的一个构造发生(construct occurring)。语法是“抽象的”，因为它不代表真实语法中出现的每个细节，而只是结构（或与内容相关的细节）。 例如分组括号在树结构中是隐含的，因此不必将它们表示为单独的节点。 同样，像 if-condition-then 表达式这样的句法结构可以通过具有三个分支的单个节点来表示（Branch：[1] condition [2] if-body [3] else-body）。这将抽象语法树与具体语法树（传统上称为解析树: _parse trees_）区分开来。 解析树通常由解析器在源代码翻译和编译过程中构建。 一旦构建完成，附加信息就会通过后续处理（例如上下文分析）添加到 AST 中。 ___抽象语法树用于程序分析和程序转换系统___。

### 1.2 Lexer vs. Parser

_Lexer_（词法分析器）和 _Parser_（解析器）具体的区别是什么？我们从乔姆斯基文法来回答这个问题。

___乔姆斯基文法___
乔姆斯基文法（Chomsky）是计算科学形式文法的一个分支。形式文法描述形式语言的基本想法是，从一个特殊的初始符号出发，不断的应用一些产生式规则，从而生成出一个字串的集合。产生式规则指定了某些符号组合如何被另外一些符号组合替换。

|文法|语言|自动机|产生式规则|
|-------|----|------|---------|
|level 0|递归可枚举语言|图灵机|α -> β（无限制）|
|level 1|上下文相关语言|线性有界非确定图灵机|αAβ -> αγβ|
|level 2|上下文无关语言|非确定下推自动机|A-> γ|
|level 3|正规语言|有限状态自动机|A->aB A->a|

- ___Level 0___（无限制文法或短语结构文法）包括所有的文法。该类型的文法能够产生所有可被图灵机识别的语言。可被图灵机识别的语言是指能够使图灵机停机的字串，这类语言又被称为递归可枚举语言。注意递归可枚举语言与递归语言的区别，后者是前者的一个真子集，是能够被一个总停机的图灵机判定的语言。
- ___Level 1___（上下文相关文法）生成上下文相关语言。这种文法的产生式规则取如 αAβ -> αγβ 一样的形式。这里的A是非终结符号，而 α, β 和 γ 是包含非终结符号与终结符号的字串；α, β 可以是空串，但 γ 必须不能是空串；这种文法也可以包含规则 S->ε ，但此时文法的任何产生式规则都不能在右侧包含 S 。这种文法规定的语言可以被线性有界非确定图灵机接受。
- ___Level 2___（上下文无关文法）生成上下文无关语言。这种文法的产生式规则取如A-> γ 一样的形式。这里的A是非终结符号，γ 是包含非终结符号与终结符号的字串。这种文法规定的语言可以被非确定下推自动机接受。上下文无关语言为大多数程序设计语言的语法提供了理论基础。
- ___Level 3___（正规文法）生成正规语言。这种文法要求产生式的左侧只能包含一个非终结符号，产生式的右侧只能是空串、一个终结符号或者一个终结符号后随一个非终结符号；如果所有产生式的右侧都不含初始符号 S ，规则 S -> ε 也允许出现。这种文法规定的语言可以被有限状态自动机接受，也可以通过正则表达式来获得。正规语言通常用来定义检索模式或者程序设计语言中的词法结构。

___Lexer vs. Parser___

- 他们都从各自的输入中读取各自需要的 _Symbols_
  - _Lexer_： 一般读入的是 ASCII Characters
  - _Parser_：读入 _Token_
- 从乔姆斯基文法来看他们理解的文法类型：
  - _Lexer_： Regular Grammar (Chomsky's level 3)
  - _Parser_：Context-free Grammar (Chomsky's level 2).
- 他们都会将语义附件到各自输出的语言片段上：
  - _Lexer_：通过对 _lexemes_（词素）进行分类将语义附加到特别的 _Token_ 上。例如C++的 _Lexer_ 会将'+'，'=='，'/'，等等分类成 _'operator' Token_。
  - _Parser_：会将输入的 _Tokens_ 分类成特别的 _nonterminals_，并且构建一个 _parse tree_。例如C++的 _Parser_ 会将[number][operator][number], [id][operator][id], [id][operator][number][operator][number]这些输入的Tokens分类成 _"expression" nonterminal_
- 他们都可以为识别的元素附加一些额外的含义（数据）。
  - 当 _Lexer_ 识别出构成正确数字的字符序列时，它可以将其转换为其二进制值并与“数字”标记一起存储。
  - 当 _Parser_ 识别出一个表达式时，它可以计算它的值并与语法树的“表达式”节点一起存储。
- 他们都在他们的输出中产生了他们识别的语言的正确句子。
  - _Lexer_ produce tokens
  - _Parser_ produce syntax trees

## 2. Toy Language AST

### 语言

```ToyLanguage
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  # Define a variable `a` with shape <2, 3>, initialized with the literal value.
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];

  # This call will specialize `multiply_transpose` with <2, 3> for both
  # arguments and deduce a return type of <3, 2> in initialization of `c`.
  var c = multiply_transpose(a, b);

  # A second call to `multiply_transpose` with <2, 3> for both arguments will
  # reuse the previously specialized and inferred version and return <3, 2>.
  var d = multiply_transpose(b, a);

  # A new call with <3, 2> (instead of <2, 3>) for both dimensions will
  # trigger another specialization of `multiply_transpose`.
  var e = multiply_transpose(b, c);

  # Finally, calling into `multiply_transpose` with incompatible shape will
  # trigger a shape inference error.
  var f = multiply_transpose(transpose(a), c);
}
```

### 组件

- Main
  - 命令行解析工具：```cl::ParseCommandLineOptions(argc, argv, "toy compiler\n")```
  - 调用 _Lexer_：```LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename))```
  - 调用 _Parser_：```Parser parser(lexer)```
  - 返回```return parser.parseModule()```
- Lexer
  - 输入的是上述源码
  - 输出的是Token：

  |Token|Semantics|Descript|
  |-----|---------|--------|
  |tok_semicolon       |';'| |
  |tok_parenthese_open |'('| |
  |tok_parenthese_close|')'| |
  |tok_bracket_open    |'{'| |
  |tok_bracket_close   |'}'| |
  |tok_sbracket_open   |'['| |
  |tok_sbracket_close  |']'| |
  |tok_eof             |EOF| |
  |tok_return          |"return"| |
  |tok_var             |"var"   | |
  |tok_def             |"def"   | |
  |tok_identifier      |ID| |
  |tok_number          |Number| |

- AST Expression
  - _VarDeclExprAST_: ```decl ::= var identifier [ type ] = expr```
    - Sample：```var a = [[1, 2, 3], [4, 5, 6]];```
    - 构成：
      - ```std::string name;```
      - ```VarType type;```
      - ```std::unique_ptr<ExprAST> initVal;```
  - _VariableExprAST_ : ```decl_list ::= identifier | identifier, decl_list```
    - 变量表达式，构成：
      - ```std::string name;```
  - _NumberExprAST_ : ```numberexpr ::= number```
    - 数字表达式，构成：
      - ```double Val;```
  - _LiteralExprAST_ :
    - 字面量表达式，使用值和维度信息来描述。构成“
      - ```std::vector<std::unique_ptr<ExprAST>> values;```
      - ```std::vector<int64_t> dims;```
  - _ReturnExprAST_: ```return :== return ; | return expr ;```
    - 可以返回空，或表达式（可选）。构成：
      - ```llvm::Optional<std::unique_ptr<ExprAST>> expr;```
  - _BinaryExprAST_:
    - 二值计算表达式，构成：
      - ```char op;```
      - ```std::unique_ptr<ExprAST> lhs, rhs;```
  - _CallExprAST_
    - 函数调用表达式，构成：
      - ```std::string callee;```
      - ```std::vector<std::unique_ptr<ExprAST>> args;```
  - _PrintExprAST_
    - Builtin函数调用，构成：
      - ```std::unique_ptr<ExprAST> arg;```

- AST
  - _PrototypeAST_: ```prototype ::= def id '(' decl_list ')'```
    - __def func_example(arg0, arg1, ...)__
    - 这是一个函数的 ___原型___ 表达（不是函数本身）。构成：
      - ```Location location;```
      - ```std::string name;```
      - ```std::vector<std::unique_ptr<VariableExprAST>> args;```
  - _FunctionAST_:
    - 这是函数定义，由以下部分构成：
      - ```std::unique_ptr<PrototypeAST> proto;```
      - ```std::unique_ptr<ExprASTList> body;```
  - ModuleAST
    - 模块定义，由一系列的Function构成：
      - ```std::vector<FunctionAST> functions;```

### Parser.parseModule()

1. 创建 _std::vector\<___FunctionAST___> functions;_
2. 循环解析 std::unique_ptr\<___FunctionAST___> func = parseDefinition()
   - std::unique_ptr\<___PrototypeAST___> proto = parsePrototype()
   - std::unique_ptr\<___ExprASTList___> block = parseBlock()
   - return std::make_unique\<___FunctionAST___>(std::move(proto), std::move(block));
3. 返回 _std::make_unique\<___ModuleAST___>(std::move(functions));_
4. Dump AST Module

#### parseBlock()

- 以 __“{”__ 为开始，以 __“}”__ 或者EOF 为结束，在此范围中循环查找。
  - ___tok_var___: 解析 VarDeclExprAST
  - ___tok_return___: 解析 ReturnExprAST
  - 以外，解析表达式：
    - identifier
    - number
    - '(' 的时候递归调用解析表达式函数
    - '[' 的时候开始解析 LiteralExprAST

#### Dump AST Module

```AST
Module:
  Function 
    Proto 'multiply_transpose' @test/Examples/Toy/Ch1/ast.toy:4:1'
    Params: [a, b]
    Block {
      Return
        BinOp: * @test/Examples/Toy/Ch1/ast.toy:5:25
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:10
            var: a @test/Examples/Toy/Ch1/ast.toy:5:20
          ]
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:5:25
            var: b @test/Examples/Toy/Ch1/ast.toy:5:35
          ]
    } // Block
  Function 
    Proto 'main' @test/Examples/Toy/Ch1/ast.toy:8:1'
    Params: []
    Block {
      VarDecl a<> @test/Examples/Toy/Ch1/ast.toy:11:3
        Literal: <2, 3>[ <3>[ 1.000000e+00, 2.000000e+00, 3.000000e+00], <3>[ 4.000000e+00, 5.000000e+00, 6.000000e+00]] @test/Examples/Toy/Ch1/ast.toy:11:11
      VarDecl b<2, 3> @test/Examples/Toy/Ch1/ast.toy:15:3
        Literal: <6>[ 1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00] @test/Examples/Toy/Ch1/ast.toy:15:17
      VarDecl c<> @test/Examples/Toy/Ch1/ast.toy:19:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:19:11
          var: a @test/Examples/Toy/Ch1/ast.toy:19:30
          var: b @test/Examples/Toy/Ch1/ast.toy:19:33
        ]
      VarDecl d<> @test/Examples/Toy/Ch1/ast.toy:22:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:22:11
          var: b @test/Examples/Toy/Ch1/ast.toy:22:30
          var: a @test/Examples/Toy/Ch1/ast.toy:22:33
        ]
      VarDecl e<> @test/Examples/Toy/Ch1/ast.toy:25:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:25:11
          var: b @test/Examples/Toy/Ch1/ast.toy:25:30
          var: c @test/Examples/Toy/Ch1/ast.toy:25:33
        ]
      VarDecl f<> @test/Examples/Toy/Ch1/ast.toy:28:3
        Call 'multiply_transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:11
          Call 'transpose' [ @test/Examples/Toy/Ch1/ast.toy:28:30
            var: a @test/Examples/Toy/Ch1/ast.toy:28:40
          ]
          var: c @test/Examples/Toy/Ch1/ast.toy:28:44
        ]
    } // Block
```
