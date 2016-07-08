# Moon PEG

The `moon-peg` lib is a parser expression grammar (PEG) library. It also has parser classes that you can extend to build a parser.

## Parser Expression Grammar

### [`moon.peg.grammar.Parser`](src/moon/peg/grammar/Parser.hx)

PEG packrat parser with direct and indirect left recursion support. Recursive descent can't handle left recursive grammar. This parser is like recursive descent, but uses some techniques from [the paper by Warth, Douglass, Millstein](http://www.vpri.org/pdf/tr2007002_packrat.pdf), such as memoization, to support both direct and indirect left recursion.

There's only ordered choice and greedy choice. No unordered choice (context-free grammar).

There's 2 ways of using this class:

1. Use Parser class without class parameters. This allows you to load any grammar files at runtime.
     
   ```haxe
   var p = new Parser(peg);
   var ast = p.parse(codes);
   ```

2. Use Parser class with String parameter. This uses haxe's genericBuild, so the grammar file is processed at compile-time. You do not need the grammar file after compilation.
     
   ```haxe
   var p = new Parser<"data/lisp.peg">();
   var ast = p.parse(codes);
   ```
   
#### Grammar Examples

This is an example of an expression grammar that you might find online or in textbooks.

```
S = AB*
AB = A B
A = 'a'+
B = 'b'+
```

Direct and indirect recursion example:

```
$a = "x" / "y";

// direct recursion
s = s a / a;

// indirect recursion
r = t;
t = r a / a;
```

For another example, see [LispTest.hx](test/moon/peg/LispTest.hx), [LispParser.hx](src/moon/peg/lang/LispParser.hx), and its corresponding grammar file [Lisp.peg](data/Lisp.peg).

#### Grammar Rules

##### The ParseTree Enum

ParseTree                      | Notes
-------------------------------|-------
`Empty`                        | No value
`Value(v:String)`              | terminal value
`Tree(v:ParseTree)`            | single value (for pass-thru)
`Multi(a:Array<ParseTree>)`    | multi-values (for seq, A*, A+, etc...)
`Node(id:String, v:ParseTree)` | with child nodes


##### Basic Operations

Category                | Example    | Notes
------------------------|------------|-------
Rule                    | S = A      | &nbsp;
Regular Expression      | [A-Z]+     | &nbsp;
Rule Reference          | A          | &nbsp;
Empty                   | epsilon    | Always succeeds. Matches empty string.
Back Reference          | 2          | Any integer. 2 refers to 2nd matched item
End of Rule             | A;         | Semicolons are automatically inserted. In ambiguous cases, you need to manually add the semicolon.
Ordered Choice          | A / B      | Returns the first success
Greedy Choice*          | A &#124; B | Both evaluated. Result that consumes more is used.
Sequence                | A B        | &nbsp;
Grouping                | (A &#124; B) C  | Sequence of (A or B) followed by C.
Zero or More            | A*         | Greedy match
One or More             | A+         | Greedy match
Optional                | A?         | &nbsp;
Look-ahead              | &A         | Does not consume match
Negative Look-ahead     | !A         | Does not consume match

##### Special Operations

Category                | Example    | Notes
------------------------|------------|-------
Hide                    | @A         | Matches and consumes. If successful, return Empty instead.
Pass                    | $A         | Unwrap result of A. i.e. $(Node("X", v)) ==> v
Anon                    | %A         | Prevent creation of Node to current rule.
Transform               | A:X        | Wrap result of A to a Node
Transform               | A:","      | Flatten result to value separated by a String
Transform               | A:n        | If n=0, return original result. Otherwise, return nth item that's matched
Transform               | A:(1 0)    | Create a Multi where the numbers are the indexes of the result of A
Transform               | A:$f       | Calls a custom transformation function

##### Anon

Prevent creation of Node to current rule.
`$A = B` ==> `A = %B`

Usage: `A = B | %C`
- if B matches ==> Node("A", resultOfB)
- if C matches ==> resultOfC


##### Transform

You can define transformations of nodes within the grammar.

Eg. A:(1 0)
You can use this to resequence result. If resultOfA is Multi(["abc", "123"]) then,

- A:(2 1)     ==> Multi(["123", "abc"])
- A:(2 1 0)   ==> Multi(["123", "abc", "abc", "123"]) // 0 is "abc", "123"
- A:(2 1 2 1) ==> Multi(["123", "abc", "123", "abc"])
- A:(2 "-" 1) ==> Multi(["123", "abc-123", "abc"])


Transformations can be nested. The result of one transformation, can be further transformed, like in the following rule:
- A:(0:"-" 1)

## Todos

- Line/character position info.
- Refactor the macro that generates the parser at compile-time (ugly code done long ago).
- Unit tests
- Optimize

## Contributing

Feel free to contribute. Contributions, bug fixes, and feature requests are welcomed.


## Credits
  
- Parser `moon.peg.grammar` (reference, ideas)
  Warth, Douglass, Millstein: http://www.vpri.org/pdf/tr2007002_packrat.pdf
  Mark Engelberg: https://github.com/Engelberg/instaparse
  
## License
  
MIT