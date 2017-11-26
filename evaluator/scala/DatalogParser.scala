package dcs

/*
Converts a string (e.g., "f(X) :- g(X), h(X,Y).") into a Datalog AST.
Currently implemented using Scala parsing combinators which is convenient,
but very slow.

http://www.codecommit.com/blog/scala/formal-language-processing-in-scala
http://www.codecommit.com/blog/scala/the-magic-behind-parser-combinators
http://debasishg.blogspot.com/2008/04/external-dsls-made-easy-with-scala.html
*/
import scala.util.parsing.combinator.lexical.StdLexical
import scala.util.parsing.combinator.RegexParsers
import scala.util.parsing.combinator.syntactical.StandardTokenParsers
import scala.util.parsing.input.PagedSeqReader
import scala.collection.immutable.PagedSeq
import scala.util.parsing.input.CharArrayReader.EofCh
import tea.Utils

trait Item {
  def args : List[Item]
  def foreachItem(f:Item=>Any) : Unit = {
    f(this)
    args.foreach(_.foreachItem(f))
  }
}

trait ExprItem extends Item
trait AtomItem extends ExprItem
trait StatementItem extends Item

case class IdentItem(name:String) extends AtomItem {
  def args = Nil
  override def toString = name
}
case class NumItem(value:Double) extends AtomItem {
  def args = Nil
  override def toString = value.toString
}
case class StrItem(value:String) extends AtomItem {
  def args = Nil
  override def toString = "'"+value+"'"
}
case class ListItem(elements:List[AtomItem]) extends AtomItem {
  def args = elements
  override def toString = "["+elements.mkString(",")+"]"
}

case class TermItem(name:String, args:List[ExprItem]) extends ExprItem {
  override def toString = {
    val safeName = name match {
      case "=" => "equals"
      case "<" => "lessThan"
      case ">" => "moreThan"
      case name => name
    }
    safeName+"("+args.mkString(",")+")"
  }
}
case class NegItem(arg:ExprItem) extends ExprItem {
  def args = arg :: Nil
  override def toString = "\\+"+arg
}
case class AndItem(args:List[ExprItem]) extends ExprItem {
  override def toString = "("+args.mkString(",")+")"
}
case class OrItem(args:List[ExprItem]) extends ExprItem {
  override def toString = "("+args.mkString(";")+")"
}

// If source == null
case class Rule(target:TermItem, source:ExprItem) extends StatementItem {
  def args = target :: source :: Nil
  //override def toString = target + " :- " + source + "."
  override def toString = {
    if (source == null) target + "."
    else target + " :- " + source + "."
  }
}
/*case class Fact(target:TermItem) extends StatementItem {
  def args = target :: Nil
  override def toString = target + "."
}*/

class MyLexical extends StdLexical with RegexParsers {
  override type Elem = Char
  delimiters ++= List(":-", "(", ")", "[", "]", "\\+", ",", ";", ".", "=", "<", ">", "<=", ">=", "/", "~")

  private val Q = "'".charAt(0)
  private val QQ = "\"".charAt(0)

  override def token: Parser[Token] = 
    ( (letter|'?'|'_') ~ rep(letter|digit|'?'|'_') ^^ { case first ~ rest => processIdent(first :: rest mkString "") }
    | """[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?""".r ^^ { case chars => NumericLit(chars mkString "") }
    | Q ~ rep( chrExcept(Q, '\n', EofCh) ) ~ Q ^^ { case Q ~ chars ~ Q => StringLit(chars mkString "") }
    | QQ ~ rep( chrExcept(QQ, '\n', EofCh) ) ~ QQ ^^ { case QQ ~ chars ~ QQ => StringLit(chars mkString "") }
    | EofCh                                             ^^^ EOF
    | Q ~> failure("unclosed string literal")        
    | QQ ~> failure("unclosed string literal")        
    | delim                                             
    | failure("illegal character")
    )

  override def whitespace: Parser[Any] = rep(
    whitespaceChar
    | '/' ~ '*' ~ comment // Start of comment
    | (('/' ~ '/') | '#' | '%') ~ rep(chrExcept(EofCh, '\n')) // Single line comment
    | '/' ~ '*' ~ failure("unclosed comment")
    )
  override protected def comment: Parser[Any] = (
      '*' ~ '/'  ^^ { case _ => ' ' }
    | chrExcept(EofCh) ~ comment
    )
}

////////////////////////////////////////////////////////////

object DatalogParser extends StandardTokenParsers {
  override val lexical = new MyLexical

  def program : Parser[List[StatementItem]] = (rule|fact)*

  def fact : Parser[Rule] = (termItem ~ ".") ^^ {
    case target ~ "." => Rule(target, null)
  }
  def rule : Parser[Rule] = (termItem ~ ":-" ~ andOrItem ~ ".") ^^ {
    case target ~ ":-" ~ source ~ "." => Rule(target, source)
  }

  // Expr: x = 3 | (f(b,c),g(a)) | \+(...)

  // Atoms
  def atomItem = identItem | numItem | strItem | listItem
  def identItem : Parser[IdentItem] = ident ^^ IdentItem
  def numItem : Parser[NumItem] = numericLit ^^ { case x => NumItem(x.toDouble) }
  def strItem : Parser[StrItem] = stringLit ^^ StrItem
  def listItem : Parser[ListItem] = ("[" ~> repsep(atomItem, ",") <~ "]") ^^ ListItem // [a,b,c]

  // Term: f(a,b)
  def argItem : Parser[ExprItem] = opItem | termItem | parenItem | atomItem
  def termItem : Parser[TermItem] = ident ~ "(" ~ repsep(argItem, ",") ~ ")" ^^ { // f(a,g(b))
    case name ~ "(" ~ args ~ ")" => TermItem(name, args)
  }
  def rel : Parser[String] = "=" | "<" | ">" | "<=" | ">=" | "/"
  def relItem : Parser[TermItem] = (atomItem ~ rel ~ argItem) ^^ {
    case a ~ r ~ b => TermItem(r, a :: b :: Nil)
  }
  def opItem : Parser[TermItem] = (atomItem ~ "/" ~ atomItem) ^^ {
    case a ~ r ~ b => TermItem(r, a :: b :: Nil)
  }

  // Expressions:
  def exprItem : Parser[ExprItem] = negItem | parenItem | relItem | termItem
  def negItem : Parser[NegItem] = ("\\+" ~> exprItem) ^^ NegItem // \+(f(a),g(a))
  def parenItem : Parser[ExprItem] = "(" ~> andOrItem <~ ")"
  def andOrItem : Parser[ExprItem] = andItem | orItem | exprItem
  def andItem : Parser[AndItem] = (exprItem ~ "," ~ rep1sep(exprItem, ",")) ^^ { case firstArg ~ _ ~ args => AndItem(firstArg :: args) }
  def orItem : Parser[OrItem] = (exprItem ~ ";" ~ rep1sep(exprItem, ";")) ^^ { case firstArg ~ _ ~ args => OrItem(firstArg :: args) }

  def load(path:String) : List[StatementItem] = {
    val tokens = new lexical.Scanner(new PagedSeqReader(PagedSeq fromFile path))
    phrase(program)(tokens) match {
      case Success(statements, _) => statements
      case e:NoSuccess => loadSafe(path)
    }
  }

  def loadSafe(path:String) : List[StatementItem] = {
    // load one line at a time, catch exceptions
    Utils.logs("DatalogParser.loadSafe(): start");
    val list = scala.collection.mutable.ListBuffer.empty[StatementItem]
    for(line <- io.Source.fromFile(path).getLines()) {
      Utils.logs(line);
      val tokens = new lexical.Scanner(line)
      phrase(program)(tokens) match {
        case Success(statements, _) => list += statements(0)
        case e:NoSuccess => list += null
      }
    } 
    Utils.logs("DatalogParser.loadSafe(): end");
    list.toList
  }
}
