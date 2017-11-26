package dcs

import scala.collection.mutable.ArrayBuffer
import tea.Utils

/*
A training example, which includes the basic case (BaseExample), which is just
an input sequence of words and an answer.

But it also contains the other case where a training example is a basket, which
contains a set of constraints.  This is not currently used.
*/

trait Example {
  def id : String
  def hasTrueAnswer : Boolean
  def summary : String
  def toStatementItem : StatementItem
}

// A single sentence
class BaseExample(val id:String, val words:List[String], val world:World, val trueAnswer:Answer=null, val groupId:String=null) extends Example {
  var trueExpr : ExprItem = null // Datalog expression
  var trueTree : Node = null // Logical tree
  var isSimple = false // e.g., for the answer of a question

  def hasTrueAnswer = trueAnswer != null
  def summary = Utils.fmts("%s => %s | %s", words.mkString(" "), Renderer.render(trueAnswer), trueExpr)

  def toStatementItem : Rule = {
    if (trueAnswer == null) return null
    val ansItem = trueAnswer.toTermItem
    if (ansItem == null) return null
    Rule(TermItem("_parse",
      ListItem(words.map{w => StrItem(w)}) ::
      ansItem ::
      {if (groupId == null) Nil else StrItem(groupId) :: Nil}), null)
  }
}

// Constraint: ans --- consAns
case class Constraint(id:String, mode:String, sentence:String, world:World) {
  require (Constraint.isValidMode(mode))
  val words = SentenceTokenizer.tokenizeSentence(sentence)
  val ex = new BaseExample(id, words, world)
  override def toString = mode+" "+sentence
}
object Constraint {
  def isValidMode(mode:String) = mode == "=" || mode == ">" || mode == "<" || mode == "~" || mode == "-"
}

// A basket: a set of constraints.
case class Basket(id:String, world:World) extends Example {
  val constraints = new ArrayBuffer[Constraint]

  def setLastAsAnswer = {
    constraints.last.ex.isSimple = true
  }

  def createConstraint(mode:String, sentence:String) = {
    new Constraint(id+":"+constraints.size, mode, sentence, world)
  }

  def summary = constraints.mkString(" | ")

  def size = constraints.size
  def hasTrueAnswer = constraints.size > 1

  def +=(cons:Constraint) : Unit = {
    if (constraints.size == 0) require (cons.mode == "=")
    constraints += cons
  }
  def addConstraint(mode:String, sentence:String) = {
    this += createConstraint(mode, sentence)
  }

  def toStatementItem = {
    Rule(TermItem("_basket", constraints.map { constraint =>
      ListItem(StrItem(constraint.mode) :: StrItem(constraint.sentence) :: Nil)
    }.toList), null)
  }
}
