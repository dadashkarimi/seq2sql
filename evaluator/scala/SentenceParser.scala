package dcs

import tea.Utils
import tea.Tree
import java.util.ArrayList
import tea.BaseTree
import tea.CompositeTree
import tea.CachedLineBasedExecutor
import edu.berkeley.nlp._

/*
Wrapper for a syntactic parser which takes a sentence and returns a constituency or a dependency parse tree.
*/

case class ParseTree(tag:String, children:List[ParseTree]) {
  var span : Span = null
  var head : ParseTree = null // The leaf node
  def isLeaf = children.size == 0
  def isPreterminal = children.size == 1 && children.head.isLeaf

  def toStrTree : Tree = {
    if (children == Nil) BaseTree(tag)
    else CompositeTree(BaseTree(tag) :: children.map(_.toStrTree))
  }

  def tags : List[String] = {
    if (isPreterminal) tag::Nil
    else children.flatMap(_.tags)
  }

  override def toString = {
    if (children == Nil) tag
    else Utils.fmts("(%s[%s]/%s %s)", tag, span, head, children.mkString(" "))
  }
}

case class DepTree(word:String, children:List[DepTree], wordSpan:Span, span:Span) {
  /*def toStrTree : Any = {
    (children.filter(_.span._1 < wordSpan._1).map(_.toStrTree) ++
     Array(word) ++
     children.filter(_.span._1 > wordSpan._1).map(_.toStrTree)).toArray
  }*/
  def toStrTree : Tree = {
    if (children.size == 0) BaseTree(word)
    else CompositeTree(BaseTree(word) :: children.map(_.toStrTree))
  }
}

class SentenceParser {
  type BerkeleyTree = edu.berkeley.nlp.syntax.Tree[String]
  val executor = new CachedLineBasedExecutor(PO.command, PO.cachedPath)

  def forceLaunch = executor.forceLaunch

  def toParseTree(raw:Tree) : ParseTree = raw match {
    case BaseTree(tag) => ParseTree(tag, Nil)
    case CompositeTree(BaseTree(tag) :: children) => ParseTree(tag, children.map(toParseTree))
    case CompositeTree(x::Nil) => toParseTree(x) // This is mostly for the root, which looks like ((S ...))
    case CompositeTree(Nil) => null // No parse tree
    case _ => throw Utils.fails("Invalid: %s", raw)
  }

  def toDepTree(tree:ParseTree) : DepTree = {
    val children = tree.children.map(toDepTree)
    if (tree.head == tree) {
      require (children.size == 0)
      DepTree(tree.tag, children, tree.span, tree.span)
    }
    else {
      val headSubtree = (tree.children zip children).find(_._1.head == tree.head).get._2
      DepTree(headSubtree.word,
        children.filter {subtree => subtree.span._1 < headSubtree.span._1} ++
        headSubtree.children ++
        children.filter {subtree => subtree.span._1 > headSubtree.span._1},
        headSubtree.wordSpan,
        tree.span)
    }
  }

  def setSpans(tree:ParseTree, i:Int) : Int = {
    if (tree.children == Nil) {
      tree.span = Span(i,i+1)
    }
    else {
      var j = i
      tree.children.foreach { subtree =>
        j = setSpans(subtree, j)
      }
      tree.span = Span(i,j)
    }
    tree.span._2
  }

  def setHeads(tree:ParseTree) : BerkeleyTree = {
    val lingChildren = new ArrayList[BerkeleyTree]
    tree.children.foreach {subtree => lingChildren.add(setHeads(subtree))}
    val lingTree = new BerkeleyTree(tree.tag, lingChildren)
    val lingHead = SentenceParser.headFinder.determineHead(lingTree)
    //dbgs("HEAD of %s is %s", lingTree, lingHead)
    if (lingHead == null)
      tree.head = tree
    else {
      val i = lingChildren.indexOf(lingHead)
      require (i != -1)
      tree.head = tree.children(i).head
      //dbgs("HEAD of %s is %s", tree, tree.head)
    }
    lingTree
  }

  def parse(rawSentence:String) : ParseTree = {
    var sentence = rawSentence.trim
    if (PO.lowercase) sentence = sentence.toLowerCase

    val output = executor(sentence)
    val tree = toParseTree(TSU.rawToTree(TSU.stringToNodes(output)(0)))
    if (tree != null) {
      setSpans(tree, 0)
      setHeads(tree)
    }
    tree
  }

  def load = executor.load
  def save = executor.save
  def finish = executor.finish
}

class SentenceParserOptions {
  import tea.OptionTypes._
  @Option(gloss="Path to sentence parser (stdin:sentences, stdout:parse trees)") var command : String = null
  @Option(gloss="Path to store cached parse trees") var cachedPath : String = null
  @Option(gloss="Path to store cached parse trees") var lowercase = true
}
object PO extends SentenceParserOptions

object SentenceParser {
  var _theParser : SentenceParser = null
  def theParser = {
    if (_theParser == null) _theParser = new SentenceParser
    _theParser
  }
  def launchTheParser = theParser.forceLaunch
  def initTheParser = { theParser }
  def finishTheParser = { if (_theParser != null) _theParser.finish }

  val headFinder = new ling.CollinsHeadFinder

  def main(args:Array[String]) : Unit = {
    PO.command = { if (args.size > 0) args(0) else null }
    PO.cachedPath = { if (args.size > 1) args(1) else null }
    val parser = new SentenceParser
    Utils.foreachLine("/dev/stdin", { line:String =>
      println(parser.parse(line))
      parser.save
      true
    })
    parser.finish
  }
}
