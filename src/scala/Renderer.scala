package dcs

import scala.collection.mutable.HashSet
import tea.Utils

trait Renderable {
  def render : String
}

object Renderer {
  var displayMaxSetSize = 1

  //// String utilities
  def noun(s:String, n:Double) = {
    if (n == 1) s
    else MorphologicalAnalyzer.theMorpher.pluralize(s)
  }

  // English specific
  def humanNum(value:Double) : String = {
    if (value.isInfinite) "infinite"
    else if (value >= 1e12) humanNum(value/1e12)+" trillion"
    else if (value >= 1e9) humanNum(value/1e9)+" billion"
    else if (value >= 1e6) humanNum(value/1e6)+" million"
    else if (value >= 1e3) "%d,%03d".format((value/1000).toInt, value.toInt%1000)
    else if (value >= 100) value.toInt.toString
    else if (value == value.toInt) value.toInt.toString
    else "%.1f".format(value)
  }

  def simplifyPredName(name:String) = {
    // Simplify (e.g., "{oregon}" => "oregon", "state/1" => "state")
    name.replaceAll("""^\{\[""", "").replaceAll("""\]\}$""", "").replaceAll("""/\d+$""", "")
  }

  def renderTree(x:Any) = TSU.render(x).mkString(" ")

  def render(x:Any) : String = x match {
    case null => "(null)"
    case x:Renderable => x.render
    case x:String => x
    case x:Double => Utils.fmts("%s", x)
    case l:List[Any] => "["+l.map(render(_)).mkString(",")+"]" // Display all
    case l:scala.collection.Set[Any] => // Includes both mutable and immutable
      // Print all elements
      "{"+l.map(render).mkString(",")+"}"
    case x:AnyRef => throw Utils.fails("Unknown: %s %s", x, x.getClass)
    case x => throw Utils.fails("Unknown: %s", x)
  }

  def logTree(s:String, x:Any) = {
    if (s == null) TSU.render(x).foreach(Utils.logs("%s", _))
    else Utils.track_printAll("%s", s) { TSU.render(x).foreach(Utils.logs("%s", _)) }
  }
}
