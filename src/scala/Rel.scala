package dcs

/*
A relation is a label on a DCS tree
*/

trait Rel
object Rel {
  val JoinPattern = """(\d+)-(\d+)""".r
  val AnaphoraPattern = """A(\*)?(\d+)""".r
  val ExecutePattern = """X([\d,]+)""".r

  def parse(s:String) : Rel = {
    s match {
      case JoinPattern(s1,s2) => JoinRel(s1.toInt, s2.toInt)
      case "++" => CollectRel
      case "E" => ExtractRel
      case "C" => CompareRel
      case "Q" => QuantRel
      case AnaphoraPattern(b, s) => AnaphoraRel(b != null, s.toInt)
      case ExecutePattern(s) => ExecuteRel(s.split(",").map(_.toInt).toList)
      case _ => null
    }
  }
}

case class JoinRel(j1:Int, j2:Int) extends Rel { override def toString = j1+"-"+j2 }
object EqualRel extends Rel { override def toString = "=" }
object CollectRel extends Rel { override def toString = "++" }

trait MarkerRel extends Rel
object ExtractRel extends MarkerRel { override def toString = "E" }
object CompareRel extends MarkerRel { override def toString = "C" }
object QuantRel extends MarkerRel { override def toString = "Q" }
case class AnaphoraRel(copy:Boolean, i:Int) extends MarkerRel { override def toString = "A"+{if (copy) "*" else ""}+i }

case class ExecuteRel(cols:List[Int]) extends Rel { override def toString = "X"+cols.mkString(",") }
