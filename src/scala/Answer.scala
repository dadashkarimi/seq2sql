package dcs

import EntityTypes.Entity
import tea.Utils

/*
An answer is an object that is used to interact with the user,
and essentially wraps a Predicate object (which is a set of entities).
*/
class Answer(val pred:Predicate) extends Renderable {
  def render = pred.render
  def size = pred.size
  def abs = pred.abs
  def contains(e:Entity) = pred.contains(e).perform

  def isError = pred.isError

  override def hashCode = (size, abs).hashCode // Use a very conserative hash
  override def equals(that:Any) = that match {
    case that:Answer =>
      this.pred == that.pred || {
        this.size == that.size && {
          val op = this.pred.enumerate
          op.possible && op.perform.forall(that.contains)
        }
      }
    case _ => false
  }

  def isSubsetOf(that:Answer) = {
    //dbgs("%s %s => %s %s", render(this.pred), render(that.pred), this.abs, that.abs)
    this.size <= that.size && this.abs == that.abs && {
      val op = this.pred.enumerate
      op.possible && op.perform.forall(that.contains)
    }
  }

  def overlapsWith(that:Answer) = {
    this.abs == that.abs && {
      val op1 = this.pred.enumerate
      val op2 = that.pred.enumerate
      if (!op1.possible || !op2.possible) false
      else if (op1.outCost < op2.outCost)
        op1.perform.exists(that.contains)
      else
        op2.perform.exists(this.contains)
    }
  }

  def humanRender : String = {
    def properNoun(s:String) = {
      s.split(" ").map { w =>
        if (w.size == 2) w.toUpperCase
        else w.capitalize
      }.mkString(" ")
    }
    // Answer is a set of entities
    if (pred.isError) "(error)"
    else if (pred.hasInfiniteSize) "(infinite set)"
    else {
      val op = pred.enumerate
      if (!op.possible) Utils.fmts("(%s elements)", Renderer.humanNum(pred.size)) // For exchangeable sets
      else {
        val items = op.perform.map { e =>
          e.map {
            case NumValue(CountDomain, x) => Renderer.humanNum(x)
            case x:NumValue => {
              // Find unit such that when rendered, produces a medium sized number
              val u = x.abs.units.min(Ordering[Double].on[NumUnit] { u =>
                (u.unapplyUnit(x) - 1).abs
              })
              u.humanRender(x)
            }
            case NameValue(s) => "\""+properNoun(s)+"\""
            case TypedValue(e, dom) => e.map { s => properNoun(s.toString) }.mkString(" ") //+ " ["+dom.render+"]"
            case SetValue(pred) => "(set)"
            case RepeatedValue(dom, n) => "("+Renderer.humanNum(n)+" "+dom.toString.replaceAll("""/\d+$""", "")+")"
            case x => throw Utils.fails("Unexpected: %s", x)
          }.mkString(":")
        }
        if (items.size == 0) "(none)"
        else if (items.size == 1) items.head
        else {
          val list = items.toList.sortWith(_ < _)
          if (list.size <= IO.prettyMaxSetSize) list.mkString(", ")
          else list.slice(0, IO.prettyMaxSetSize).mkString(", ")+", ... ("+list.size+" total)"
        }
      }
    }
  }

  def toTermItem : TermItem = {
    val op = pred.enumerate
    if (!op.possible) return null
    val v = IdentItem("A")
    val entities = op.perform
    val body = {
      if (entities.size == 0) TermItem("false1", v::Nil) // Empty
      else {
        OrItem(op.perform.map { // Disjunction over all the elements in the set
          case NameValue(name)::Nil => StrItem(name)
          case NumValue(CountDomain, v)::Nil => NumItem(v)
          case NumValue(dom, v)::Nil => val u = dom.units.head; TermItem(u.name, NumItem(v)::Nil)
          case TypedValue(e, TypedDomain(domName))::Nil => TermItem(domName.split("/")(0), e.map{v => StrItem(v.toString)})
          case x => throw Utils.fails("Invalid item: %s", x)
        }.toList.map{x => TermItem("equals", v :: x :: Nil)})
      }
    }
    TermItem("answer", v :: body :: Nil)
  }
}
