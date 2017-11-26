package dcs

import scala.collection.mutable.ArrayBuffer
import tea.Utils
import EntityTypes._

trait Value extends Renderable { // Concrete (or abstract) value
  def abs : Value // Return the abstraction of this value
  def size : Double // How many values is actualy represented by this object (usually 1)
}
trait Domain extends Value { // An abstract value
  def abs = this
  def size = Predicate.infSize
  override def toString = render
}

case class TypedDomain(name:String) extends Domain { def render = name }
case class TypedValue(e:Entity, dom:TypedDomain) extends Value { // e.g., (['north', 'dakota'], state)
  def size = 1
  def abs = dom
  def render = e.mkString("_")+":"+dom.name
}
object TypedValue {
  def parse(s:String) : Option[TypedValue] = {
    val i = s.indexOf(':')
    if (i != -1) Some(TypedValue(s.substring(0, i).split('_').map(NameValue).toList, TypedDomain(s.substring(i+1))))
    else None
  }
}

object ErrorDomain extends TypedDomain("error")

object NameDomain extends TypedDomain("name")
case class NameValue(value:String) extends Value {
  def render = value
  def abs = NameDomain
  def size = 1
  override def toString = value
}

trait SetDomain extends Domain
object HeteroSetDomain extends SetDomain { def render = "hetero" }
object EmptySetDomain extends SetDomain { def render = "empty" }
case class SingleSetDomain(e:Entity) extends SetDomain {
  def render = "{["+e.map(_.render).mkString(",")+"]}"
}

case class SetValue(pred:Predicate) extends Value {
  def size = 1
  def render = pred.render

  // Abstract sets of multiple entities.
  def abs = SetDomain.create(pred.abs)
}
object SetDomain {
  def create(pred:Predicate) = {
    pred.size match {
      case 0 => EmptySetDomain
      case 1 => SingleSetDomain(pred.enumerate.perform.head)
      case _ => HeteroSetDomain
    }
  }
}

case class RepeatedValue(base:Domain, size:Double) extends Value {
  def abs = base.abs
  def render = Utils.fmts("%s*%s", base.render, size)
}
