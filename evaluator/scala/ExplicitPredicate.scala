package dcs

import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.LinkedHashSet
import EntityTypes.Entity
import EntityTypes.EntityIterable
import tea.Utils

/*
Represent the set of entities in a predicate explicitly.
Maintain indices for fast join operations.
*/

// Index the coordinates for easy projection
class ExplicitPredicate(override val arity:Int, val name:String=null) extends Predicate {
  require (arity >= 0, arity)
  val elements = new LinkedHashSet[Entity] // Used linked structure so that enumerate is deterministic
  def size = elements.size

  lazy val abs = {
    if (arity == 0) this
    else new ExplicitPredicate(arity, name) // Will grow as this relation gets populated
  }

  // Indices: i -> x -> {e : e(i) = x}
  def useIndex = indexMaps != null
  var indexMaps : Array[HashMap[Value,ExplicitPredicate]] = null
  var projMaps : Array[ExplicitPredicate] = null

  // Force an index
  override def createIndex : Unit = {
    if (arity > 1) {
      indexMaps = Utils.map(arity, { i:Int => new HashMap[Value,ExplicitPredicate] })
      projMaps = Utils.map(arity, { i:Int => new ExplicitPredicate(1) }) // coord -> {x}
      elements.foreach(addToIndex)
    }
  }

  /*def clear : Unit = {
    elements.clear
    abs.clear
    if (useIndex) {
      indexMaps.foreach(_.clear)
      projMaps.foreach(_.clear)
    }
  }*/

  private def addToIndex(e:Entity) = {
    require (useIndex)
    e.zipWithIndex.foreach { case (x,i) =>
      indexMaps(i).getOrElseUpdate(x, {
        new ExplicitPredicate(arity, name+":"+i)
      }) += e
      projMaps(i) += Entity(x)
    }
  }

  def +=(e:Entity) : Unit = {
    if (arity != e.size) throw Utils.fails("%s already has arity %s, tried to add %s", name, arity, e.size)
    if (elements.add(e)) {
      if (useIndex) addToIndex(e)
      if (e.size > 0 && !e.head.isInstanceOf[Domain])
        abs += e.map(_.abs)
    }
  }
  def ++=(es:EntityIterable) = es.foreach(this += _)
  def ++=(that:Predicate) = that.enumerate.perform.foreach(this += _)

  def contains(e:Entity) = ConstOperation(e != null && elements.contains(e))
  def enumerate = new ExplicitOperation[EntityIterable](1, 1 max elements.size, elements)

  override def doSelect(indices:List[Int], ie:Entity) = {
    if (indices.size == 0 || indices.size == arity || indexMaps == null) super.doSelect(indices, ie)
    else {
      // Use the index for faster lookup
      if (ie == null) {
        // Rough estimate: assume that elements are distributed uniformly over values
        val numValues = indexMaps(indices(0)).size
        val cost = 1.0 * elements.size / (1 max numValues)
        new LinearOperation[Predicate](cost, throw Utils.impossible)
      }
      else {
        indexMaps(indices.head).get(ie.head) match {
          case Some(pred) =>
            //dbgs("SELECT %s:%s %s => %s:%s", this, abs.render, ie, pred, pred.abs.render)
            pred.select(indices.tail, ie.tail)
          case None => ConstOperation(Predicate.empty(arity))
        }
      }
    }
  }

  override def proj(indices:List[Int]) = {
    if (projMaps != null && indices.size == 1)
      projMaps(indices(0))
    else
      super.proj(indices)
  }
  //override def toString = if (indexMaps != null) name+"-idx" else name

  override def hashCode = elements.hashCode
  override def equals(that:Any) = that match {
    case that:ExplicitPredicate => this.elements == that.elements
    case SingletonPredicate(e) => size == 1 && this.elements.head == e
    case _ => false
  }
}
