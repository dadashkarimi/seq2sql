package dcs

import EntityTypes._
import MyUtils.IntList
import tea.Utils
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet

/*
A predicate is a set of tuples of the same arity.
A predicate should support the following operations, each of which have costs: enumerate, contains, select.

Abstract: SingletonPredicate, ExplicitPredicate
Concrete: SingletonPredicate, ExplicitPredicate, InfPredicate,
          SelectedPredicate, RenamedPredicate, ProdPredicate, UnionPredicate [wrappers]
*/
trait Predicate extends Renderable {
  def render : String = {
    val op = enumerate
    if (!op.possible) "{..}"
    else Renderer.render(MyUtils.toHashSet(op.perform))
  }

  def name : String // ABSTRACT
  def size : Double // ABSTRACT
  def abs : Predicate // ABSTRACT
  def arity : Int = abs.arity

  def isError = this match {
    case SingletonPredicate(Entity(ErrorDomain)) => true
    case SingletonPredicate(Entity(TypedValue(_,ErrorDomain))) => true
    case _ => false
  }

  // Abstract and concrete denotation.
  lazy val absDen = Denotation(false, abs, List(arity), List(NullAux)).canonicalize
  lazy val conDen = Denotation(true, this, List(arity), List(NullAux))
  def den(isCon:Boolean) = if (isCon) conDen else absDen

  def corePred = this // What kind of predicate am I really?
  def hasInfiniteSize = size == Predicate.infSize
  def isArgRequired : List[Boolean] = MyUtils.repeat(arity, true) // By default all arguments are required
  def hasInverse = true // Are we allowed to specify the last element of the predicate and get a value out?

  def enumerate : Operation[EntityIterable] // List all the elements
  def contains(e:Entity) : Operation[Boolean] // See if an element is in the set

  // Special case: can join predicates that have arity 1 onto coordinate i
  def join(that:Predicate, i:Int) : Operation[Predicate] = impossibleOperation[Predicate]
  
  def impossibleOperation[A] = new ExplicitOperation[A](Operation.infCost, Operation.infCost, {
    //throw Utils.fails(name+": impossible operation")
    Utils.errors(name+": impossible operation")
    throw InterpretException(name+": impossible operation")
  })

  // Convert to an entity of this format
  def remap(selJ:List[Int], selE:Entity, n:Int=arity) = {
    if (selE == null) null
    else {
      require (selJ.size == selE.size, Renderer.render(selJ) + " " + Renderer.render(selE))
      IntList(n).map{i => selE(selJ.indexOf(i))}
    }
  }

  // Return subset of entities matching at selJ (e), projected onto J.
  def select(selJ:List[Int], selE:Entity) : Operation[Predicate] = {
    if (selJ == Nil) new ExplicitOperation[Predicate](1, enumerate.outCost, this)
    else if (size == 0) ConstOperation(Predicate.empty(arity))
    else if (IntList(arity).forall(selJ.contains)) { // Containment
      val e = remap(selJ, selE)
      contains(e).extendConst { b:Boolean => if (b) SingletonPredicate(e) else Predicate.empty(arity) }
    }
    else {
      val op = doSelect(selJ, selE) // If custom implementation exists, use it
      if (op.possible) op
      else new ExplicitOperation[Predicate](1, Operation.infCost, new SelectedPredicate(this, selJ, selE, MyUtils.IntList(arity))) // Thunk it
    }
  }

  // Override this if we want to do something clever besides containment or enumerate/filter.
  def doSelect(selJ:List[Int], selE:Entity) : Operation[Predicate] = { // Default: enumerate/filter
    val op = enumerate
    new LinearOperation(op.outCost, {
      val entities = op.perform.filter { e => matches(e, selJ, selE) }
      if (entities.size == 0) Predicate.empty(arity)
      else if (entities.size == 1) SingletonPredicate(entities.head)
      else {
        val resultPred = new ExplicitPredicate(arity)
        resultPred ++= entities
        resultPred
      }
    })
  }

  def proj(selJ:List[Int]) : Predicate = { // Default implementation: just enumerate and slice out
    //Utils.dbgs("PROJ %s on %s", this, selJ)
    if (selJ == Nil) {
      if (size == 0) Predicate.empty(0)
      else return Predicate.nil
    }
    if (selJ == IntList(arity)) return this
    val outPred = new ExplicitPredicate(selJ.size, name+"["+selJ.mkString(",")+"]")
    val op = enumerate
    if (!op.possible) throw InterpretException(name+": can't enumerate to project onto "+selJ+" "+getClass)
    op.perform.foreach { e =>
      outPred += selJ.map(e)
    }
    outPred
  }

  // Only use when can enumerate
  def ++(that:Predicate) = {
    val outPred = new ExplicitPredicate(arity, this.name+"+"+that.name)
    outPred ++= this.enumerate.perform
    outPred ++= that.enumerate.perform
    outPred
  }
  def map(f:Entity=>Entity) = {
    val outPred = new ExplicitPredicate(arity, this.name)
    enumerate.perform.foreach { e => outPred += f(e) }
    outPred
  }
  def isSubsetOf(that:Predicate) = this.enumerate.perform.forall(that.contains(_).perform)
  def absIsSubsetOf(that:Predicate) = {
    this.enumerate.perform.forall { e => 
      // Correct thing to do: check subtyping for sets (currently, just punt on sets)
      if (e.exists(_.isInstanceOf[SetDomain])) true
      else that.contains(e).perform
    }
  }

  def rename(newName:String) = new RenamedPredicate(newName, this)

  // Helper methods
  def matches(the_e:Entity, selJ:List[Int], e:Entity) = { // Whether selJ of the_e matches e
    (selJ zip e).forall { case (i,x) => the_e(i) == x }
  }
  def value2pred(x:Value) = SingletonPredicate(Entity(x))
  def values2pred(xs:Value*) = SingletonPredicate(xs.toList)
  def values2pred(xs:Iterable[Value]) = SingletonPredicate(xs.toList)
  def entities2pred(entities:EntityIterable) = {
    val pred = new ExplicitPredicate(entities.head.size)
    pred ++= entities
    pred
  }

  // Assume there is a singleton value
  def pred2value(pred:Predicate) : Option[Value] = {
    if (pred.size == 1) {
      val e = pred.enumerate.perform.head 
      if (e.size == 1) Some(e.head)
      else None
    }
    else
      None
  }

  def createIndex : Unit = { }

  def missingArgError = InterpretException(name+": missing argument")
  def badArgError = InterpretException(name+": bad argument")
  def typeError = InterpretException(name+": type error")
  def emptySetError = InterpretException(name+": empty set")
  def myError(s:String) = InterpretException(name+": "+s)
}
object Predicate {
  val infSize = Double.PositiveInfinity
  val empty = (0 to 100).map{arity => new ExplicitPredicate(arity, "empty")}.toArray
  val nil = SingletonPredicate(Nil)
  def error(msg:String) = SingletonPredicate(Entity(TypedValue(NameValue(msg)::Nil, ErrorDomain)))
  val count = CountPredicate(null, false)

  // Takes a predicate (e.g., population) and return a predicate where the second argument is replaced repeated values.
  def expand(pred:Predicate, dom:TypedDomain) = pred.map {
    case Entity(x, NumValue(CountDomain, n)) => Entity(x, RepeatedValue(dom, n))
    case e => throw Utils.fails("Invalid entity: %s", Renderer.render(e))
  }

  def predString(predicates:List[Predicate]) = if (predicates == Nil) "." else predicates.map(_.name).mkString(",")
}

// Wrap pred with a different name
case class RenamedPredicate(name:String, pred:Predicate) extends Predicate {
  override def render = pred.render
  def size = pred.size
  def abs = pred.abs
  override def corePred = pred
  override def isArgRequired = pred.isArgRequired
  override def hasInverse = pred.hasInverse
  def enumerate = pred.enumerate
  def contains(e:Entity) = pred.contains(e)
  override def join(that:Predicate, i:Int) = pred.join(that, i)
  override def select(selJ:List[Int], selE:Entity) = pred.select(selJ, selE)
  override def proj(selJ:List[Int]) = pred.proj(selJ)
  override def createIndex = pred.createIndex
}

////////////////////////////////////////////////////////////

// pred with selJ:selE is backend
// projJ specifies frontend
case class SelectedPredicate(pred:Predicate, selJ:List[Int], selE:List[Value], projJ:List[Int]) extends Predicate {
  if (!IntList(pred.arity).forall{j => selJ.contains(j) || projJ.contains(j)})
    throw Utils.fails("Not cover space exactly: pred=%s, selJ=%s, selE=%s, projJ=%s", pred, selJ, selE, projJ)

  val abs = pred.abs.select(selJ, selE).perform.proj(projJ)

  override def corePred = pred
  def name = pred.name+"("+(selJ zip selE).map {case (j,x) => j+":"+x}.mkString(",")+")"
  def size = pred.size // Hack: upper bound!

  // Convert frontend to backend
  def projToOrig(pe:Entity) : Option[Entity] = { // Add the selected selE
    Some(IntList(pred.arity).map {i =>
      val si = selJ.indexOf(i) 
      val pi = projJ.indexOf(i)
      if (si != -1 && si != -1) { // Both specified - better agree
        if (selE(si) == pe(pi)) selE(si)
        else return None
      }
      else if (si != -1) selE(si) // Take selected part
      else if (pi != -1) pe(pi) // Take projected part
      else throw Utils.impossible
    })
  }

  def enumerate = pred.enumerate.extendLinear{ es:EntityIterable =>
    es.flatMap { e =>
      if (matches(e, selJ, selE)) Some(projJ.map(e)) else Nil
    }
  }
  def contains(pe:Entity) = {
    if (pe == null) pred.contains(null)
    else {
      projToOrig(pe) match {
        case Some(e) => pred.contains(e)
        case _ => ConstOperation(false)
      }
    }
  }

  // Just append to the arguments which are selected
  override def select(selJ:List[Int], selE:Entity) : Operation[Predicate] = {
    pred.select(this.selJ ++ selJ.map(projJ), if (selE == null) null else this.selE ++ selE)
  }

  override def proj(deltaProjJ:List[Int]) = {
    val newProjJ = deltaProjJ.map(projJ)
    if (IntList(pred.arity).forall{i => selJ.contains(i) || newProjJ.contains(i)})
      SelectedPredicate(pred, selJ, selE, newProjJ) // Just delay if we can
    else // Have no choice but to perform this operation
      super.proj(deltaProjJ)
  }
}

////////////////////////////////////////////////////////////

case class SingletonPredicate(the_e:Entity) extends Predicate {
  def name = "{"+Renderer.render(the_e)+"}"
  def size = 1
  override def arity = the_e.size
  val abs = {
    if (the_e.size == 0 || the_e.head.isInstanceOf[Domain]) this // No change
    else values2pred(the_e.map(_.abs))
  }
  def enumerate = ConstOperation(the_e::Nil)
  def contains(e:Entity) = new SimpleOperation(the_e == e)

  override def hashCode = the_e.hashCode
  override def equals(that:Any) = that match {
    case that:ExplicitPredicate => that.elements.size == 1 && that.elements.head == the_e
    case SingletonPredicate(e) => the_e == e
    case _ => false
  }
}

////////////////////////////////////////////////////////////
// Generic templates for predicates

trait InfPredicate extends Predicate {
  def size = Predicate.infSize
  def enumerate = impossibleOperation[EntityIterable]
}

// Denotation: { (x1,...,xk,y) : y in compute(x1,...,xk)) }
// An entity is a sequence of inputs (could be in any order though) followed by an output.
// Select must have all inputs.
trait FuncPredicate extends InfPredicate {
  def contains(e:Entity) : Operation[Boolean] = {
    compute(if (e == null) null else e.slice(0,arity-1)).extendConst{p:Predicate => p.contains(Entity(e(arity-1))).perform} // Assume possible
  }

  override def doSelect(selJ:List[Int], selE:Entity) = {
    if (selJ.size == arity-1 && IntList(arity-1).forall(selJ.contains)) { // Compute
      val xs = if (selE == null) null else remap(selJ, selE, arity-1)
      compute(xs).extendConst{p:Predicate => ProdPredicate.create(SingletonPredicate(xs), p)}
    }
    else super.doSelect(selJ, selE)
  }

  def compute(in:Entity) : Operation[Predicate] // ABSTRACT
}

// Denotation: { (x,y) : y in compute(x) }
trait ReversibleFuncPredicate extends InfPredicate {
  def contains(e:Entity) : Operation[Boolean] = {
    computeForw(if (e == null) null else e(0)).extendConst{p:Predicate => p.contains(Entity(e(1))).perform} // Assume possible to do containment inside
  }

  override def doSelect(selJ:List[Int], selE:Entity) = {
    if (selJ == List(0)) // Compute forward
      computeForw(if (selE == null) null else selE(0)).extendConst{p:Predicate => ProdPredicate.create(value2pred(selE(0)), p)}
    else if (selJ == List(1)) // Compute backward
      computeBack(if (selE == null) null else selE(0)).extendConst{p:Predicate => ProdPredicate.create(p, value2pred(selE(0)))}
    else super.doSelect(selJ, selE)
  }

  def computeForw(x:Value) : Operation[Predicate] // ABSTRACT
  def computeBack(y:Value) : Operation[Predicate] // ABSTRACT
}

case class ProdPredicate(pred1:Predicate, pred2:Predicate) extends Predicate {
  if (!(pred1.arity > 0 && pred2.arity > 0)) throw InterpretException("Bad ProdPredicate") // HACK
  //require (pred1.arity > 0 && pred2.arity > 0, pred1 + " " + pred2)
  def name = pred1.name + "*" + pred2.name
  def size = pred1.size * pred2.size
  def split = pred1.arity

  val abs = {
    val result = new ExplicitPredicate(pred1.arity + pred2.arity)
    pred1.abs.enumerate.perform.foreach { e1 =>
      pred2.abs.enumerate.perform.foreach { e2 =>
        result += e1 ++ e2
      }
    }
    result
  }
  
  def enumerate = {
    val op1 = pred1.enumerate
    val op2 = pred2.enumerate
    require (op1.outCost > 0 && op2.outCost > 0)
    new LinearOperation[EntityIterable](op1.outCost*op2.outCost, {
      val out = new ArrayBuffer[Entity]
      op1.perform.foreach { e1 =>
        op2.perform.foreach { e2 =>
          out += e1 ++ e2
        }
      }
      out
    })
  }

  override def join(p:Predicate, i:Int) = {
    if (i < split)
      pred1.join(p, i).extendConst{p1:Predicate => ProdPredicate.create(p1, pred2)}
    else
      pred2.join(p, i-split).extendConst{p2:Predicate => ProdPredicate.create(pred1, p2)}
  }

  def contains(e:Entity) = {
    val op1 = pred1.contains(if (e == null) null else e.slice(0, split))
    val op2 = pred2.contains(if (e == null) null else e.slice(split, e.size))
    val retCost = op1.retCost+op2.retCost
    new ExplicitOperation(retCost, if (retCost == Operation.infCost) Operation.infCost else 1, op1.perform && op2.perform)
  }

  override def proj(indices:List[Int]) = {
    val indices1 = indices.filter(_ < split)
    val indices2 = indices.filter(_ >= split)
    if (indices == indices1 ++ indices2) ProdPredicate.create(pred1.proj(indices1), pred2.proj(indices2.map(_-split)))
    else if (indices == indices2 ++ indices1) ProdPredicate.create(pred2.proj(indices2.map(_-split)), pred1.proj(indices1))
    else super.proj(indices)
  }

  override def doSelect(indices:List[Int], ie:Entity) = {
    val indices1 = indices.filter(_ < split)
    val indices2 = indices.filter(_ >= split).map(_ - split)
    val e1 = if (ie == null) null else (indices zip ie).flatMap{case (i,x) => if (i < split) Some(x) else None}
    val e2 = if (ie == null) null else (indices zip ie).flatMap{case (i,x) => if (i >= split) Some(x) else None}
    val op1 = pred1.select(indices1, e1)
    val op2 = pred2.select(indices2, e2)
    new ExplicitOperation[Predicate](op1.retCost+op2.retCost, op1.outCost*op2.outCost, ProdPredicate.create(op1.perform, op2.perform))
  }
}
object ProdPredicate {
  def create(pred1:Predicate, pred2:Predicate) = {
    if (pred1.arity == 0) pred2
    else if (pred2.arity == 0) pred1
    else if (pred1.size == 0 || pred2.size == 0) new ExplicitPredicate(pred1.arity+pred2.arity)
    else ProdPredicate(pred1, pred2)
  }
}

// Note: we don't check for duplicates across the preds
case class UnionPredicate(preds:List[Predicate]) extends Predicate {
  require (preds.size > 1)
  def name = "union"+preds.size
  def size = preds.map(_.size).sum // Not right if there are duplicates

  //Utils.dbgs("UNION %s %s", size, Renderer.render(preds))

  val abs = {
    val result = new ExplicitPredicate(preds.head.arity)
    preds.foreach { pred =>
      pred.abs.enumerate.perform.foreach { e => result += e }
    }
    result
  }

  def enumerate = {
    val ops = preds.map(_.enumerate)
    new LinearOperation[EntityIterable](Operation.totalOutCost(ops), {
      val out = new HashSet[Entity]
      ops.foreach { op => out ++= op.perform }
      out
    })
  }

  def contains(e:Entity) = {
    val ops = preds.map(_.contains(e))
    val cost = Operation.totalRetCost(ops)
    new ExplicitOperation(cost, if (cost == Operation.infCost) Operation.infCost else 1, ops.exists(_.perform))
  }

  override def join(p:Predicate, i:Int) = {
    val ops = preds.map(_.join(p, i))
    new ExplicitOperation[Predicate](Operation.totalRetCost(ops), Operation.totalOutCost(ops), {
      UnionPredicate.create(ops.map(_.perform))
    })
  }

  override def proj(indices:List[Int]) = {
    UnionPredicate.create(preds.map(_.proj(indices)))
  }

  override def doSelect(indices:List[Int], ie:Entity) = {
    val ops = preds.map(_.doSelect(indices, ie))
    new ExplicitOperation[Predicate](Operation.totalRetCost(ops), Operation.totalOutCost(ops), {
      UnionPredicate.create(ops.map(_.perform))
    })
  }
}
object UnionPredicate {
  def create(preds:List[Predicate]) = {
    val filteredPreds = preds.filter(_.size > 0)
    filteredPreds.size match {
      case 0 => Predicate.empty(preds.head.arity)
      case 1 => filteredPreds.head
      case _ => UnionPredicate(filteredPreds)
    }
  }
}
