package dcs

import scala.collection.mutable.HashMap
import EntityTypes.Entity
import EntityTypes.EntityIterable
import tea.Utils

/*
Defines some standard predicates.
*/

// Denotation: { (x1,...,xk, x1_..._xk:name) }
// Constructor predicate for typed values.
case class ConstructPredicate(name:String, k:Int) extends FuncPredicate {
  def compute(in:Entity) = new SimpleOperation[Predicate](value2pred(TypedValue(in, TypedDomain(name))))
  override def contains(e:Entity) = {
    if (k == 0) { // Special case: test non-descript objects.
      new SimpleOperation({e match {
        case Entity(TypedValue(_, TypedDomain(name2))) => name == name2
        case Entity(RepeatedValue(TypedDomain(name2), _)) => name == name2
        case _ => throw typeError
      }})
    }
    else
      super.contains(e)
  }

  val abs =
    values2pred(MyUtils.repeat(k, NameDomain) ++ List(TypedDomain(name))) ++
    Domains.predWithNumDomains{t => MyUtils.repeat(k, t) ++ List(TypedDomain(name))}

  override def doSelect(selJ:List[Int], selE:Entity) = {
    if (IO.verbose >= 5) Utils.dbgs("CONSTRUCT %s", selE)
    if (selJ.contains(k)) new SimpleOperation({
      // As long as the output exists, we can retrieve the input
      val y = selE(selJ.indexOf(k)).asInstanceOf[TypedValue]
      if (y.e.size != k) Predicate.empty(selJ.size)
      else {
        val e = MyUtils.append(y.e, y)
        if (matches(e, selJ, selE)) SingletonPredicate(e)
        else Predicate.empty(selJ.size)
      }
    })
    else super.doSelect(selJ, selE)
  }
}

// Denotation: { (x,x) }
case class EqualPredicate(name:String) extends ReversibleFuncPredicate {
  val abs = Domains.predWithNumDomains{t => Entity(t,t)} ++
            Domains.predWithTypedDomains{t => Entity(t,t)}
  def computeForw(x:Value) = new SimpleOperation(value2pred(x))
  def computeBack(x:Value) = new SimpleOperation(value2pred(x))
}

// Denotation: { (unitless number value, value with a unit) }
// Example: hours
case class UnitConversionPredicate(name:String, munit:NumUnit) extends ReversibleFuncPredicate {
  val abs = values2pred(CountDomain, munit.abs)

  def computeForw(x:Value) = new SimpleOperation(x match {
    case NumValue(CountDomain, value) => value2pred(munit.applyUnit(value))
    case _ => throw typeError
  })
  def computeBack(y:Value) = new SimpleOperation(y match {
    case y:NumValue if y.abs == munit.abs => value2pred(NumValue(CountDomain, munit.unapplyUnit(y)))
    case _ => throw typeError
  })

  override def join(pred:Predicate, i:Int) = (pred, i) match {
    case (pred:NumRangePredicate, 0) => 
      if (pred.valueType != CountDomain) throw typeError
      ConstOperation(ProdPredicate(pred, new NumRangePredicate(munit.abs, pred.lower, pred.lowerStrict, pred.upper, pred.upperStrict)))
    case _ => super.join(pred, i)
  }
}

case class NumRangePredicate(valueType:NumDomain, lower:Double, lowerStrict:Boolean, upper:Double, upperStrict:Boolean) extends InfPredicate {
  def name = Utils.fmts("%s%s,%s%s", {if (lowerStrict) '(' else '['}, lower, upper, {if (lowerStrict) ')' else ']'})
  def abs = value2pred(valueType)
  def contains(e:Entity) = new SimpleOperation(e match {
    case (x:NumValue) :: Nil if x.abs == valueType => {if (lowerStrict) x.value > lower else x.value >= lower} &&
                                                      {if (upperStrict) x.value < upper else x.value <= upper}
    case _ => throw typeError
  })

  // FUTURE: handle multiple segments, currently just take the convex hull
  /*def union(that:NumRangePredicate) = {
    if (this.valueType != that.valueType) throw typeError
    val (lower, lowerStrict) = {
      if (this.lower < that.lower) (this.lower, this.lowerStrict)
      else if (that.lower < this.lower) (that.lower, that.lowerStrict)
      else (this.lower, this.lowerStrict && that.lowerStrict)
    }
    val (upper, upperStrict) = {
      if (this.upper > that.upper) (this.upper, this.upperStrict)
      else if (that.upper > this.upper) (that.upper, that.upperStrict)
      else (this.upper, this.upperStrict && that.upperStrict)
    }
    NumRangePredicate(valueType, lower, lowerStrict, upper, upperStrict)
  }*/
}

// Denotation: {(x, c*x) : x is a number}
case class ScalePredicate(name:String, factor:Double) extends ReversibleFuncPredicate {
  require (factor != 0)
  val abs = Domains.predWithNumDomains{t => Entity(t,t)}
  def computeForw(x:Value) = new SimpleOperation(x match {
    case x:NumValue => value2pred(x*factor)
    case _ => throw typeError
  })
  def computeBack(x:Value) = new SimpleOperation(x match {
    case x:NumValue => value2pred(x/factor)
    case _ => throw typeError
  })
}

// Denotation: {(x, x.toString) : x is a number}
case class NumToStrPredicate(name:String) extends ReversibleFuncPredicate {
  val abs = values2pred(CountDomain,NameDomain)
  def computeForw(x:Value) = new SimpleOperation(x match {
    case NumValue(CountDomain,v) => value2pred(NameValue(Utils.fmts("%s",v)))
    case _ => throw typeError
  })
  def computeBack(x:Value) = new SimpleOperation(x match {
    case NameValue(v) => value2pred(NumValue(CountDomain, v.toDouble))
    case _ => throw typeError
  })
}

// Denotation: {(a,b,a+b) : a,b are strings}
case class ConcatPredicate(name:String) extends FuncPredicate {
  val abs = values2pred(NameDomain,NameDomain,NameDomain)
  def compute(in:Entity) = new SimpleOperation(in match {
    case Entity(NameValue(s), NameValue(t)) => value2pred(NameValue(s+t))
    case _ => throw typeError
  })
}

// Denotation: {(x, x.toLowerCase) : x is a name}
case class LowercasePredicate(name:String) extends FuncPredicate {
  val abs = values2pred(NameDomain,NameDomain)
  def compute(in:Entity) = new SimpleOperation(in match {
    case Entity(NameValue(s)) => value2pred(NameValue(s.toLowerCase))
    case _ => throw typeError
  })
}

// Denotation: {(x, y) : x begins with y}
case class StartsWithPredicate(name:String) extends InfPredicate {
  val abs = values2pred(NameDomain,NameDomain)
  def contains(e:Entity) = new SimpleOperation(e match {
    case Entity(NameValue(x), NameValue(y)) => x.startsWith(y)
    case _ => throw typeError
  })
}

// Handle negation where the interaction is only one variable.
// Denotation: {(x,y) : x r y}, where r in {<, >, <=, >=}
case class NumRelPredicate(name:String, greater:Boolean, strict:Boolean) extends ReversibleFuncPredicate {
  val abs = Domains.predWithNumDomains{t => Entity(t,t)}

  // More efficient than going through ReversibleFuncPredicate's contains
  override def contains(e:Entity) = new SimpleOperation(e match {
    case (x:NumValue) :: (y:NumValue) :: Nil =>
      if (x.abs != y.abs) throw typeError
      if (greater) { if (strict) x.value > y.value else x.value >= y.value }
      else         { if (strict) x.value < y.value else x.value <= y.value }
    case _ => throw typeError
  })

  def computeForw(x:Value) = new ExplicitOperation[Predicate](1, Operation.infCost, x match {
    case x:NumValue =>
      if (greater) // (> x ?)
        new NumRangePredicate(x.abs, -1.0/0, false, x.value, strict)
      else // (< x ?)
        new NumRangePredicate(x.abs, x.value, strict, 1.0/0, false)
    case _ => throw typeError
  })

  def computeBack(y:Value) = new ExplicitOperation[Predicate](1, Operation.infCost, y match {
    case y:NumValue =>
      if (greater) // (> ? y)
        new NumRangePredicate(y.abs, y.value, strict, 1.0/0, false)
      else // (< ? y)
        new NumRangePredicate(y.abs, -1.0/0, false, y.value, strict)
    case _ => throw typeError
  })
}

// Denotation: {(x, S) : min(S) <= x <= max(S)}
case class BetweenSetPredicate(name:String) extends InfPredicate {
  val abs = Domains.predWithNumDomains{t => Entity(t,SingleSetDomain(Entity(t)))}
  def contains(e:Entity) = new SimpleOperation(e match {
    case Entity(NumValue(dx,x), SetValue(setPred)) =>
      val op = setPred.enumerate
      if (!op.possible) throw myError("Can't enumerate")
      var existsSmaller = false
      var existsLarger = false
      op.perform.foreach {
        case Entity(NumValue(dy,y)) if dx == dy =>
          if (y <= x) existsSmaller = true
          if (y >= x) existsLarger = true
        case _ => throw typeError
      }
      existsSmaller && existsLarger
    case _ => throw typeError
  })
}

// Denotation: {(a, b, x) : a <= x <= b}
case class BetweenPredicate(name:String, strict:Boolean) extends FuncPredicate {
  override def hasInverse = false
  val abs = Domains.predWithNumDomains{t => Entity(t,t,t)}
  def compute(in:Entity) = new SimpleOperation(in match {
    case Entity(NumValue(da,a), NumValue(db,b)) if da == db =>
      NumRangePredicate(da, a, strict, b, strict)
    case _ => throw typeError
  })
}

// Denotation: { (x, y) : y <= x, y is ordinal, x is count }
case class CountTopRankedPredicate(name:String) extends FuncPredicate {
  val abs = values2pred(CountDomain, OrdinalDomain)
  def compute(in:Entity) = new ExplicitOperation(1, Operation.infCost, in match {
    case NumValue(CountDomain,x) :: Nil => new NumRangePredicate(OrdinalDomain, 1, false, x, false)
    case _ => throw typeError
  })
}
case class FracTopRankedPredicate(name:String) extends FuncPredicate {
  val abs = values2pred(FracDomain, FracDomain)
  def compute(in:Entity) = new ExplicitOperation(1, Operation.infCost, in match {
    case NumValue(FracDomain,x) :: Nil => new NumRangePredicate(FracDomain, 0, false, x, false)
    case _ => throw typeError
  })
}

// Denotation: { (a,b,i) : a <= i < b }
case class IntRangePredicate(name:String) extends FuncPredicate {
  val abs = values2pred(CountDomain, CountDomain, CountDomain)
  def compute(in:Entity) = new SimpleOperation(in match { // Not quite constant...
    case NumValue(CountDomain,a) :: NumValue(CountDomain,b) :: Nil =>
      val outPred = new ExplicitPredicate(1)
      (a.toInt to b.toInt-1).foreach { i =>
        outPred += Entity(NumValue(CountDomain,i))
      }
      outPred
    case _ => throw typeError
  })
}

// Denotation: { (f,a_1,...,a_k) : probability f }
case class RandPredicate(name:String, k:Int) extends InfPredicate {
  val abs = k match {
    case 1 => Domains.predWithDomains(Domains.allDomains, {t => Entity(CountDomain,t)})
    case 2 => Domains.predWithDomains(Domains.allDomains, Domains.allDomains, {(t1,t2) => Entity(CountDomain,t1,t2)})
    case _ => throw Utils.fails("%s too large", k)
  }
  def contains(e:Entity) = new SimpleOperation(e match {
    case NumValue(CountDomain, f) :: rest => IO.random.nextDouble < f
    case _ => throw typeError
  })
}

// Denotation: { (lower,upper,a_1,...,a_k) : a_k ~ Unif[lower...upper] }
case class RandIntPredicate(name:String, k:Int) extends FuncPredicate {
  val abs = k match {
    case 1 => Domains.predWithDomains(Domains.allDomains, {t => Entity(CountDomain,CountDomain,t,CountDomain)})
    case 2 => Domains.predWithDomains(Domains.allDomains, Domains.allDomains, {(t1,t2) => Entity(CountDomain,CountDomain,t1,t2,CountDomain)})
    case _ => throw Utils.fails("%s too large", k)
  }
  def compute(in:Entity) = new SimpleOperation(in match {
    case NumValue(_, lower) :: NumValue(_, upper) :: rest =>
      val x = IO.random.nextInt(upper.toInt - lower.toInt) + lower.toInt
      value2pred(NumValue(CountDomain, x))
    case _ => throw typeError
  })
}

case class CountPredicate(name:String, negate:Boolean) extends FuncPredicate {
  val abs = Domains.predWithTypedDomains{t => Entity(SingleSetDomain(Entity(t)),CountDomain)} ++ values2pred(EmptySetDomain,CountDomain)
  override def hasInverse = false
  def count(e:Entity) = e.foldLeft(1.0) { case (acc,v) => acc*v.size }
  def compute(in:Entity) = new SimpleOperation(in match {
    //case SetValue(pred) :: Nil => value2pred(NumValue(CountDomain, pred.size.toDouble * {if (negate) -1 else +1}))
    case SetValue(pred) :: Nil =>
      if (pred.hasInfiniteSize) value2pred(NumValue(CountDomain, Predicate.infSize))
      else {
        val op = pred.enumerate
        if (!op.possible) throw typeError
        value2pred(NumValue(CountDomain, op.perform.foldLeft(0.0) { case (acc,e) => acc+count(e)} * {if (negate) -1 else +1}))
      }
    case _ => throw typeError
  })
}

case class ArithPredicate(name:String, op:Char) extends FuncPredicate {
  val abs = op match {
    case '+' => Domains.predWithNumDomains{t => Entity(t,t,t)}
    case '-' => Domains.predWithNumDomains{t => Entity(t,t,t)}
    case '*' => Domains.predWithDomains(Domains.numDomains, Domains.numDomains, {(t1,t2) => Entity(t1,t2,ProdNumDomain.lookup(t1,t2))})
    case '/' => Domains.predWithDomains(Domains.numDomains, Domains.numDomains, {(t1,t2) => Entity(t1,t2,RatioNumDomain.lookup(t1,t2))})
    case _ => throw Utils.impossible
  }
  def compute(in:Entity) = new SimpleOperation(in match {
    case (x:NumValue) :: (y:NumValue) :: Nil => {
      val z = op match {
        case '+' => x+y
        case '-' => x-y
        case '*' => x*y
        case '/' => x/y
        case _ => throw Utils.impossible
      }
      value2pred(z)
    }
    case _ => throw typeError
  })
}

// Denotation: { (x,S) : x in S }
case class MemberPredicate(name:String) extends InfPredicate {
  val abs = Domains.predWithTypedDomains{t => Entity(t, SingleSetDomain(Entity(t)))} ++
            Domains.predWithTypedDomains{t => Entity(t, EmptySetDomain)}

  def contains(e:Entity) = new SimpleOperation(e match {
    case x :: SetValue(set) :: Nil => set.contains(Entity(x)).perform
    case _ => throw typeError
  })

  override def doSelect(selJ:List[Int], e:Entity) = {
    selJ.indexOf(1) match { // Get the set (must be specified)
      case -1 => impossibleOperation[Predicate]
      case set_i => new SimpleOperation(e(set_i) match {
        case set @ SetValue(setPred) =>
          val x_i = selJ.indexOf(0) // Is x specified?
          if (x_i == -1) // Enumerate all elements of the set
            ProdPredicate.create(setPred, value2pred(set))
          else if (setPred.contains(Entity(e(x_i))).perform) // Check if x is in the set
            SingletonPredicate(Entity(e(x_i), set))
          else Predicate.empty(2)
        case _ => throw typeError
      })
    }
  }
}

// Denotation: {(x1, ..., xk, y) : y = union of xi}
case class SetUnionPredicate(name:String, k:Int) extends FuncPredicate {
  // FUTURE: allow empty sets
  val abs = Domains.predWithTypedDomains{t => MyUtils.repeat(k+1, SingleSetDomain(Entity(t)))}

  override def hasInverse = false

  def compute(in:Entity) = new SimpleOperation({
    val preds = in.map {
      case SetValue(pred:Predicate) => pred
      case _ => throw typeError
    }
    if (!(preds.head.arity >= 0 && preds.forall(_.arity == preds.head.arity))) throw badArgError
    val outPred = new ExplicitPredicate(preds.head.arity)
    preds.foreach { pred =>
      outPred ++= pred.enumerate.perform
    }
    value2pred(SetValue(outPred))
  })
}

trait QuantPredicate extends FuncPredicate {
  override def isArgRequired = List(true, true, false)
  def quantAbs(dom:Domain) = {
    Domains.predWithDomains(Domains.allDomains, {t => Entity(SingleSetDomain(Entity(t)), EmptySetDomain, dom)}) ++
    Domains.predWithDomains(Domains.allDomains, {t => Entity(SingleSetDomain(Entity(t)), SingleSetDomain(Entity(t)), dom)}) ++
    Domains.predWithDomains(Domains.allDomains, Domains.allDomains, {case (t1,t2) => Entity(SingleSetDomain(Entity(t1,t2)), EmptySetDomain, dom)}) ++
    Domains.predWithDomains(Domains.allDomains, Domains.allDomains, {case (t1,t2) => Entity(SingleSetDomain(Entity(t1,t2)), SingleSetDomain(Entity(t1,t2)), dom)})
  }
}

// Denotation: { (r,s,frac) : |r intersect s| = frac*|r|, r is non-empty, s \subset r }
case class FracQuantPredicate(name:String, fracPred:Predicate=null) extends QuantPredicate {
  val abs = quantAbs(FracDomain)

  def compute(in:Entity) = new SimpleOperation(in match {
    case SetValue(pred1) :: SetValue(pred2) :: Nil =>
      //Utils.dbgs("FracQuantPredicate: %s %s", pred1.render, pred2.render)
      val n1 = pred1.size
      val n2 = pred2.enumerate.perform.count{e => pred1.contains(e).perform}
      val f = {
        if (n1 == 0) {
          require (n2 == 0)
          0.0
        }
        else 1.0*n2/n1
      }
      val e = Entity(NumValue(FracDomain, f))
      if (fracPred == null || fracPred.contains(e).perform) // Valid fraction
        SingletonPredicate(e)
      else
        Predicate.empty(1)
    case _ => throw typeError
  })

  override def join(fracPred:Predicate, i:Int) = {
    if (i == 2) ConstOperation(FracQuantPredicate(name, fracPred))
    else super.join(fracPred, i)
  }
}

// Denotation: { (r,s) : |r intersect s| = n }
case class CountQuantPredicate(name:String, countPred:Predicate=null) extends QuantPredicate {
  val abs = quantAbs(CountDomain)

  def compute(in:Entity) = new SimpleOperation(in match {
    case SetValue(pred1) :: SetValue(pred2) :: Nil =>
      val n = pred2.enumerate.perform.count{e => pred1.contains(e).perform}
      val e = Entity(NumValue(CountDomain, n))
      if (countPred == null || countPred.contains(e).perform) // Valid fraction
        SingletonPredicate(e)
      else
        Predicate.empty(1)
    case _ => throw typeError
  })

  override def join(countPred:Predicate, i:Int) = {
    if (i == 2) ConstOperation(CountQuantPredicate(name, countPred))
    else super.join(countPred, i)
  }
}

// Denotation: {({(a,x)}, aggregate value over x's)}
case class AggregatePredicate(name:String, mode:String) extends FuncPredicate {
  val abs =
    values2pred(EmptySetDomain, CountDomain) ++ // FUTURE: can we get the return type, which might not be CountDomain?
    Domains.predWithDomains(Domains.typedDomains, Domains.numDomains, {(at,xt) => Entity(SingleSetDomain(Entity(at,xt)), xt)})

  def combine(v1:NumValue, v2:NumValue) = mode match {
    case "min" => v1 min v2
    case "max" => v1 max v2
    case "maxNegate" => v1 max v2
    case "mean" => v1 + v2
    case "sum" => v1 + v2
    case _ => throw Utils.fails("Bad mode: %s", mode)
  }
  def compute(in:Entity) = new SimpleOperation(in match {
    case SetValue(pred) :: Nil =>
      val op = pred.enumerate
      if (!op.possible) throw myError("can't enumerate")
      var result : NumValue = null
      op.perform.foreach {
        case Entity(a,b:NumValue) =>
          if (result == null) result = b
          else result = combine(result, b)
        case _ => throw typeError
      }
      if (result == null) {
        if (mode == "sum") throw myError("sum over empty")
        else Predicate.empty(1)
      }
      else {
        if (mode == "mean") result = result * (1.0/pred.size)
        else if (mode == "maxNegate") result = result * (-1.0)
        value2pred(result)
      }
    case _ => throw typeError
  })
}

trait CompareSupPredicate extends InfPredicate

// Denotation: {(s,besta,k,f) : s is a set of (a,n) pairs and besta is an a that occurs with the k-th largest/smallest value of n}
// For example: argmax contains ({(a1,3),(a2,9)}, a2)
// if num, then n is computed by counting distinct elements.
case class SupPredicate(name:String, num:Boolean, max:Boolean, ordPred:Predicate=null, fracPred:Predicate=null) extends CompareSupPredicate {
  val abs = Domains.predWithDomains(Domains.typedDomains,
    if (num) Domains.numDomains else Domains.typedDomains,
    {(at,xt) => Entity(SingleSetDomain(Entity(at,xt)), at, OrdinalDomain, FracDomain)})

  override def isArgRequired = List(true, true, false, false)

  def sort(entities:EntityIterable) : Array[Entity] = {
    // Sort entities and pick out the elements required by the rank predicate
    entities.toList.sortWith {
      case (Entity(_, a:NumValue), Entity(_, b:NumValue)) => if (max) a > b else a < b
      case _ => throw Utils.impossible
    }.toArray
  }

  def contains(e:Entity) = impossibleOperation[Boolean]

  def getDegrees(entities:EntityIterable) : EntityIterable = {
    if (num) return entities

    // entities is a set of (a,b) pairs; return set of (a,n) pairs, where n is the number of distinct b's that a occurs with.
    val counts = new HashMap[Value,Int]
    entities.foreach {
      case Entity(a,b) => counts(a) = counts.getOrElseUpdate(a, 0) + 1
      case _ => throw Utils.impossible
    }
    counts.map { case (a,n) => Entity(a, NumValue(CountDomain, n)) }
  }

  override def doSelect(selJ:List[Int], selE:Entity) = {
    if (selJ == List(0)) {
      new SimpleOperation(selE match {
        case (inVal @ SetValue(pred)) :: Nil =>
          val op = pred.enumerate
          if (!op.possible) throw new InterpretException("sup failed because can't enumerate")
          if (ordPred == null && fracPred == null) { // Default: 1 best
            val resultPred = SupPredicate.computeSup(max, getDegrees(op.perform))
            ProdPredicate.create(value2pred(inVal),
              ProdPredicate.create(resultPred,
                SingletonPredicate(Entity(NumValue(OrdinalDomain, 1),
                                          NumValue(FracDomain, 1.0*resultPred.size/pred.size)))))
          }
          else {
            val sortedEntities = sort(getDegrees(op.perform))
            val resultPred = new ExplicitPredicate(4)
            def check(i:Int) = {
              val kVal = NumValue(OrdinalDomain, i+1)
              val fVal = NumValue(FracDomain, 1.0*(i+1)/sortedEntities.size)
              if ((ordPred == null || ordPred.contains(Entity(kVal)).perform) &&
                  (fracPred == null || fracPred.contains(Entity(fVal)).perform))
                resultPred += Entity(inVal, sortedEntities(i)(0), kVal, fVal)
            }
            sortedEntities.zipWithIndex.foreach { case (e, i) =>
              var j : Int = i // Add all the elements tied with position i as well
              while (j >= 0 && sortedEntities(j)(1) == sortedEntities(i)(1)) { // Add tied before
                check(j)
                j -= 1
              }
              j = i+1
              while (j < sortedEntities.size && sortedEntities(j)(1) == sortedEntities(i)(1)) { // Add tied after
                check(j)
                j += 1
              }
            }
            resultPred
          }
        case _ => throw Utils.fails("Unexpected: %s", selE)
      })
    }
    else
      super.doSelect(selJ, selE)
  }

  override def join(pred:Predicate, i:Int) = i match {
    case 2 => ConstOperation(SupPredicate(name, num, max, ordPred=pred, fracPred))
    case 3 => ConstOperation(SupPredicate(name, num, max, ordPred, fracPred=pred))
    case _ => super.join(pred, i)
  }
}
object SupPredicate {
  def computeSup(max:Boolean, entities:EntityIterable) = {
    var besta = new ExplicitPredicate(1)
    var bestn = 0.0
    entities.foreach {
      case a :: NumValue(_, n) :: Nil => // Assume measurement values are of same type
        if (besta.size == 0) {
          besta += Entity(a)
          bestn = n
        }
        else if (n == bestn)
          besta += Entity(a)
        else if (if (max) n > bestn else n < bestn) { // Better
          //besta.clear; besta += Entity(a)
          besta = new ExplicitPredicate(1)
          besta += Entity(a)
          bestn = n
        }
      case _ => throw InterpretException("computeSup: type error")
    }
    besta
  }
}

// FUTURE: differential, scalar (2 times more or 500 feet taller)
// Denotation: { (deg,head,base) : deg(head) >= deg(base)|base }; e.g., head is more expensive than base
case class ComparePredicate(name:String, max:Boolean, strict:Boolean) extends CompareSupPredicate {
  override def hasInverse = false

  val abs = Domains.predWithDomains(Domains.typedDomains, Domains.numDomains, {(at,nt) => Entity(SingleSetDomain(Entity(at,nt)), at, at)}) ++
            Domains.predWithDomains(Domains.typedDomains, Domains.numDomains, {(at,nt) => Entity(SingleSetDomain(Entity(at,nt)), at, nt)})

  def getDegree(pred:Predicate, o:Value) : Option[NumValue] = o match {
    case d:NumValue => Some(d)
    case _ =>
      val op = pred.select(List(0), List(o))
      if (!op.possible) throw myError("Can't retrieve from degree predicate")
      val op2 = op.perform.enumerate
      if (!op2.possible) throw myError("Can't enumerate results of degree predicate")
      val entities = op2.perform
      if (entities.size != 1) return None // Can't get meaningful result if more than one degree (ignore sandwichology problem)
      Some(entities.head(1).asInstanceOf[NumValue])
  }

  def compare(x:NumValue, y:NumValue) : Boolean = {
    if (x.abs != y.abs) throw typeError
    if (x.value >= 0 && y.value >= 0) {
      if (max) { if (strict) x > y else x >= y }
      else     { if (strict) x < y else x <= y }
    }
    else {
      ComparePredicate(name, !max, strict).compare(x.absoluteValue, y.absoluteValue)
      // HACK: we assume that all values are positive by default (e.g., elevation, length, time);
      // When we use the negate predicate), think of it as reversing the ordering on that value;
      // changing its sign is just a temporary solution;
      // This difference is revealed with examples like "smaller than 50 feet", where we end up comparing
      // -30 (due to smaller) and 50 with max=true, it should be testing compare(30, 50, max=false)
    }
  }

  override def doSelect(selJ:List[Int], selE:Entity) = {
    if (selJ == List(2,0)) new LinearOperation(10000, { // True cost: depends on the size of the predicate (avoid it if possible); HACK
      selE match {
        case Entity(base,SetValue(pred)) =>
          val result = new ExplicitPredicate(1)
          getDegree(pred, base) match {
            case None => Predicate.empty(3)
            case Some(baseDeg) =>
              pred.enumerate.perform.foreach { 
                case Entity(head, headDeg:NumValue) => if (compare(headDeg, baseDeg)) result += Entity(head)
                case _ => throw typeError
              }
              ProdPredicate.create(value2pred(SetValue(pred)), ProdPredicate.create(result, value2pred(base)))
          }
        case _ => throw typeError
      }
    })
    else
      super.doSelect(selJ, selE)
  }

  def contains(e:Entity) = new SimpleOperation(e match {
    case Entity(SetValue(pred),head,base) =>
      (getDegree(pred, head), getDegree(pred, base)) match {
        case (Some(x), Some(y)) => compare(x,y)
        case _ => false
      }
    case _ => throw typeError
  })
}

object StandardLibrary {
  def getPredicate(stdName:String, name:String) = stdName match {
    case "false1" => new ExplicitPredicate(1).rename(name)

    case "add" => new ArithPredicate(name, '+')
    case "sub" => new ArithPredicate(name, '-')
    case "mul" => new ArithPredicate(name, '*')
    case "div" => new ArithPredicate(name, '/')

    case "lessThan" => new NumRelPredicate(name, greater=false, strict=true)
    case "moreThan" => new NumRelPredicate(name, greater=true, strict=true)
    case "lessThanEq" => new NumRelPredicate(name, greater=false, strict=false)
    case "moreThanEq" => new NumRelPredicate(name, greater=true, strict=false)
    case "between" => new BetweenPredicate(name, strict=false)

    case "countTopRanked" => new CountTopRankedPredicate(name)
    case "fracTopRanked" => new FracTopRankedPredicate(name)

    case "argminCount" => new SupPredicate(name, num=false, max=false)
    case "argmaxCount" => new SupPredicate(name, num=false, max=true)
    case "argmin" =>      new SupPredicate(name, num=true,  max=false)
    case "argmax" =>      new SupPredicate(name, num=true,  max=true) 
    case "min" => new AggregatePredicate(name, "min")
    case "max" => new AggregatePredicate(name, "max")
    case "maxNegate" => new AggregatePredicate(name, "maxNegate")
    case "mean" => new AggregatePredicate(name, "mean")
    case "sum" => new AggregatePredicate(name, "sum")

    case "affirm" => new ScalePredicate(name, +1)
    case "negate" => new ScalePredicate(name, -1)
    case "count" => new CountPredicate(name, negate=false)
    case "negCount" => new CountPredicate(name, negate=true)
    case "union" => SetUnionPredicate(name, 2)

    case "not" => new FracQuantPredicate(name, SingletonPredicate(Entity(NumValue(FracDomain, 0.0))))
    case "every" => new FracQuantPredicate(name, SingletonPredicate(Entity(NumValue(FracDomain, 1.0))))
    case "most" => new FracQuantPredicate(name, NumRangePredicate(FracDomain, 0.5, false, 1, true))
    case "fracQuant" => new FracQuantPredicate(name)
    case "countQuant" => new CountQuantPredicate(name)

    case "compareLess" => new ComparePredicate(name, max=false, strict=true)
    case "compareMore" => new ComparePredicate(name, max=true, strict=true)
    case "compareLessEq" => new ComparePredicate(name, max=true, strict=false)
    case "compareMoreEq" => new ComparePredicate(name, max=false, strict=false)

    case "lowercase" => new LowercasePredicate(name)
    case "member" => new MemberPredicate(name)
    case "equals" => new EqualPredicate(name)

    case "intRange" => new IntRangePredicate(name)
    case "rand1" => new RandPredicate(name, 1)
    case "rand2" => new RandPredicate(name, 2)
    case "rand3" => new RandPredicate(name, 3)
    case "randInt1" => new RandIntPredicate(name, 1)
    case "randInt2" => new RandIntPredicate(name, 2)
    case "randInt3" => new RandIntPredicate(name, 3)
    case "num2str" => new NumToStrPredicate(name)
    case "concat" => new ConcatPredicate(name)
    case "startsWith" => new StartsWithPredicate(name)

    case _ =>
      Domains.units.find(_.name == stdName) match {
        case Some(u) => new UnitConversionPredicate(name, u)
        case None => throw Utils.fails("Unknown standard predicate name: %s", stdName)
      }
  }
}
