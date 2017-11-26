package dcs

import fig.basic.Indexer
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import tea.Utils
import EntityTypes.Entity

/*
A denotation consists of:
 - Main predicate (set of tuples), which is a concatenation of all predicates.
 - An array of auxiliary denotations (stores).
*/
case class Denotation(isCon:Boolean, main:Predicate, arities:List[Int], auxes:List[Aux]) extends Renderable {
  require (main.arity == arities.sum, main.arity + " " + this)
  require (arities.size == auxes.size, this)

  var canonical = false

  def isError = main.isError
  def isEmpty = main.size == 0
  def isErrorOrEmpty = isError || isEmpty

  def arity = if (width == 0) 0 else arities.head
  def absPred(i:Int) = main.proj(J(i::Nil))

  def isMarked = auxes.head != NullAux
  def width = arities.size
  def numMarkers = auxes.count(_ != NullAux)
  def numExtractMarkers = auxes.count(_.r == ExtractRel)

  def isTrue = width == 0 && !isEmpty

  def render = main.render+":"+auxes.map{aux => if (aux == NullAux) "_" else aux.render}.mkString(";")

  def init = Denotation.init(isCon)

  def J(I:List[Int]) = {
    // Compute components J of main corresponding to the columns I
    var j = 0
    val offsets = arities.map{arity => j += arity; j-arity}
    I.flatMap{i => MyUtils.IntList(arities(i)).map(_+offsets(i))}
  }

  def proj(i:Int) : Denotation = proj(List(i))
  def proj(I:List[Int]) = Denotation(isCon, main.proj(J(I)), I.map(arities), I.map(auxes))

  // Consolidate tuples in indices I.
  def consolidate(I:List[Int]) : Denotation = {
    require (I.forall{i => auxes(i) == NullAux}, this.render)
    val newMain = main.proj(J(MyUtils.consolidate(MyUtils.IntList(width), I, {l:List[Int] => l})))
    val newArities = MyUtils.consolidate(arities, I, {l:List[Int] => l.sum::Nil})
    val newAuxes = MyUtils.consolidate(auxes, I, {l:List[Aux] => l.head::Nil})
    Denotation(isCon, newMain, newArities, newAuxes)
  }

  def createSet(pred:Predicate) = if (isCon) SetValue(pred) else SetDomain.create(pred)
  def collect : Denotation = {
    if (width < 1 || width > 2) throw InterpretException("collect: bad width")
    if (auxes.head != NullAux) throw InterpretException("collect: head marked")
    if (width == 1)
      Denotation(isCon, SingletonPredicate(createSet(main)::Nil), 1::arities.tail, auxes) // Optimization
    else {
      require (width == 2)
      if (auxes(1).r == CompareRel) throw InterpretException("collect: bad side relation")
      // Condition on non-initial columns (back) and create a set of the matching first column (front)
      val result = new ExplicitPredicate(main.arity - arity + 1)
      val (frontIndices, backIndices) = MyUtils.IntList(main.arity).partition(_ < arity)
      val backPred = main.proj(backIndices)
      backPred.enumerate.perform.foreach { back_e =>
        result += createSet(main.select(backIndices, back_e).perform.proj(frontIndices)) :: back_e
      }
      // Add the empty sets by looking at the base of the non-initial columns
      if (auxes(1).base.width != 1) throw InterpretException("collect: base not simple")
      val extraPred = auxes(1).base.main
      require (extraPred.arity == arities(1), auxes(1).render)
      val op = extraPred.enumerate
      if (!op.possible) throw InterpretException("collect: can't enumerate")
      op.perform.foreach { back_e =>
        if (!backPred.contains(back_e).perform)
          result += createSet(Predicate.empty(arity)) :: back_e
      }
      Denotation(isCon, result, 1::arities.tail, auxes)
    }
  }

  def projNullAux = proj(MyUtils.IntList(width).filter{i => auxes(i) != NullAux}) // Keep only marked columns
  def projNonInitialNullAux = proj(MyUtils.IntList(width).filter{i => i == 0 || auxes(i) != NullAux}) // Keep only first and marked columns
  def withAux(i:Int, aux:Aux) : Denotation = {
    require (i < width, render)
    //require ((auxes(i) == NullAux) != (aux == NullAux), this.render) // either go from null to non-null or vice-versa
    if ((auxes(i) == NullAux) == (aux == NullAux)) throw InterpretException("missing or duplicate aux")
    Denotation(isCon, main, arities, MyUtils.set(auxes, i, aux))
  }
  def nullifyAux(i:Int) = withAux(i, NullAux)

  // Move the i-th column to the first column
  def moveFront(i:Int) : Denotation = {
    if (i == 0) return this
    val I = MyUtils.moveFront(MyUtils.IntList(width), i)
    val newMain = main.proj(J(I))
    val newArities = MyUtils.moveFront(arities, i)
    val newAuxes = MyUtils.moveFront(auxes, i)
    Denotation(isCon, newMain, newArities, newAuxes)
  }

  def applyFunc(func:Denotation, arg:Denotation) = init.applyRel(JoinRel(0, 1), func.applyRel(JoinRel(0, 0), arg))

  def execute(i:Int) = {
    val result = projNullAux.doExecute(i)
    if (MO.verbose >= 6 && IO.displayAbsCon(isCon))
      Utils.dbgs("Denotation.execute(%s): %s => %s", i, this.render, result.render)
    result
  }
  private def doExecute(i:Int) : Denotation = {
    if (i >= width) throw InterpretException("execute: out of bounds")
    var d = moveFront(i).nullifyAux(0)
    auxes(i) match {
      case Aux(ExtractRel, func, base) => d
      case Aux(CompareRel, func, base) =>
        if (width < 2) throw InterpretException("compare: too thin")
        // (information, object) => (degree, object)
        d = d.arity match {
          case 1 => applyFunc(Denotation.count(isCon), d.collect) // (e.g., river traverses most states)
          case 2 => d.firstAux_proj(1) // (e.g., most populous state)
          case _ => throw InterpretException("compare: bad arity")
        }
        //Utils.dbgs("COMPARE %s %s", d.render, d.width)
        // Nullify the aux, consolidate and apply the function, then put it back
        applyFunc(func, d.nullifyAux(1).consolidate(List(1,0)).collect).withAux(0, d.auxes(1))
      case Aux(QuantRel, func, base) =>
        //Utils.dbgs("QUANT %s => %s", this.render, moveFront(i).render)
        func.applyRel(JoinRel(0, 0), base.collect)
            .applyRel(JoinRel(1, 0), d.collect).projNullAux
      case Aux(AnaphoraRel(copy, ii), func, base) => throw Utils.fails("Not supported now") // FUTURE
      case aux => throw Utils.fails("Unknown aux: %s", aux)
    }
  }

  def firstAux_proj(j:Int) = {
    val J = MyUtils.IntList(main.arity).filter{jj => jj == j || jj >= arity}
    Denotation(isCon, main.proj(J), 1::arities.tail, auxes)
  }

  // Relations that can be applied on top of
  def raise(r:Rel) : Denotation = {
    //Utils.dbgs("RAISE %s %s %s", r, this.render, this)
    r match {
      case CollectRel => collect
      case ExecuteRel(cols) =>
        if (cols == Nil) throw InterpretException("execute nil")
        MyUtils.modifiedListIndices(cols.reverse).foldLeft(this) { case (d,i) => d.execute(i) }
      case JoinRel(0, j) => firstAux_proj(j)
      case _ => throw Utils.fails("Unknown relation: %s", r)
    }
  }
  def safeRaise(r:Rel) = try { raise(r) } catch { case e:InterpretException => Denotation.error(isCon, e.message) }
  def cacheRaise(r:Rel) : Denotation = {
    Denotation.relCache.getOrElseUpdate(new DenRelDen(null, r, this), this.safeRaise(r).canonicalize)
  }

  def join(j1:Int, j2:Int, c:Denotation) : Denotation = join(j1::Nil, j2::Nil, c)
  def join(J1:List[Int], J2:List[Int], c:Denotation) : Denotation = {
    if (isTrue && J1 == List(0) && J2.size == 1) // Special case: unary projection
      return c.firstAux_proj(J2.head)

    if (J1.size != J2.size) throw InterpretException("join: different number of components")
    if (!(J1.forall(_ < arity) && J2.forall(_ < c.arity))) throw InterpretException("join: out of bounds")

    if (c.width == 1 && !c.isMarked) { // Optimization: marginalize out c
      val newMain = PredicateOps.join(main, c.main, J1, J2, JoinMode.firstOnly, isCon).perform
      val newArities = arities ++ c.arities.tail
      val newAuxes = auxes ++ c.auxes.tail
      Denotation(isCon, newMain, newArities, newAuxes)
    }
    else {
      val newMain = PredicateOps.join(main, c.main, J1, J2, JoinMode.disjoint, isCon).perform
      val newArities = arities ++ c.arities
      val newAuxes = auxes ++ c.auxes
      Denotation(isCon, newMain, newArities, newAuxes).projNonInitialNullAux
    }
  }

  def applyRel(r:Rel, c:Denotation) : Denotation = /*Utils.track("applyRel: %s %s %s", this.render, r, c.render)*/ {
    require (isCon == c.isCon, List(render, isCon, c.render, c.isCon))

    // Restriction on markers
    if (numMarkers + {if (r.isInstanceOf[MarkerRel]) 1 else 0} + c.numMarkers > 2) throw InterpretException("too many markers")
    if (numExtractMarkers + {if (r == ExtractRel) 1 else 0} + c.numExtractMarkers > 1) throw InterpretException("too many extracts")

    val result = r match {
      case EqualRel => {
        if (isTrue) c
        else join(MyUtils.IntList(arity), MyUtils.IntList(c.arity), c)
      }
      case JoinRel(j1,j2) => join(j1::Nil, j2::Nil, c)
      case r:MarkerRel =>
        if (width == 0) throw InterpretException("mark: too thin")
        withAux(0, Aux(r, c, this))
      case r => applyRel(EqualRel, c.raise(r))
    }

    if (!isTrue && MO.verbose >= 6 && IO.displayAbsCon(isCon))
      Utils.dbgs("Denotation.applyRel: %s %s %s => %s", this.render, r, c.render, result.render)
    result
  }
  def safeApplyRel(r:Rel, c:Denotation) : Denotation = {
    if (isError) return this
    if (c.isError) return c
    try { applyRel(r, c) }
    catch { case e:InterpretException => Denotation.error(isCon, e.message) }
  }

  def cacheApplyRel(r:Rel, c:Denotation) : Denotation = {
    Denotation.relCache.getOrElseUpdate(new DenRelDen(this, r, c), this.safeApplyRel(r, c).canonicalize)
  }

  def isConcretizationOf(absDen:Denotation) : Boolean = {
    require (isCon && !absDen.isCon)
    //Utils.dbgs("%s => %s | %s | %s | %s", main.render, main.abs.render, main.abs.enumerate.perform, absDen.main.enumerate.perform, main.abs isSubsetOf absDen.main)
    isError || {
      (main.abs absIsSubsetOf absDen.main) && width == absDen.width && (auxes zip absDen.auxes).forall { case (conAux,absAux) =>
        if (conAux == NullAux || absAux == NullAux) conAux == absAux
        else conAux.r == absAux.r && (conAux.base isConcretizationOf absAux.base) && (conAux.func isConcretizationOf absAux.func)
      }
    }
  }

  def meet(that:Denotation) = applyRel(EqualRel, that)
  def cacheMeet(that:Denotation) = cacheApplyRel(EqualRel, that)
  def safeMeet(that:Denotation) = safeApplyRel(EqualRel, that)

  def canonicalize = {
    require (!isCon)
    Denotation.canonicalDens.getOrElseUpdate(this, {
      //Utils.dbgs("CANONICAL %s: %s", Denotation.canonicalDens.size, this.render)
      canonical = true; this
    })
  }
}

case class Aux(r:MarkerRel, func:Denotation, base:Denotation) extends Renderable {
  def render = r+"["+func.render+" "+base.render+"]"
}
object NullAux extends Aux(null, null, null)

// For hashing, where d1 and d2 (assumed to be canonicalized) are hashed using their identity hashcodes.
class DenRelDen(val d1:Denotation, val r:Rel, val d2:Denotation) {
  //require (d1.canonical && d2.canonical)
  //require (!d1.isCon && !d2.isCon)
  override def equals(that:Any) = that match {
    case that:DenRelDen => (this.d1 eq that.d1) && (this.r == that.r) && (this.d2 eq that.d2)
    case _ => false
  }
  override def hashCode = (System.identityHashCode(d1) * 37 + r.hashCode) * 41 + System.identityHashCode(d2)
}

object Denotation {
  val canonicalDens = new HashMap[Denotation,Denotation]

  val absInit = Denotation(false, Predicate.nil, Nil, Nil).canonicalize
  val conInit = Denotation(true, Predicate.nil, Nil, Nil)
  def init(isCon:Boolean) = if (isCon) conInit else absInit

  def error(isCon:Boolean, msg:String) = Predicate.error(msg).den(isCon)
  //def error(isCon:Boolean, msg:String) = { Utils.dbgs("ERR %s", msg); Predicate.error(msg).den(isCon) }
  def count(isCon:Boolean) = Predicate.count.den(isCon)

  val relCache = new HashMap[DenRelDen, Denotation]
}
