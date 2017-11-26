package dcs

import fig.basic.Indexer
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import tea.Utils
import EntityTypes.Entity

/*
A support is used only for evaluating Datalog rules.
Contains an abstract predicate (like a type) and a concrete predicate (which
contains the actual set of entities).
The labels are the variable names.
*/

class Support(val abs:Predicate, val con:Predicate, val labels:List[String]) {
  if (abs.arity != labels.size || con.arity != labels.size)
    throw Utils.fails("Mis-match: %s has arity %s, %s has arity %s, labels %s", abs.render, abs.arity, con.render, con.arity, labels.mkString(","))
  if (!(con.abs isSubsetOf abs))
    throw Utils.fails("pred %s: %s is not subtype of type %s", con, con.abs.render, abs.render)

  def arity = labels.size

  def proj(sublabels:List[String]) = {
    if (sublabels == labels) this // Optimization
    else {
      val indices = MyUtils.indexOf(labels, sublabels)
      new Support(abs.proj(indices), con.proj(indices), sublabels)
    }
  }

  override def toString = Utils.fmts("%s[%s]:%s%s", con.render, labels.mkString(","), con.abs.render,
    if (con.abs != abs) "<"+abs.render else "")

  def fullString = toString+"="+con.render

  // Cost of the operation is the number of elements produced
  def join(that:Support) : Operation[Support] = {
    if (this.arity == 0) return new ExplicitOperation[Support](1, that.con.enumerate.outCost, that)
    if (that.arity == 0) return new ExplicitOperation[Support](1, this.con.enumerate.outCost, this)

    val joinLabels = this.labels.filter(that.labels.contains) // Labels we have in common
    val indices1 = MyUtils.indexOf(this.labels, joinLabels)
    val indices2 = MyUtils.indexOf(that.labels, joinLabels)

    val resultAbs = PredicateOps.join(this.abs, that.abs, indices1, indices2, JoinMode.share, false).perform
    if (resultAbs.size == 0)
      throw InterpretException(Utils.fmts("Join failed syntactically: %s:%s[%s] and %s:%s[%s] => %s",
        this.con.render, this.abs.render, indices1.mkString(","),
        that.con.render, that.abs.render, indices2.mkString(","),
        resultAbs.render))
    val resultLabels = this.labels ++ that.labels.filter(!joinLabels.contains(_)) // Join labels

    PredicateOps.join(this.con, that.con, indices1, indices2, JoinMode.share, true).extendConst{resultCon =>
      new Support(resultAbs, resultCon, resultLabels)
    }
  }
}
object Support {
  val init = new Support(Predicate.nil, Predicate.nil, Nil)
}
