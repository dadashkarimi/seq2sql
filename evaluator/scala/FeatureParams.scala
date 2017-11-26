package dcs

import gnu.trove.{TObjectDoubleHashMap,TObjectDoubleProcedure}
import fig.exec.Execution
import fig.record.Record
import tea.Utils
import tea.Utils.{map,foreach,fmt,fmts,fmt1,fmt2,returnFirst,assertValid}
import tea.Utils.{track,begin_track,end_track,logs,dbgs,fails,errors,warnings}
import tea.{TreeStringUtils,Tree,BaseTree,CompositeTree}
import tea.Globals._

object TroveUtils {
  def foreachEntry[T](map:TObjectDoubleHashMap[T], func:(T,Double)=>Any) = {
    map.forEachEntry(new TObjectDoubleProcedure[T] {
      def execute(k:T, v:Double) = { func(k, v); true }
    })
  }
  def foreachSortedEntry[T](map:TObjectDoubleHashMap[T], numTop:Int, threshold:Double, func:(T,Double)=>Any)(implicit m:Manifest[T]) = {
    // Sort by magnitude
    var items = new Array[Utils.PairByReverseSecond[T,Double]](map.size)
    var i = 0
    map.forEachEntry(new TObjectDoubleProcedure[T] {
      def execute(k:T, v:Double) = {
        items(i) = new Utils.PairByReverseSecond(k, math.abs(v))
        i += 1
        true
      }
    })
    Utils.partialSort_!(items, numTop)

    // Sort by value
    items = Utils.map(numTop min map.size, { i:Int => new Utils.PairByReverseSecond(items(i)._1, map.get(items(i)._1)) })
    Utils.sort_!(items)
    foreach(items.size, { i:Int =>
      if (math.abs(items(i)._2) > threshold)
        func(items(i)._1, items(i)._2)
    })
  }

  def incr[T](map:TObjectDoubleHashMap[T], k:T, v:Double) = map.adjustOrPutValue(k, v, v)
}

/*
A parameter vector is just a mapping from Feature (e.g., String) to a weight.
*/
class FeatureParams[Feature](LO:LearnOptions)(implicit m:Manifest[Feature]) {
  // Weights
  private val weights = new TObjectDoubleHashMap[Feature]
  private val sumg2 = new TObjectDoubleHashMap[Feature] // sum of gradient squares (for Duchi's adaptive gradient)

  def numFeatures = weights.size
  def get(f:Feature) = weights.get(f)
  def put(f:Feature, v:Double) = weights.put(f, v)
  def clear = {
    weights.clear
    sumg2.clear
  }

  def incr(f:Feature, v:Double) = {
    if (math.abs(v) > 1e-10) { // Be careful not to divide by zero
      val d = {
        if (LO.useAdaptiveGrad) {
          val s = TroveUtils.incr(sumg2, f, v*v)
          v/math.sqrt(s)
        }
        else
          v
      }
      //logs("Params.incr %s %s", f, d)
      TroveUtils.incr(weights, f, d)
    }
  }
  def foreachFeatureWeight(func:(Feature,Double)=>Any) = {
    TroveUtils.foreachEntry(weights, func)
  }
  def foreachSortedFeatureWeight(numTop:Int, threshold:Double, func:(Feature,Double)=>Any) = {
    TroveUtils.foreachSortedEntry(weights, numTop, threshold, func)
  }

  def repCheck = {
    foreachFeatureWeight { (f:Feature,w:Double) =>
      if (w.isNaN || w.isInfinity)
        throw fails("Feature %s has bad weight: %s", f, w)
    }
  }

  // Take the numTop features with largest magnitude
  // But then print them out sorted
  def output(puts:String=>Any) = {
    // Weights
    val numTop = Integer.MAX_VALUE
    foreachSortedFeatureWeight(numTop, 1e-10, { (f:Feature,w:Double) =>
      puts(fmts("%s\t%s", f, w))
    })
  }
}
