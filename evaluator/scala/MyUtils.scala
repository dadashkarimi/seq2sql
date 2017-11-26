package dcs

import java.io.File
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import tea.Utils
import tea.TreeStringUtils

object MyUtils {
  //// General utilities
  val IntList = Utils.map(20, { i:Int => (0 to i-1).toList })
  val IntList2 = Utils.map(20, { i:Int =>
    Utils.map(10, { n:Int =>
      (i to i+n-1).toList
    })
  })
  def repeat[A](n:Int, x:A) : List[A] = {
    if (n <= 0) Nil
    else x :: repeat(n-1, x)
  }
  def append[A](l:List[A], x:A) = l ++ List(x)
  def indexOf[A](l1:List[A], l2:List[A]) = l2.map { x => // Return indices of l2 in l1
    val i = l1.indexOf(x)
    if (i == -1) throw Utils.fails("indexOf([%s],[%s]) failed: %s not present", l1.mkString(","), l2.mkString(","), x)
    i
  }
  def argmax[A](map:HashMap[A,Double]) : A = {
    if (map.size == 0) throw Utils.fails("Tried to compute argmax of empty map")
    var bestx : A = map.keys.head
    var besty = Double.NegativeInfinity
    map.foreach { case (x,y) =>
      if (y > besty) {
        bestx = x
        besty = y
      }
    }
    bestx
  }
  def isSubsetOf[A](set1:HashSet[A], set2:HashSet[A]) = {
    set1.size <= set2.size && set1.forall(set2.contains)
  }
  def normalize[A](map:HashMap[A,Double]) = {
    val sum = map.values.sum
    if (sum == 0) throw Utils.fails("Can't normalize with sum 0")
    map.keys.foreach { a => map(a) /= sum }
  }
  def toHashSet[A](elements:Iterable[A]) = elements match {
    case set:HashSet[A] => set
    case _ =>
      val set = new HashSet[A]
      set ++= elements
      set
  }

  def print[A](x:A) = { Utils.logs("PRINT %s", x); x }

  // path contains some data used to generate A (but might be slow).
  // So cache A (in serialized form).
  def cache[A](path:String, f: =>A) = {
    import fig.basic.IOUtils
    val cachedPath = path+".cached"
    if (path == "/dev/stdin") f
    else if (!(new File(cachedPath)).exists || (new File(path)).lastModified > (new File(cachedPath)).lastModified) { // Out of date or doesn't even exist
      val a = f
      IOUtils.writeObjFileHard(cachedPath, a)
      a
    }
    else {
      IOUtils.readObjFileHard(cachedPath).asInstanceOf[A]
    }
  }

  def findMap[A,B](l:List[A], f:A=>Option[B]) : Option[B] = l match {
    case Nil => None
    case x :: xs => f(x) match {
      case None => findMap(xs, f)
      case yOpt => yOpt
    }
  }

  //def setNull[A <: AnyRef](l:List[A], i:Int) : List[A] = l.zipWithIndex.map{case (x,ii) => if (i == ii) null else x} // Return l with l(i) = null
  def set[A](l:List[A], i:Int, x:A) : List[A] = l.zipWithIndex.map{case (y,ii) => if (i == ii) x else y} // Return l with l(i) = null
  def moveFront[A](l:List[A], i:Int) = l(i) :: without(l, i) // Move l(i) to the beginning
  def without[A](l:List[A], i:Int) = l.slice(0,i) ++ l.slice(i+1,l.size) // Return l with l(i) deleted

  // Pull out l(I) and apply f to it and stick it back in I(0)
  def consolidate[A](l:List[A], I:List[Int], f:List[A]=>List[A]) = {
    MyUtils.IntList(l.size).flatMap{i =>
      if (i == I(0)) f(I.map(l))
      else if (I.contains(i)) None
      else Some(l(i))
    }
  }

  // Think of removing elements at indices from a list.
  // Return a list of indices which corresponds to the indices of the mutated list.
  // For example: 0 1 -> 0 0
  def modifiedListIndices(indices:List[Int]) = {
    var state = IntList(indices.max+1)
    indices.map { i =>
      val j = state.indexOf(i)
      state = state.filter(_ != i)
      j
    }
  }

  // Try to compute the value.  Wait timeout, and then call stop and return errorValue.
  // Up to the caller to 
  def performWithTimeout[A <: AnyRef](timeout:Int, computeValue: =>Any, stop: =>Any) : Unit = {
    if (timeout == 0) { computeValue; return }
    var exception : Throwable = null
    val thread = new Thread(new Runnable { def run = {
      try { computeValue } catch { case e => exception = e }
    } })
    thread.start
    //Utils.dbgs("STARTED")
    thread.join(timeout*1000)
    //Utils.dbgs("CHECK")
    if (thread.isAlive) { // Still running!
      stop
      //Utils.dbgs("CALLED STOP")
      thread.join
      Utils.logs("Timed out (%s seconds)", timeout)
    }
    if (exception != null) throw exception
  }
}
object TimedOutException extends Exception

//// For displaying
object TSU extends TreeStringUtils {
  macroChar = 0
}
