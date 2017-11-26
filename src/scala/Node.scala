package dcs

import scala.collection.mutable.ArrayBuffer
import tea.Utils

/*
A Node represents a DCS tree.
Note that a Node is immutable, so that we can easiy represent a forest of DCS trees by sharing
subtree Nodes.
*/

// Important: don't use a case class
class Node(val predicates:List[Predicate], val edges:List[Edge], val absDen:Denotation) extends Ordered[Node] with Renderable {
  require (!absDen.isCon, absDen.render)
  require (absDen.canonical, absDen.render)
  require (predicates.size <= 1)
  override def toString = predString

  def arity = absDen.arity
  def width = absDen.width
  def numMarkers = absDen.numMarkers
  def render = Renderer.renderTree(toStrTree)

  def pred = predicates match {
    case pred :: Nil => pred
    case _ => throw Utils.fails("Wanted exactly one predicate, but have %s", predicates)
  }

  var conDen : Denotation = null

  // For parsing
  var predSpan : Span = null // The span corresponding to pred (if it exists)
  var ancSpan : Span = null // The span corresonding to this subtree (excluding nulls)
  var span : Span = null // The span corresponding to this subtree (including nulls); this is never null

  def spanOf(attachLeft:Boolean) = if (attachLeft) span._1 else span._2

  // Possibilities:
  //   - anchor: predicates exists, predSpan exists
  //   - implicit: predicates exists, predSpan does not exist
  //   - empty: neither exists
  def hasImplicitPred = predicates != Nil && predSpan == null
  def hasAnchoredPred = predicates != Nil && predSpan != null

  var score : Double = Double.NaN
  var maxBeamPos : Int = edges.foldLeft(0) {case (m,Edge(r,c)) => m max c.maxBeamPos} // How big we have to keep our beams to build this guy

  def isLeftChild(c:Node) = { require (hasAnchoredPred); c.span._2 <= predSpan._1 }
  def isRightChild(c:Node) = { require (hasAnchoredPred); c.span._1 >= predSpan._2 }

  def nullSpan(c:Node) : Option[Span] = {
    val i = edges.indexWhere(_.c == c)
    require (i != -1)
    if (!hasAnchoredPred) None
    else if (isLeftChild(c)) {
      if (c.ancSpan == null) None
      else {
        val pos = {if (i+1 < edges.size && edges(i+1).c.ancSpan != null) edges(i+1).c.ancSpan._1 min predSpan._1 else predSpan._1}
        Some(Span(c.ancSpan._2, pos))
      }
    }
    else if (isRightChild(c)) {
      if (c.ancSpan == null) None
      else {
        val pos = {if (i-1 >= 0 && edges(i-1).c.ancSpan != null) edges(i-1).c.ancSpan._2 max predSpan._2 else predSpan._2}
        Some(Span(pos, c.ancSpan._1))
      }
    }
    else throw Utils.fails("Not possibly a child")
  }
  
  def hasLeftChildren = predSpan != null && edges.size > 0 && predSpan._1 > edges.head.c.span._1

  def setWithChildrenScores = {
    score = edges.foldLeft(0.0) { case (sum,Edge(r,c)) => sum + c.score }
    Utils.assertValid(score)
  }

  def computeConDen : Denotation = { // Parallels computeAbsDen
    if (conDen != null) return conDen // Cached
    conDen = Denotation.conInit
    predicates.foreach { pred => conDen = conDen safeMeet pred.conDen }
    Node.reorderEdges(edges).foreach { case Edge(r, c) => conDen = conDen.safeApplyRel(r, c.computeConDen) }
    assertDenConsistency
    conDen
  }

  def assertDenConsistency = {
    if (!(computeConDen isConcretizationOf absDen))
      Utils.errors("%s has denotation %s, which is not concretization of %s", render, conDen.render, absDen.render)
  }

  def compare(that:Node) = {
    val d = that.score - this.score
    if (d < 0) -1
    else if (d > 0) +1
    else 0
  }

  def foreachNodePreOrder(f:Node=>Any) : Unit = {
    f(this)
    edges.foreach { e => e.c.foreachNodePreOrder(f) }
  }

  def applyRel(r:Rel, c:Node, attachLeft:Boolean) : List[Node] = {
    lazy val prefix = "Node.applyRel: %s %s%s %s".format(this.render, if (attachLeft) "<" else ">", r, c.render)
    
    def invalid = {
      if (MO.verbose >= 6) Utils.dbgs("%s => invalid", prefix)
      Nil
    }

    val newPredicates = predicates
    val e = Edge(r, c)
    val newEdges = {
      if (attachLeft) e :: edges
      else edges ++ List(e)
    }

    val newAbsDen = Node.computeAbsDen(newPredicates, newEdges)
    if (newAbsDen.isErrorOrEmpty) return invalid

    // Restriction
    if (c.edges.exists(_.r == ExtractRel) && !r.isInstanceOf[ExecuteRel]) return invalid // Extraction needs to be close

    // Prune away bad tree structures (this is separate from ensuring abstract denotations are valid).
    if (!c.argsValid(r)) return invalid
    r match {
      case JoinRel(j1,j2) =>
        // If arity > 1, parent must not already have edge with j1 coming out (there must only be one of argument j1 to the parent function)
        if (arity > 1 && edges.exists(e => e.r match { case JoinRel(j,_) if j == j1 => true; case _ => false })) return invalid
      case _:MarkerRel =>
        if (edges.exists(e => e.r.isInstanceOf[MarkerRel])) return invalid // Don't allow two of the same marker relations
      case _ =>
    }

    val result = new Node(newPredicates, newEdges, newAbsDen)
    result.predSpan = predSpan
    result.ancSpan = {
      if (ancSpan == null) c.ancSpan
      else if (c.ancSpan == null) ancSpan
      else Span(ancSpan._1 min c.ancSpan._1, ancSpan._2 max c.ancSpan._2)
    }
    result.span = Span(span._1 min c.span._1, span._2 max c.span._2)

    if (MO.verbose >= 6) Utils.dbgs("%s => %s", prefix, result.render)
    result::Nil
  }

  def argsValid(r:Rel) = r match {
    case JoinRel(j1, j2) => parentCanJoinTo(j2)
    case _ => true
  }

  def raise(r:Rel) : List[Node] = {
    // Prune away bad tree structures (this is separate from ensuring abstract denotations are valid).
    if (!argsValid(r)) return Nil
    r match {
      case _:ExecuteRel => if (width == 1 && absDen.auxes(0).r == ExtractRel) return Nil // Disallow useless extraction from root (it's a no-op)
        //Utils.dbgs("FFFFF %s %s => %s", r, this.render, absDen.cacheRaise(r).render)
      case _ =>
    }

    //val newAbsDen = try { absDen.cacheRaise(r) } catch { case e => Utils.dbgs("BAD: %s %s", r, this.render); throw e }
    val newAbsDen = absDen.cacheRaise(r)
    if (newAbsDen.isErrorOrEmpty) Nil
    else {
      val result = new Node(Nil, new Edge(r, this) :: Nil, newAbsDen)
      result.ancSpan = this.ancSpan
      result.span = this.span
      result::Nil
    }
  }

  def copy = {
    val result = new Node(this.predicates, this.edges, this.absDen)
    result.conDen = this.conDen
    result.predSpan = this.predSpan
    result.ancSpan = this.ancSpan
    result.span = this.span
    result
  }

  def predString = Predicate.predString(predicates)
  def spanString = predString+span

  def toStrTree : Any = toStrTree(null)
  def toStrTree(edgeLabel:String) : Any = {
    val buf = new ArrayBuffer[Any]
    if (edgeLabel != null) buf += "["+edgeLabel+"]"
    if (IO.displaySpans) buf += predSpan+"|"+span
    if (IO.displayTypes) buf += absDen.render
    if (IO.displayDens && conDen != null) buf += conDen.render
    if (predicates != Nil) buf += predString // Predicate
    edges.foreach { e => buf += e.c.toStrTree(e.r.toString) } // Edges
    buf.toArray
  }

  // Return whether if we can attach a join relation to read component j out of node v
  //  - Predicate in v must have all its required arguments at least once.
  def parentCanJoinTo(j:Int) = forceParentIndex == -1 || forceParentIndex == j
  // -1: no restriction
  // -2: no matter what, doesn't work
  lazy val forceParentIndex = predicates match {
    case pred :: Nil => computeForceParentIndex(pred)
    case _ => -1 // No predicate there, nothing to check
  }
  def computeForceParentIndex(pred:Predicate) : Int = {
    if (pred.arity == 1) return -1 // No restriction on unary predicates

    val hit = new Array[Int](pred.arity)
    edges.foreach {
      case Edge(JoinRel(j, _), c) => hit(j) += 1; require (hit(j) == 1)
      case Edge(CompareRel, _) => return 0 // For (mountain 0-0 (elevation C argmax)) structures, only allow 0 to come out
      case _ =>
    }
    var j = 0
    var force_j = -2 // Default
    var req = pred.isArgRequired
    while (j < arity) {
      if (req.head && hit(j) == 0) {
        if (force_j != -2) return -2 // Already have a force, so can't satisfy two
        force_j = j // Argument j is required and not present, then we are forcing it
      }
      j += 1
      req = req.tail
    }
    if (!pred.hasInverse && force_j != pred.arity-1) force_j = -2 // If no inverse, then must pull out last argument
    force_j
  }
}

object Node {
  def reorderEdges(edges:List[Edge]) = edges match {
    case (e @ Edge(QuantRel, _)) :: rest => rest ++ List(e) // Handle left quantifier
    case _ => edges
  }
  def computeAbsDen(predicates:List[Predicate], edges:List[Edge]) = {
    var absDen : Denotation = Denotation.absInit
    predicates.foreach { pred => absDen = absDen cacheMeet pred.absDen }
    reorderEdges(edges).foreach { case Edge(r, c) => absDen = absDen.cacheApplyRel(r, c.absDen) }
    absDen
  }

  def create(predicates:List[Predicate], edges:List[Edge]) = {
    val absDen = computeAbsDen(predicates, edges)
    if (absDen.isErrorOrEmpty) throw Utils.fails("Type check failed: %s AND %s", predicates.map(_.absDen.render).mkString(" "), edges.map{e => e.r+":"+e.c.absDen.render}.mkString(" "))
    new Node(predicates, edges, absDen)
  }

  val emptyNode = (0 to 100).map {i =>
    val v = Node.create(Nil, Nil)
    //v.predSpan = Span(i, i)
    v.span = Span(i, i)
    v
  }

  // Return whether if we can attach a join relation to read component j out of node v
  //  - Predicate in v must have all its required arguments at least once.
  /*def parentCanJoinTo(v:Node, j:Int) : Boolean = v.predicates match {
    case pred :: Nil =>
      if (v.edges.exists(_.r == CompareRel)) return true // Allow (mountain 0-0 (elevation C argmax)) structures
      val hit = new Array[Int](pred.arity)
      if (j != -1) hit(j) += 1
      v.edges.foreach {
        case Edge(JoinRel(jj, _), c) =>
          hit(jj) += 1
          if (v.arity > 1 && hit(jj) > 1) return false // Don't allow duplicates
        case _ =>
      }
      //Utils.dbgs("parentCanJoinTo: %s j=%s %s %s", v.render, j, hit.toList, pred.isArgRequired)
      (j == -1 || pred.hasInverse || j == pred.arity-1) &&
      (hit zip pred.isArgRequired).forall { case (n,req) => !req || n > 0 }
    case _ => true // No predicate there, nothing to check
  }*/
}

case class Edge(r:Rel, c:Node)
