package dcs

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import fig.basic.Indexer
import tea.Utils
import LearnTypes._
import PhraseTypes._
import EntityTypes._

/*
The core of the DCS system, which contains
 - The construction mechanism: given a sentence, build a set of DCS trees.
 - Definition of the features on these trees, which allows us to score the trees.
*/

// Perform inference/prediction prediction.
trait InferState {
  def learner : MyLearner
  def createPoints : Seq[MyPoint] // For discriminative learning

  // These are some statistics that we use for evaluation.
  def oracleCorrect : Boolean
  def predCorrect : Boolean // Did we get this example correct?
  def numCorrectCandidates : Int
  def numCandidates : Int
  def numErrorCandidates : Int

  def impossible = !oracleCorrect && numCandidates < MO.beamSize // An almost-correct indication that our representation is too weak to even attain the truth (there is still slack)

  def update = if (oracleCorrect) {
    createPoints.foreach(learner.addPoint)
  }
}

// For computing syntactic features
case class PathNode(tag:String, head:Boolean) // Whether the head of the parse tree with tag is node at leaf i
case class Path(nodes:List[PathNode], i:Int) { // A path leading downwards to a leaf at position i
  def ::(node:PathNode) = Path(node::nodes, i)
  def toLeafStr = nodes.map(_.tag).mkString("-")
  def toHeadStr(abs:Boolean) = { // Abstract away the tags?
    val i : Int = nodes.indexWhere(_.head) match { case -1 => nodes.size; case i => i } // Position of head
    if (abs) "*"*i
    else nodes.slice(0, i).map(_.tag).mkString("-") // Up to but not including head
  }
}

case class BaseExampleInferState(ex:BaseExample, learner:MyLearner, ctxtId:String) extends InferState {
  val params = learner.params
  val world = ex.world
  val U = ex.world.U
  val morpher = MorphologicalAnalyzer.theMorpher

  var words = U.beginWord :: U.rewriteSentence(ex.words)
  val N = words.size
  Utils.logs("REWRITE %s => %s", ex.words.mkString(" "), words.mkString(" "))

  // Syntactic parsing features
  class ParseInfo {
    // Take words and strip out some things, but keep track of original positions
    val input = words.zipWithIndex.flatMap { case (w,i) =>
      if (w == U.beginWord) None // Don't include some tokens
      else Some((w,i))
    }
    val parser = SentenceParser.theParser
    val tree = parser.parse(input.map(_._1).mkString(" "))
    //Renderer.logTree(words.mkString(" "), parser.toDepTree(tree).toStrTree)
    if (MO.verbose >= 6 && tree != null)
      Renderer.logTree(null, tree.toStrTree)

    var tags = if (tree == null) null else U.beginWord :: tree.tags // part of speech tags

    // If user specifies POS tags use them
    if (tags != null) {
      val newPairs = (words zip tags).map { case (w,t) =>
        w.split("_") match {
          case Array(ww,tt) => (ww,tt) // If word is "<word>_<tag>", then use <tag> as POS tag, replacing what the parser/tagger gave
          case _ => (w,t)
        }
      }
      words = newPairs.map(_._1)
      tags = newPairs.map(_._2)
    }

    // HACK: comparatives: "more populous than Texas" => "populous more than Texas"
    // Need to replace this with a general solution to handle non-projectivity.
    // Should have the recursive construction be built on top of a lattice rather than a linear sequence.
    // This way, variants arising from local re-ordering can be added as additional hypotheses.
    if (tags != null && MO.hackComparatives) {
      def hackComparatives(pairs:List[(String,String)]) : List[(String,String)] = pairs match {
        case Nil => Nil
        case ("more",t1)::(w,t)::("than",t2)::rest => (w,t)::("more",t1)::("than",t2)::hackComparatives(rest)
        case x :: rest => x :: hackComparatives(rest)
      }
      val newPairs = hackComparatives(words zip tags)
      words = newPairs.map(_._1)
      tags = newPairs.map(_._2)
    }

    // For each pair of positions (real word positions)
    val paths = Utils.map(N, { i:Int => Utils.map(N, { j:Int =>
      null : (Path,Path)
    }) })
    def computePathsFromLeaves(tree:ParseTree) : List[Path] = {
      def isHead(p:Path) = tree.head.span._1 == p.i // Does this path extend to the current node as the head?
      if (tree.span.size == 1)
        Path(PathNode(tree.tag, true)::Nil, tree.span._1) :: Nil
      else {
        val childPaths = tree.children.map(computePathsFromLeaves).map { paths =>
          paths.map { p => PathNode(tree.tag, tree.head.span._1 == p.i)::p } // Extend each path to include the current node
        }
        childPaths.zipWithIndex.foreach { case (paths1,z1) =>
          childPaths.zipWithIndex.foreach { case (paths2,z2) =>
            if (z1 != z2) { // The paths must be coming from different children
              paths1.foreach { p1:Path =>
                paths2.foreach { p2:Path =>
                  paths(input(p1.i)._2)(input(p2.i)._2) = (p1, p2) // Convert parse input positions to real word positions
                }
              }
            }
          }
        }
        childPaths.flatten
      }
    }
    if (tree != null) computePathsFromLeaves(tree)
  }
  val parseInfo = if (MO.useSyntax) new ParseInfo else null
  val tags = if (parseInfo != null && parseInfo.tags != null) parseInfo.tags else null
  val tree = if (parseInfo != null && parseInfo.tree != null) parseInfo.tree else null

  // Predicates that we can insert at whim
  def key2preds(key:String) = U.getPredHandles(key::Nil).map(_._2).map(handle2pred).toList
  val implicitBridgePredicates = key2preds("-BRIDGE-")
  val argmaxPred = handle2pred(PredicateName(U.toPredName("argmax", 4)))
  val memberPred = handle2pred(PredicateName(U.toPredName("member", 2)))

  // Whether name(v). is a fact in the knowledge base.
  def predContains(name:String, v:Double) = handle2pred(PredicateName(U.toPredName(name, 1))).enumerate.perform.toList.contains(Entity(NumValue(CountDomain, v)))
  val extAction = predContains("extAction", 1)
  val extUnion = predContains("extUnion", 1)
  //Utils.dbgs("extAction = %s, extUnion = %s", extAction, extUnion)

  def getPredicates(phrase:Phrase, tokenTag:String) = {
    val preds = U.getPredHandles(phrase).map(_._2).map(handle2pred) ++ world.getPredicatesOfPhrase(phrase)
    def posPreds = { // Back off to tags if nothing matches for the phrase
      val tags = (tokenTag :: U.getPosTags(phrase.mkString(" "))).distinct
      tags.flatMap{x => key2preds(":"+x)}.distinct
    }
    val usePosPreds = {
      if (MO.usePosOnlyIfNoMatches) preds.size == 0 // Very stingy about using POS preds
      else if (MO.usePosOnlyIfNoPhraseMatches) !(U.origPhraseMap.contains(phrase) || U.learnedPhraseMap.contains(phrase)) // More lenient
      else true
    }
    preds ++ {if (usePosPreds) posPreds else Nil}
  }
  def handle2pred(handle:PredicateHandle) = handle match {
    case PredicateName(name) =>
      val pred = world.getPredicate(name)
      if (pred == null) throw Utils.fails("Unknown predicate: %s", name) 
      pred
    case PredicateConstant(pred) => pred
  }

  // For each position in the sentence, a list of transitions out to (predicate, end position)
  val usedPredicates = new HashSet[Predicate]
  val endTransitions = Utils.map(N+1, { j:Int => new ArrayBuffer[(Int,Predicate)] })
  val transitions : Array[List[(Predicate,Int)]] = Utils.map(N, { i:Int => // i -> (predicate, j)
    (i+1 to N).flatMap { j => // Predicate spanning i...j
      val tag = {if (tags == null || j-i != 1) "UNK" else tags(i)}
      val preds = getPredicates(words.slice(i, j), tag)
      val pairs = preds.map((_, j))
      if (words(i) != U.beginWord) pairs.foreach(usedPredicates += _._1)
      preds.foreach { pred => endTransitions(j) += ((i,pred)) }
      pairs
    }.toList
  })
  
  // For beginWord, only keep its predicate (if any) if that predicate doesn't show up elsewhere
  words.zipWithIndex.foreach { case (word, i) => if (word == U.beginWord) {
    transitions(i) = transitions(i).filter { case (pred,j) => !usedPredicates.contains(pred) } // Keep pred if it doesn't exist elsewhere
    usedPredicates ++= transitions(i).map(_._1)
  } }

  def interpret(v:Node) = {
    require (!v.absDen.isErrorOrEmpty, v.render)
    try { v.computeConDen } // Include all arguments of the denotation
    catch { case e =>
      IO.displayTypes = true
      Renderer.logTree("Bad", v.toStrTree)
      throw e
    }
  }

  // List of hypotheses
  class Beam extends ArrayBuffer[Node] {
    var numConstructed = 0
    def add(v:Node, isTop:Boolean=false) : Unit = {
      //Utils.dbgs("ADD %s", v.render)
      numConstructed += 1
      if (isTop) {
        //if (v.arity != 0 && v.arity != 1) return
        if (v.numMarkers > 0) return // Should have closed off
      }

      lazy val resultPred = interpret(v).main
      if (MO.pruneErrDen) {
        if (resultPred.isError) return
        if (isTop && MO.pruneInfDen && resultPred.hasInfiniteSize) return
      }
      if (MO.pruneEmptyDen) {
        if (!isTop && resultPred.size == 0) return // Only disallow emptys at the top
      }
      computeScore(v)
      this += v
    }

    def computeNodeOverlap = {
      var total = 0
      val hit = new HashSet[Node]
      foreach { v =>
        v.foreachNodePreOrder { u =>
          total += 1
          hit += u
        }
      }
      Utils.logs("%s hypotheses: %s nodes (%s distinct)", size, total, hit.size)
    }

    def printDuplicates = {
      val hit = new HashSet[String]
      foreach { v =>
        val s = Renderer.renderTree(v.toStrTree) // Hash
        if (hit(s)) Utils.logs("Duplicate: %s", s)
        hit += s
      }
    }

    def prune(tag:String, extra: =>String, n:Int=MO.beamSize) : Beam = {
      Utils.partialSort_!(this, n)

      // Beam needs to be this to keep these hypotheses
      Utils.foreach(n min size, { i:Int =>
        this(i).maxBeamPos = this(i).maxBeamPos max i
      })

      if (MO.verbose >= {if (tag == "top") 3 else 4}) {
        Utils.track("Hypotheses[%s] %s (keep %s of %s, %s constructed)", tag, extra, n, size, numConstructed) {
          zipWithIndex.foreach { case (v, i) =>
            Renderer.logTree(Utils.fmts("%s%s score=%s: %s : %s",
              i, {if (i<n) " keep" else ""}, v.score, interpret(v).render, v.absDen.render), v.toStrTree)
            v.assertDenConsistency
          }
        }
        printDuplicates
        computeNodeOverlap
      }

      if (n < size) remove(n, size-n)
      this
    }
  }

  def computeScore(v:Node) : Unit = {
    if (!v.score.isNaN) return
    v.edges.foreach { case Edge(r,c) => computeScore(c) }
    v.setWithChildrenScores
    require (!v.score.isNaN)
    foreachLocalFeature(v, { f =>
      v.score += params.get(f)
      require (!v.score.isNaN)
    })
  }

  //////////////////////////////
  // Features

  def F_LexPred(wordsStr:String, predStr:String) = "lexpred:"+wordsStr+":"+predStr

  def absPredStr(predicates:List[Predicate]) = {
    if (predicates == Nil) "."
    else predicates.map {
      case pred:SingletonPredicate => pred.abs.render // Abstract values to their types
      case pred => pred.name // Use the predicate name
    }.mkString(",")
  }

  def dirStr(v:Node, c:Node) = {
    if (!v.hasAnchoredPred) ""
    else if (v.isLeftChild(c)) "<"
    else if (v.isRightChild(c)) ">"
    else { IO.displaySpans = true; throw Utils.fails("%s is neither left or right child of %s", c.render, v.render) }
  }

  def wordsStr(i:Int, j:Int) = words.slice(i,j).mkString("_").toLowerCase

  def sliceWordsOrEmptyWord(i:Int, j:Int) = {
    require (i <= j, i + " " + j)
    if (i == j) List("")
    else words.slice(i, j)
  }

  def foreachLocalFeature(v:Node, f:Feature=>Any) = {
    MO.features.foreach {
      case "predCount" => if (v.predicates != Nil) f("predCount")
      case "pred" => // One predicate
        if (v.predicates != Nil) f("predCount")
        f("pred:"+absPredStr(v.predicates))
      case "pred2" => { // Two predicates
        // For each path down to a node with a predicate or a leaf
        def recurse(v:Node, prefix:String, canStop:Boolean) : Unit = {
          val isStoppingPoint = v.predicates.size > 0 || v.edges.size == 0
          if (isStoppingPoint && canStop)
            f(prefix)
          else {
            v.edges.foreach { case Edge(r, c) =>
              recurse(c, prefix+"_"+dirStr(v,c)+r+":"+absPredStr(c.predicates), true)
            }
          }
        }
        recurse(v, "pred2:"+absPredStr(v.predicates), false)
      }
      case "predarg" => { // Argument structure downward of a predicate (don't include child predicates)
        val paths = new ArrayBuffer[String]
        def recurse(v:Node, prefix:String, canStop:Boolean) : Unit = {
          val isStoppingPoint = v.predicates.size > 0 || v.edges.size == 0
          if (isStoppingPoint && canStop)
            paths += prefix
          else {
            v.edges.foreach { case Edge(r, c) =>
              recurse(c, prefix+dirStr(v,c)+r, true)
            }
          }
        }
        recurse(v, "", false)
        f("predarg:"+absPredStr(v.predicates)+paths.mkString(","))
      }

      case "lexpred" => { // Word and predicate
        if (v.hasAnchoredPred) {
          f(F_LexPred(wordsStr(v.predSpan._1, v.predSpan._2), v.predString))
        }
      }
      case "lexpredmatch" => {
        if (v.hasAnchoredPred) {
          val str = wordsStr(v.predSpan._1, v.predSpan._2)
          v.pred match {
            case SingletonPredicate(x) => f("lexpredconstmatch")
            case pred =>
              val name = pred.name.split("/")(0)
              if (morpher.stem(str) == morpher.stem(name)) f("lexpredmatch")
          }
        }
      }
      case "lexctxtpred" => { // Surrounding word on either side and predicate
        if (v.hasAnchoredPred) {
          f("lexctxtpred:<"+wordsStr(v.predSpan._1-1, v.predSpan._1)+":"+v.predString)
          f("lexctxtpred:>"+wordsStr(v.predSpan._2, v.predSpan._2+1)+":"+v.predString)
        }
      }
      case "skippred" => {
        // For each predicate that overlaps with a null, make a note
        if (v.ancSpan != null) {
          val usedPreds = new ArrayBuffer[Predicate]
          v.foreachNodePreOrder { c => usedPreds ++= c.predicates }
          Utils.foreach(v.span._1, v.ancSpan._1, { i:Int =>
            transitions(i).foreach { case (pred,j) =>
              if (!usedPreds.contains(pred))
                f("skippred:"+absPredStr(pred::Nil)+":"+wordsStr(i,j))
            }
          })
          Utils.foreach(v.ancSpan._2+1, v.span._2+1, { j:Int =>
            endTransitions(j).foreach { case (i,pred) =>
              if (!usedPreds.contains(pred))
                f("skippred:"+absPredStr(pred::Nil)+":"+wordsStr(i,j))
            }
          })
        }
      }
      case "lexnull" => {
        if (v.hasAnchoredPred) {
          v.edges.foreach { case Edge(r,c) => // For each child...
            v.nullSpan(c) match {
              case Some(Span(i, j)) =>
                if (i > j) { IO.displaySpans = true; throw Utils.fails("Bad span %s,%s:\n%s\n%s", i, j, v.render, c.render) }
                val nullWords = sliceWordsOrEmptyWord(i, j) // Get the null (non-anchored) words between this node and that child
                if (c.hasImplicitPred) // c holds the implicit predicate
                  nullWords.foreach { w => f("leximpred:"+dirStr(v,c)+w+":"+absPredStr(c.predicates)) }
                else {
                  nullWords.foreach { w =>
                    f("lexrel:"+dirStr(v,c)+w+":"+r)
                    f("lexpredrel:"+dirStr(v,c)+w+":"+absPredStr(v.predicates)+":"+r)
                  }
                }
              case _ =>
            }
          }
        }
      }

      case "syntax" => {
        // Note we don't define syntactic features for implicit predicates
        if (v.hasAnchoredPred) {
          v.edges.foreach { case Edge(r, c) =>
            if (c.hasAnchoredPred && parseInfo != null) {
              parseInfo.paths(v.predSpan._1)(c.predSpan._1) match {
                case null =>
                case (vp, cp) =>
                  val leaf2Str = vp.toLeafStr+","+cp.toLeafStr
                  val head2Str = vp.toHeadStr(false)+","+cp.toHeadStr(false)
                  val headabs2Str = vp.toHeadStr(true)+","+cp.toHeadStr(true)
                  val pred2Str = absPredStr(v.predicates)+","+absPredStr(c.predicates)
                  f("syntaxleaf2:"+leaf2Str)
                  f("syntaxhead2:"+head2Str)
                  f("syntaxheadabs2:"+headabs2Str)
                  f("syntaxleaf2pred:"+leaf2Str+":"+pred2Str)
                  f("syntaxhead2pred:"+head2Str+":"+pred2Str)
                  f("syntaxheadabs2pred:"+headabs2Str+":"+pred2Str)
              }
            }
          }
        }
      }

      case "size" => {
        val size = v.computeConDen.main.size match {
          case 0 => 0
          case 1 => 1
          case _ => ">1"
        }
        f("size:"+size)
      }
      case f => throw Utils.fails("Unknown feature: %s", f)
    }
  }

  def foreachFeature(v:Node, f:Feature=>Any) : Unit = {
    foreachLocalFeature(v, f)
    v.edges.foreach { e => foreachFeature(e.c, f) }
  }

  def summary(begin:Int=0, end:Int=N) = {
    (transitions.slice(begin, end) zip (begin to end-1)).map { case (list,i) =>
      if (list.size == 0) i+":"+words(i)
      else Utils.fmts("%s:%s=%s", i, words(i), list.map{case (pred,j) => pred.name}.mkString(","))
    }.mkString(" ")
  }

  //////////////////////////////
  // Inference
  val topChart = new Beam
  val allowNulls = !ex.isSimple // If simple, don't make things complex with nulls

  def isSetUnionPred(pred:Predicate) = pred.corePred.isInstanceOf[SetUnionPredicate]
  def isQuantPred(pred:Predicate) = pred.corePred.isInstanceOf[QuantPredicate]
  def isCompareSupPred(pred:Predicate) = pred.corePred.isInstanceOf[CompareSupPredicate]
  def isComparePred(pred:Predicate) = pred.corePred.isInstanceOf[ComparePredicate]
  def isActionPred(pred:Predicate) = isCompareSupPred(pred) || isQuantPred(pred)
  def isConstantPred(pred:Predicate) = pred.size == 1

  def isNumeric(pred:Predicate) = pred.enumerate.perform.exists{case Entity(_:NumDomain) => true; case _ => false}
  //def hasNumeric(pred:Predicate) = pred.enumerate.perform.exists(_.exists(_.isInstanceOf[NumDomain]))
  /*def hasNumeric(pred:Predicate) = {
    if (pred.enumerate.perform.exists(_.exists(_.isInstanceOf[NumDomain]))) Utils.dbgs("hasNumeric: %s %s", pred.render, pred)
    pred.enumerate.perform.exists(_.exists(_.isInstanceOf[NumDomain]))
  }*/
  /*def isObject(pred:Predicate) = { // NEW
    val types = pred.enumerate.perform
    types.size == 1 && types.head.size == 1
  }*/

  def markExtract(v:Node) : List[Node] = {
    if (v.arity != 1) return Nil // Restriction: extract only single tuples
    v.applyRel(ExtractRel, Node.emptyNode(v.span._2), attachLeft=false)
  }

  def createNodeAbove(pred:Predicate, c:Node, attachLeft:Boolean) : Node = {
    val b = Node.create(pred::Nil, Nil) // Bridge node
    b.span = if (attachLeft) Span(c.span._2, c.span._2) else Span(c.span._1, c.span._1) // Put b right next to c
    b
  }

  // Special case: allow bridging from some predicates regardless
  def v_allowBridge(v:Node) = MO.implicitk1 || v.arity == 1 || v.predicates.contains(argmaxPred)
  def c_allowBridge(c:Node) = MO.implicit1k || c.arity == 1 || c.predicates.contains(memberPred)

  //def allowAggregate(v:Node, c:Node) = !v.predicates.exists(isActionPred) // Avoid building (argmax/4 ([0-0] ([++] len/2)))

  // Order: attach right first, then attach left
  def attach(v:Node, c:Node, attachLeft:Boolean, allowBridge:Boolean) : List[Node] = {
    if (!attachLeft && v.hasLeftChildren) return Nil // Can't attach right after attaching left
    if (attachLeft && MO.forceRightBranching) return Nil // Can't attach left at all if completely right-branching

    //Utils.dbgs("ATTACH %s %s:%s %s", v.render, c.render, c.absDen.main.render, allowBridge)

    // Restriction: Superlatives/comparatives and quantifiers should be at leaves, not at non-leaf v
    // Exception: allow numbers so that we can handle second largest, 70% of ..., two largest [note that c could be countTopRanked]
    //if (v.predicates.exists(isActionPred) && !hasNumeric(c.absDen.main) && !isObject(c.absDen.main)) return Nil // Don't think this is necessary
    if (v.predicates.exists(isActionPred) && !isNumeric(c.absDen.main)) return Nil

    // Restriction: Constants can't have children except for quantification (in particular, negation, e.g., "not perl")
    if (v.predicates.exists(isConstantPred) && !c.predicates.exists(isQuantPred)) return Nil

    var result : List[Node] = Nil

    // Insert extra predicates between
    def insertBridge = if (allowBridge && v_allowBridge(v) && c_allowBridge(c)) { // Bridge
      result = result ++ implicitBridgePredicates.flatMap { implicitPred =>
        val b = createNodeAbove(implicitPred, c, attachLeft)

        // Try attaching trolls to bridge b
        val bs : List[Node] = if (!MO.allowTroll) b::Nil else {
          val (left, right) = {
            if (attachLeft) (c.ancSpan._2, v.ancSpan._1)
            else            (v.ancSpan._2, c.ancSpan._1)
          }
          b :: (left to right).flatMap { i =>
            transitions(i).flatMap { case (pred,j) =>
              if (j <= right) {
                val troll = baseNode(pred, i, j)
                attach(b, troll, !attachLeft, allowBridge=false) // Attach troll in opposite direction
              }
              else
                Nil
            }
          }.toList
        }

        //Utils.track("bs") { bs.foreach{b => Utils.dbgs("%s", b.render)} }

        bs.flatMap{b => attach(b, c, attachLeft, allowBridge=false).flatMap(attach(v, _, attachLeft, allowBridge=false))}
      }
    }

    //// Put one or more relations between

    if (c.predicates.exists(isActionPred)) {
      if (c.predicates.exists(isQuantPred))
        result = result ++ v.applyRel(QuantRel, c, attachLeft) // Quantification (e.g., no city)
      else if (c.predicates.exists(isCompareSupPred))
        result = result ++ v.applyRel(CompareRel, c, attachLeft) // Comparison (e.g., most cities)
      else
        throw Utils.impossible
      if (extAction) insertBridge
    }
    else {
      // Join v and c, collecting if necessary
      result = result ++ MyUtils.IntList(v.arity).flatMap{vi => 
        MyUtils.IntList(c.arity).flatMap{ci =>
          //Utils.dbgs("JOIN %s %s-%s %s", v.render, vi, ci, c.render)
          v.applyRel(JoinRel(vi, ci), c, attachLeft)
        } ++
        c.raise(CollectRel).flatMap(v.applyRel(JoinRel(vi, 0), _, attachLeft)) // Aggregate!
        //{if (!allowAggregate(v,c)) Nil else c.raise(CollectRel).flatMap(v.applyRel(JoinRel(vi, 0), _, attachLeft))} // Aggregate!
      }
      insertBridge
      //result.foreach{a => Utils.dbgs("PRE %s %s => %s", v.render, c.render, a.render)}
    }

    // Insert member/2 above union/3 (e.g., cities in Oregon and Washington)
    if (extUnion && !v.predicates.contains(memberPred) && c.predicates.exists(isSetUnionPred)) {
      //Utils.dbgs("INSERT MEMBER: %s %s", v.render, c.render)
      val b = createNodeAbove(memberPred, c, attachLeft)
      result = result ++ attach(b, c, attachLeft, false).flatMap(attach(v, _, attachLeft, allowBridge))
      //result.foreach{a => Utils.dbgs("RESULT %s %s => %s", v.render, c.render, a.render)}
    }

    if (allowBridge) {
      result = result ++ result.flatMap(markExtract) // Extract
      result = result ++ result.flatMap{a => a.raise(ExecuteRel(MyUtils.IntList(a.numMarkers)))} // Execute
    }

    result
  }

  // Order: attach right first, then attach left
  def attach_OLD(v:Node, c:Node, attachLeft:Boolean, allowBridge:Boolean) : List[Node] = {
    if (!attachLeft && v.hasLeftChildren) return Nil // Can't attach right after attaching left
    if (attachLeft && MO.forceRightBranching) return Nil // Can't attach left at all if completely right-branching

    // Restriction: Superlatives and quantifiers should be at leaves, not at non-leaf v
    // Exception: allow numbers so that we can handle second largest, 70% of ...
    if (v.predicates.exists(isActionPred) && !isNumeric(c.absDen.main)) return Nil
    //if (v.predicates.exists(isComparePred)

    // TODO: permit comparatives to attach arguments
    // TODO: non-projective

    // Restriction: Constants can't have children except for quantification (in particular, negation, e.g., "not perl")
    if (v.predicates.exists(isConstantPred) && !c.predicates.exists(isQuantPred)) return Nil

    var result : List[Node] = Nil

    if (c.predicates.exists(isQuantPred))
      result = v.applyRel(QuantRel, c, attachLeft) // Quantification (no city)
    else if (c.predicates.exists(isCompareSupPred)) {
      //Utils.dbgs("COMPARE %s %s", v.render, c.render)
      result = v.applyRel(CompareRel, c, attachLeft) // Comparison (e.g., most cities)
    }
    else {
      // Join v and c, collecting if necessary
      result = MyUtils.IntList(v.arity).flatMap{vi => 
        MyUtils.IntList(c.arity).flatMap{ci =>
          //Utils.dbgs("JOIN %s %s-%s %s", v.render, vi, ci, c.render)
          v.applyRel(JoinRel(vi, ci), c, attachLeft)
        } ++
        c.raise(CollectRel).flatMap(v.applyRel(JoinRel(vi, 0), _, attachLeft))
      }

      // Non-projective join
      // TODO: allow incomplete edges at the bottom
      if (MO.allowNonProjectivity && v.hasAnchoredPred && v.edges.size > 0) {
        if (attachLeft) {
          val e = v.edges.last
          if (e.c.edges.size == 0 && v.isRightChild(e.c)) {
            result ++= attach_OLD(e.c, c, attachLeft, false).map{ d =>
              Node.create(v.predicates, v.edges ++ List(Edge(e.r, d)))
            }
          }
        }
        else {
          //val d = v.edges.head.c
          //if (v.isLeftChild(d)) result ++= attach_OLD(d, c, attachLeft, false)
        }
      }
    
      //result.foreach{a => Utils.dbgs("PRE %s %s => %s", v.render, c.render, a.render)}

      if (allowBridge && (MO.implicitk1 || v.arity == 1) && (MO.implicit1k || c.arity == 1)) { // Bridge
        result = result ++ implicitBridgePredicates.flatMap { implicitPred =>
          val b = Node.create(implicitPred::Nil, Nil) // Bridge node
          b.span = if (attachLeft) Span(c.span._2, c.span._2) else Span(c.span._1, c.span._1) // Put b right next to c

          // Try attaching trolls to bridge b
          val bs : List[Node] = if (!MO.allowTroll) b::Nil else {
            val (left, right) = {
              if (attachLeft) (c.ancSpan._2, v.ancSpan._1)
              else            (v.ancSpan._2, c.ancSpan._1)
            }
            b :: (left to right).flatMap { i =>
              transitions(i).flatMap { case (pred,j) =>
                if (j <= right) {
                  val troll = baseNode(pred, i, j)
                  attach_OLD(b, troll, !attachLeft, allowBridge=false) // Attach troll in opposite direction
                }
                else
                  Nil
              }
            }.toList
          }

          //Utils.track("bs") { bs.foreach{b => Utils.dbgs("%s", b.render)} }

          /*val b = Node.create(implicitPred::Nil, Nil) // Intermediate node
          b.ancSpan = c.ancSpan
          b.span = c.span*/
          /*bs.foreach { b =>
            Utils.dbgs("BRIDGE %s %s %s", v.render, b.render, c.render)
            attach_OLD(b, c, attachLeft, allowBridge=false).flatMap(attach_OLD(v, _, attachLeft, allowBridge=false)).foreach { a =>
              Utils.dbgs("  %s", a.render)
            }
          }*/
          bs.flatMap{b => attach_OLD(b, c, attachLeft, allowBridge=false).flatMap(attach_OLD(v, _, attachLeft, allowBridge=false))}
        }
      }
    }

    if (allowBridge) {
      result = result ++ result.flatMap(markExtract) // Extract
      result = result ++ result.flatMap{a => a.raise(ExecuteRel(MyUtils.IntList(a.numMarkers)))} // Execute
    }

    result
  }

  //// Main inference steps
  Utils.logs("%s", summary())
  if (ex.trueExpr != null) Utils.logs("True: %s => %s", ex.trueExpr, ex.trueAnswer.render)
  // Dynamic program state (represents set of TreeDRSes)
  //   - Span i...j
  // Attach right arguments before attaching left (bottom up)
  val chart = Utils.map(N+1, { i:Int =>
    Utils.map(N+1, { j:Int =>
      null : Beam
    })
  })

  def baseNode(pred:Predicate, i:Int, j:Int) = {
    val node = Node.create(pred::Nil, Nil)
    node.predSpan = Span(i, j)
    node.ancSpan = Span(i, j)
    node.span = Span(i, j)
    node
  }

  var stop = false
  def getBeam(i:Int, j:Int) = {
    val beam = new Beam
    val len = j-i

    // Base case
    transitions(i).foreach {
      case (pred, jj) if j == jj =>
        val node = baseNode(pred, i, j)
        (node :: markExtract(node)).foreach(beam.add(_))
      case _ =>
    }

    // Generate nulls (only generate to the left, and only if v has nothing attached to it).
    if (len >= 2 && (allowNulls || words(i) == U.beginWord)) {
      chart(i+1)(j).foreach { v => // Attach null left child
        if (v.edges.count(_.r != ExtractRel) == 0) { // Can only attach closest to the predicate itself (except ExtractRel)
          val resultv = v.copy
          resultv.span = v.span match { case Span(i,j) => Span(i-1,j) }
          //logTree(fmts("LEFT %s %s", i, j), resultv.toStrTree)
          beam.add(resultv)
        }
      }
    }

    // Recursive case
    Utils.foreach(i+1, j, { k:Int => // Split point k
      chart(i)(k).foreach { v1 =>
        chart(k)(j).foreach { v2 =>
          if (stop) throw TimedOutException
          if (MO.oldAttach) {
            attach_OLD(v1, v2, attachLeft=false, allowBridge=true).foreach(beam.add(_)) // v2 becomes v1's right child
            attach_OLD(v2, v1, attachLeft=true, allowBridge=true).foreach(beam.add(_)) // v1 becomes v2's left child
          }
          else {
            attach(v1, v2, attachLeft=false, allowBridge=true).foreach(beam.add(_)) // v2 becomes v1's right child
            attach(v2, v1, attachLeft=true, allowBridge=true).foreach(beam.add(_)) // v1 becomes v2's left child
          }
        }
      }
    })

    beam
  }

  def computeChart = {
    Utils.foreach(1, N+1, { len:Int =>
      Utils.foreach(0, N-len+1, { i:Int =>
        val j = i+len
        chart(i)(j) = getBeam(i,j).prune(Utils.fmts("%s...%s", i, j), summary(i, j))
      })
    })

    // Can add final right nulls
    if (allowNulls) {
      Utils.foreachReverse(N, 1, { j:Int => // For each ending point j, pad nulls from j to N
        chart(0)(j).foreach { v =>
          if (stop) throw TimedOutException
          val resultv = v.copy
          resultv.span = Span(0, N)
          chart(0)(N).add(resultv)
        }
      })
    }

    // Finish it off
    chart(0)(N).foreach { v =>
      if (stop) throw TimedOutException
      var vs = {
        if (v.arity == 1) v::Nil
        else if (v.predicates.exists(isSetUnionPred)) { // Read out elements of a set
          val b = createNodeAbove(memberPred, v, attachLeft=false)
          if (MO.oldAttach)
            attach_OLD(b, v, attachLeft=false, false).flatMap(_.raise(JoinRel(0,0)))
          else
            attach(b, v, attachLeft=false, false).flatMap(_.raise(JoinRel(0,0)))
        }
        else MyUtils.IntList(v.arity).flatMap{j=>v.raise(JoinRel(0,j))} // Read out one component if necessary
      }
      vs.foreach{v => topChart.add(v, true)}
    }
    topChart.prune("top", "-")
    true
  }
  MyUtils.performWithTimeout(MO.timeout, computeChart, {stop=true})

  val numCandidates = topChart.size

  //Utils.logs("Now computing answers for %s candidates", numCandidates)
  val predAnswers = topChart.map { v => new Answer(interpret(v).main) }.toArray // candidate index -> answer
  val isCorrect = predAnswers.map { ans => // candidate index -> whether that candidate gets the correct answer
    !ans.isError && (!ex.hasTrueAnswer || ans == ex.trueAnswer) // Can't be error and must match true answer if that exists
  }
  val targetFactors = isCorrect.map{b => if (b) 1.0 else 0.0} // candidate index -> extra factor for computing target distribution
  val numCorrectCandidates = isCorrect.count(_ == true)
  val oracleCorrect = numCorrectCandidates > 0
  val numErrorCandidates = predAnswers.count(_.isError)
  val candidateProbs = topChart.map(_.score).toArray
  if (numCandidates > 0)
    Utils.expNormalizeOrFail_!(candidateProbs)

  // Oracle
  val targetProbMass = {
    if (MO.useBayesianAveraging) {
      (candidateProbs zip isCorrect).filter(_._2 == true).map(_._1).sum
    }
    else {
      if (oracleCorrect) candidateProbs(isCorrect.indexWhere(_ == true)) else 0
    }
  }

  // Prediction
  val predAnswerProbs = new HashMap[Answer,Double] // Answer -> predicted probabilities
  val predAnswerIndices = new HashMap[Answer,ArrayBuffer[Int]] // Answer -> candidate indices with that answer
  predAnswers.zipWithIndex.foreach { case (ans, i) => // For each candidate...
    if (ans.isError) // Simple constraint: don't predict anything that's a blatant error
      predAnswerProbs(ans) = 0
    else {
      predAnswerProbs.get(ans) match {
        case Some(_) =>
          if (MO.useBayesianAveraging) // Accumulate probability
            predAnswerProbs(ans) += candidateProbs(i)
        case None =>
          predAnswerProbs(ans) = candidateProbs(i)
      }
    }
    predAnswerIndices.getOrElseUpdate(ans, new ArrayBuffer[Int]) += i
  }
  val predProbMass = if (predAnswerProbs.size == 0) 0 else predAnswerProbs.values.max
  val predAnswer = if (predAnswerProbs.size == 0) null else MyUtils.argmax(predAnswerProbs) // Finally, the thing we've been waiting for, our prediction!
  val predCorrect = predAnswer == ex.trueAnswer
  require (predAnswer == predAnswer, predAnswer.pred)

  // Return first candidate that agrees with the answer
  def findFirst(ans:Answer) : Option[Node] = (topChart zip predAnswers).find(_._2 == ans).flatMap{p=>Some(p._1)}

  //// Output some information
  if (numCandidates == 0)
    Utils.logs("No candidates")
  else {
    val printFeatures = MO.verbose >= 4 || ((!ex.hasTrueAnswer || !predCorrect) && MO.verbose >= 3) // Print if wrong or verbose is high enough
    def logWithInfo(tag:String, i:Int) = {
      val v = topChart(i)
      val extra = {
        if (!ex.hasTrueAnswer) ""
        else if (isCorrect(i)) " [CORRECT]"
        else " [WRONG]"
      }
      Utils.track_printAll("%s: %s, prob=%s%s", tag, Renderer.render(predAnswers(i)), candidateProbs(i), extra) {
        Renderer.logTree(null, v.toStrTree)
        if (printFeatures) {
          foreachFeature(v, { f:Feature =>
            Utils.logs("%s\t%s", f, params.get(f))
          })
        }
      }
    }

    if (MO.verbose >= 2) { // Oracle
      var first = true
      topChart.zipWithIndex.foreach { case (v,i) =>
        if (isCorrect(i) && ex.hasTrueAnswer) {
          if (first || MO.useBayesianAveraging || MO.verbose >= 3)
            logWithInfo("Oracle", i)
          first = false
        }
      }
    }
    if (MO.verbose >= 2) { // Pred
      if (MO.useBayesianAveraging) {
        predAnswers.zipWithIndex.foreach { case (ans,i) =>
          if (ans == predAnswer) logWithInfo("Pred", i)
        }
      }
      else
        logWithInfo("Pred", predAnswers.indexWhere(_ == predAnswer))
    }
    if (MO.verbose >= 2 && !predCorrect) { // Difference
      (findFirst(ex.trueAnswer), findFirst(predAnswer)) match {
        case (Some(truev), Some(predv)) =>
          val counts = new HashMap[Feature,Double]
          foreachFeature(truev, { f:Feature => counts(f) = counts.getOrElse(f,0.0) + 1 })
          foreachFeature(predv, { f:Feature => counts(f) = counts.getOrElse(f,0.0) - 1 })
          val weightedCounts = counts.map { case (f,v) => (f, v*params.get(f)) }
          val diff = weightedCounts.values.sum
          Utils.track("Difference: %s", diff) {
            weightedCounts.toList.sortWith(_._2 > _._2).foreach { case (f,w) =>
              if (w != 0) Utils.logs("%s\t%s * %s", f, counts(f), params.get(f))
            }
          }
        case _ =>
      }
    }

    if (ex.hasTrueAnswer) {
      val (extra, tag) = {
        if (!oracleCorrect) {
          val exItem = TermItem("parse", ListItem(ex.words.map{w => StrItem(w.replaceAll("'", ""))}) :: ex.trueExpr :: Nil)
          val (extra, tag) = {
            if (impossible) (" [impossible]", "IMPOSSIBLE")
            else (" [insufficient beam]", "INSUFFICIENT_BEAM")
          }
          Utils.logs("%s: %s. %% %s", tag, exItem, ex.words.mkString(" "))
          (extra, tag)
        }
        else {
          val minMaxBeamPos = topChart(isCorrect.indexOf(true)).maxBeamPos
          val extra = " [possible with beam size "+(minMaxBeamPos+1)+"]"
          if (predCorrect) (extra, "CORRECT")
          else (extra, "WRONG")
        }
      }
      Utils.logs("CTXT=%s,EX=%s\t%s\t%s\t%s", ctxtId, ex.id, tag, words.mkString(" "), Renderer.render(ex.trueAnswer))
      Utils.logs("Result %s: #words=%s, #preds=%s, %s/%s correct (target:%s, pred:%s)%s",
        ex.id, words.size, usedPredicates.size, numCorrectCandidates, numCandidates, targetProbMass, predProbMass, extra)
    }
  }

  // For learning
  def createPoints = {
    val choices = (topChart zip targetFactors).map { case (v, targetFactor) =>
      val features = new ArrayBuffer[Feature]
      foreachFeature(v, features += _)
      Choice(features, targetFactor)
    }
    new MyPoint(ex, choices) :: Nil
  }
}

case class ConstraintInferState(cons:Constraint, learner:MyLearner, ctxtId:String) extends InferState {
  val id = cons.id
  val ex = cons.ex

  def isPossibleAnswer(ans:Answer) = exState.predAnswerProbs.keys.exists(isConsistent(ans, _))

  def isConsistent(ans:Answer, consAns:Answer) = cons.mode match {
    case "=" => ans == consAns // Equality
    case ">" => consAns isSubsetOf ans // Subset
    case "<" => ans isSubsetOf consAns // Superset
    case "~" => ans overlapsWith consAns // Overlap
    case "-" => !(ans overlapsWith consAns) // No overlap
    case _ => throw Utils.impossible
  }

  // Constraint's contribution to ans
  def outgoingWeight(ans:Answer) = {
    // Sum the probabilities over all consAns consistent with ans
    exState.predAnswerProbs.foldLeft(0.0) { case (sum,(consAns,prob)) =>
      if (isConsistent(ans, consAns)) sum+prob else sum
    }
  }
  def incorporateOutgoing(answerWeights:HashMap[Answer,Double], isInitial:Boolean) = {
    if (isInitial) {
      require (cons.mode == "=")
      answerWeights.clear
      answerWeights ++= exState.predAnswerProbs
    }
    else {
      answerWeights.foreach { case (ans,_) => // For each ans...
        answerWeights(ans) *= outgoingWeight(ans) // Multiply it in
        //dbgs("%s incorporateOutgoing: %s %s -> %s", id, render(ans), outgoingWeight(ans), answerWeights(ans))
      }
    }
    answerWeights.retain { case (ans,w) => w > 0 }
  }

  def incorporateIncoming(answerWeights:HashMap[Answer,Double]) = {
    Utils.foreach(exState.numCandidates, { i:Int => exState.targetFactors(i) = 0 }) // Initialize

    answerWeights.foreach { case (ans,weight) => // For each ans...
      val incomingWeight = weight / outgoingWeight(ans)
      if (incomingWeight.isNaN || incomingWeight.isInfinite) throw Utils.fails("Invalid");
      exState.predAnswerIndices.foreach { case (consAns,indices) => // For each consAns...
        if (isConsistent(ans, consAns)) { // ...that's consistent,
          //dbgs("%s incorporateIncoming: %s %s -> %s %s", id, render(ans), weight, render(consAns), incomingWeight)
          indices.foreach { i => exState.targetFactors(i) += incomingWeight } // add the incoming weight
        }
      }
    }
  }

  val exState = new BaseExampleInferState(ex, learner, ctxtId)
  def createPoints = exState.createPoints
  def oracleCorrect = throw Utils.impossible
  def predCorrect = throw Utils.impossible
  def numCandidates = exState.numCandidates
  def numErrorCandidates = exState.numErrorCandidates
  def numCorrectCandidates = exState.numCorrectCandidates
}

case class BasketInferState(basket:Basket, learner:MyLearner, ctxtId:String) extends InferState {
  def origQuestion = basket.constraints.head.sentence

  // For visualization
  def answerProbsToResponse(id:String, answerProbs:HashMap[Answer,Double], consState:ConstraintInferState) = {
    val sortedAnswerProbs = answerProbs.toList.sortWith(_._2 > _._2)
    //if (currPredAnswer == null) currPredAnswer = sortedAnswerProbs.head._1
    require (initialPredAnswer != null)
    if (currPredAnswer == null) currPredAnswer = initialPredAnswer
    val predAnswer = currPredAnswer

    val items = new ArrayBuffer[ResponseItem]
    items += MessageResponseItem(predAnswer.humanRender, main=true)
    val hasOldDiff = initialPredAnswer != null && initialPredAnswer != predAnswer
    if (hasOldDiff) {
      items += MessageResponseItem("Original question: "+origQuestion, main=false)
      items += MessageResponseItem("Initial answer: "+initialPredAnswer.humanRender, main=false)
    }
    val exState = consState.exState
    val vs = exState.findFirst(predAnswer)
    val oldvs = {
      if (hasOldDiff) exState.findFirst(initialPredAnswer)
      else None
    }

    // Lexical triggers
    items += GroupResponseItem(LexicalResponseItem((0 to exState.N-1).map { i =>
      WordInfo(exState.words(i), exState.tags(i), exState.transitions(i).map { case (pred,j) =>
        val f = exState.F_LexPred(exState.wordsStr(i, j), Predicate.predString(pred::Nil))
        (pred.name, exState.params.get(f))
      }.sortWith(_._2 > _._2).map{case (predName,w) => (predName, Utils.fmts("%s", w))})
    }) :: Nil)
    
    // Trees
    val treeItems = new ArrayBuffer[ResponseItem]
    vs.foreach { v => treeItems += SemTreeResponseItem("Prediction", v, exState) }
    oldvs.foreach { oldv => treeItems += SemTreeResponseItem("Old Prediction", oldv, exState) }
    items += GroupResponseItem(treeItems.toList)

    val counts = new HashMap[Feature,Double]
    vs.foreach { v => exState.foreachFeature(v, { f:Feature => counts(f) = counts.getOrElse(f, 0.0) + 1 }) }
    oldvs.foreach { oldv => exState.foreachFeature(oldv, { f:Feature => counts(f) = counts.getOrElse(f, 0.0) - 1 }) }

    val weightedCounts = counts.map { case (f,x) => (f,x * exState.params.get(f)) }
    val score = weightedCounts.values.sum
    val featureStr = {
      if (hasOldDiff) Utils.fmts("Features (score diff. = %s)", score)
      else            Utils.fmts("Features (score = %s, prob = %s)", score, answerProbs(predAnswer))
    }
    val featuresItem = ListResponseItem(featureStr,
      weightedCounts.toList.sortWith(_._2 > _._2).map { case (f,_) =>
        ListResponseItemElement(Utils.fmts("(%s) %s : %s", counts(f), f, exState.params.get(f)))
      })
    val candidatesItem = ListResponseItem("Candidate Answers", sortedAnswerProbs.flatMap {
      case (ans, prob) if !ans.isError =>
        val tooltip = exState.findFirst(ans) match {
          case Some(ans_v) => SemTreeResponseItem(null, ans_v, exState)
          case None => null // Shouldn't happen
        }
        val link = QueryBuilder.query(question=null, answer=answerIndexer.indexOf(ans).toString)
        Some(ListResponseItemElement(Utils.fmts("%s (%s)", ans.humanRender, prob),
                                     link=link, tooltip=tooltip))
      case _ => None
    })
    items += GroupResponseItem(featuresItem :: candidatesItem :: Nil)

    Response(items.toList)
  }

  //// Do main inference

  // Recursively solve each inference problem
  val consStates = new ArrayBuffer[ConstraintInferState]
  basket.constraints.foreach { cons => consStates += new ConstraintInferState(cons, learner, ctxtId) }
  val answerWeights = new HashMap[Answer,Double] // Weights on answers, incorporating all constraints
  consStates.zipWithIndex.foreach { case (consState,i) => // Outgoing
    consState.incorporateOutgoing(answerWeights, i == 0)
  }
  consStates.foreach(_.incorporateIncoming(answerWeights)) // Incoming

  val answerIndexer = new Indexer[Answer]
  def updateAnswerIndexer = answerWeights.keys.foreach { ans => answerIndexer.getIndex(ans) } // Index the answers
  updateAnswerIndexer

  val oracleCorrect = answerWeights.values.sum > 0
  if (oracleCorrect) MyUtils.normalize(answerWeights)

  val numErrorCandidates = consStates.map(_.numErrorCandidates).sum // Across all constraints
  val numCandidates = consStates.map(_.numCandidates).sum // Across all constraints
  val numCorrectCandidates = if (consStates.size == 0) 0 else {
    consStates.head.exState.predAnswers.count { ans => // Count the number of answers in the main constraint...
      // for which each constraint cons has some answer consAns that is consistent with it
      consStates.forall { cons => cons.exState.predAnswers.exists{consAns => cons.isConsistent(ans, consAns)} }
    }
  }
  val predCorrect = if (consStates.size == 0) false else {
    val ans = consStates.head.exState.predAnswer // Predicted answer of main constraint
    consStates.forall { cons => cons.isConsistent(ans, cons.exState.predAnswer) } // ...agrees with answers of all other constraints
  }
  var initialPredAnswer : Answer = null
  var currPredAnswer : Answer = null
  //def currPredAnswer : Answer = if (answerWeights.size == 0) null else MyUtils.argmax(answerWeights)

  //// Print out information

  if (MO.verbose >= 1) {
    def displayAnswers(map:HashMap[Answer,Double]) = {
      //map.keys.toList.filter(!_.isError).map(_.render).sortWith(_<_).foreach { Utils.logs("%s", _) }
      Utils.logs("%s", map.keys.toList.filter(!_.isError).map(_.render).sortWith(_<_).mkString(", "))
    }
    consStates.foreach { consState =>
      Utils.track("%s: %s answers", consState.exState.summary(), consState.exState.predAnswerProbs.size) {
        displayAnswers(consState.exState.predAnswerProbs)
      }
    }
    Utils.track("%s common answers", answerWeights.size) {
      displayAnswers(answerWeights)
    }
  }
  val tag = {
    if (!oracleCorrect) {
      if (impossible) "IMPOSSIBLE"
      else "INSUFFICIENT_BEAM"
    }
    else {
      if (predCorrect) "CORRECT"
      else "WRONG"
    }
  }
  val exStates = consStates.map(_.exState)
  Utils.logs("CTXT=%s,EX=%s\t%s\t%s\t%s", ctxtId, basket.id, tag, consStates.map(_.exState.words.mkString(" ")).mkString("\t"), answerWeights.keys.map(_.humanRender).mkString(" "))
  Utils.logs("Result %s: #words=%s, #preds=%s, %s/%s correct [%s]",
    basket.id,
    exStates.map(_.words.size).mkString("|"),
    exStates.map(_.usedPredicates.size).mkString("|"),
    numCorrectCandidates, numCandidates, tag)

  def createPoints = {
    if (answerWeights.size == 0) Nil // Can't learn
    else consStates.flatMap(_.createPoints)
  }

  def log = {
    if (MO.verbose >= 2) Utils.track("Constraints") {
      consStates.foreach { consState =>
        Utils.track("%s", consState.cons) {
          val consAnswerWeights = new HashMap[Answer,Double]
          consState.exState.predAnswerIndices.foreach { case (consAns,indices) =>
            consAnswerWeights(consAns) = consState.exState.targetFactors(indices.head)
          }
          MyUtils.normalize(consAnswerWeights)
          Response.displayResponse(answerProbsToResponse(basket.id, consAnswerWeights, consState))
        }
      }   
    }
  }

  /*def rejectCurrentAnswer : Response = {
    if (answerWeights.size == 0)
      MessageResponseItem("I already ran out of possible answers.")
    else {
      val currPredAnswer = MyUtils.argmax(answerWeights)
      answerWeights.retain { case (ans,i) => ans != currPredAnswer }
      consStates.foreach(_.incorporateIncoming(answerWeights))

      if (answerWeights.size == 0)
        MessageResponseItem("That's it, I'm out of ideas!")
      else {
        MyUtils.normalize(answerWeights)
        val ans = answerProbsToResponse(basket.id, answerWeights, consStates.head)
        ans ++ MessageResponseItem(Utils.fmts("Rejected current answer (%s left).", answerWeights.size))
      }
    }
  }*/

  def setCurrPredAnswer(id:Int) : Response = {
    val answer = {
      try { answerIndexer.getObject(id.toInt) }
      catch { case _ => null }
    }
    if (answer == null) return MessageResponseItem("Bad answer ID: "+id)
    currPredAnswer = answer

    val ans = answerProbsToResponse(basket.id, answerWeights, consStates.head)
    ans ++ MessageResponseItem(Utils.fmts("Temporarily set answer to '%s'.", answer.humanRender))
  }

  // Online learning
  def addConstraint(mode:String, sentence:String) : Response = {
    // Do inference 
    val cons = basket.createConstraint(mode, sentence)
    val consState = new ConstraintInferState(cons, learner, "int")
    if (consState.exState.numCandidates == 0)
      return MessageResponseItem("Sorry, I could not understand you.")

    def incorporate = {
      val isInitial = consStates.size == 0
      consState.incorporateOutgoing(answerWeights, isInitial)
      consStates.foreach(_.incorporateIncoming(answerWeights))
      MyUtils.normalize(answerWeights)
      updateAnswerIndexer 
      basket += cons
      consStates += consState
    }

    if (basket.size == 0) { // First example of the basket
      incorporate
      initialPredAnswer = MyUtils.argmax(answerWeights)
      return answerProbsToResponse(basket.id, answerWeights, consState)
    }

    // See if the new cons is consistent (if there is a consAns consistent with any existing weights)
    // Avoid empty sets, because they're rarely what the user's looking for.
    val consistent = answerWeights.keys.exists { ans => consState.isPossibleAnswer(ans) && ans.pred.size != 0 }
    if (!consistent) { // If not, we ignore it
      val msg = MessageResponseItem(
        Utils.fmts("Sorry, I can't see how '%s' is consistent with your previous %s:\n%s",
          cons.sentence, Renderer.noun("utterance", consStates.size),
          basket.constraints.map { case cons => "  "+cons}.mkString("\n")))
      val ans = answerProbsToResponse(null, consState.exState.predAnswerProbs, consStates.head)
      return ans ++ msg
    }

    // Apply the constraints
    incorporate

    answerProbsToResponse(basket.id, answerWeights, consStates.head)
  }
}
