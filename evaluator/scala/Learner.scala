package dcs

/*
General-purpose library for doing classification/reranking.
For each example, have a set of choices (subset of which are correct).

Usage:
Construct a set of points and call updateParams(points).
Batch size:
 - Online: do this after each example
 - Batch: do this once
Dynamic reranking: get a fresh set of points each iteration.

updateType:
 - gradient: linearize the objective and move in that gradient
 - full: solve the problem fully, regularizing towards the current setting of the weights.
  
- Update can either work with the linearized objective (gradient) or solve completely.
*/

import fig.basic.Indexer
import fig.exec.Execution
import fig.record.Record
import java.util.Random
import scala.collection.mutable.ArrayBuffer
import tea.Utils

class LearnOptions {
  import tea.OptionTypes._
  @Option(gloss="Number of iterations") var numIters = 3
  @Option(gloss="Initial stepsize") var initStepSize = 1.0
  @Option(gloss="Power on step size reduction (1.0 = 1/t; 0.5 = 1/sqrt(t))") var stepSizeReductionPower = 0.0
  @Option(gloss="Random seed used to permute examples") var random = new Random(1)
  @Option(gloss="Permute examples each training iteration") var permuteExamples = false
  @Option(gloss="Use AdaGrad") var useAdaptiveGrad = false
  @Option(gloss="Update this many examples at once") var miniBatchSize = 1
  @Option(gloss="Type of update (gradient|full)") var updateType = "gradient"
  @Option(gloss="Regularization parameter (lambda)") var regularization = 0.0
  @Option(gloss="Whether to randomize the training examples") var randTrainOrder = false
  @Option(gloss="Maximum number of lines to output") var maxOutput = Integer.MAX_VALUE

  @Option(gloss="Passive aggressive updates: only update when the margin isn't large enough") var passiveAggressive = false
  @Option(gloss="When do gradient update, take <=maxNudgeSteps until we get to this probability") var marginProb = 0.7
  @Option(gloss="Maximum number of lines to output") var maxNudgeSteps = 1

  @OptionSet(name="backtrack") val btOpts = new BacktrackingLineSearch.Options
  @OptionSet(name="lbfgs") val lbfgsOpts = new LBFGSMaximizer.Options
}
object LO extends LearnOptions

// Goal: minimize KL between t(y) \propto targetFactor(y) p_theta(y) and p_theta(y)
case class Point[Example,Feature](val ex:Example, val choices:Seq[Choice[Feature]])
case class Choice[Feature](features:Seq[Feature], targetFactor:Double) {
  val logTargetFactor = math.log(targetFactor)
  if (logTargetFactor.isNaN) throw Utils.fails("Bad: %s", targetFactor)

  def computeScore(params:FeatureParams[Feature]) = {
    features.foldLeft(0.0) { case (sum, f) =>
      sum + params.get(f)
    }
  }

  var featureIndices : Seq[Int] = null // Used by batch learning
  //def computeScore(theta:Array[Double]) = featureIndices.foldLeft(0.0) { case (sum, f) => sum + theta(f) }
  def computeScore(theta:Array[Double]) = { // MORE EFFICIENT
    var sum = 0.0
    featureIndices.foreach { f => sum += theta(f) }
    sum
  }
}

class Learner[Example,Feature](LO:LearnOptions, val params:FeatureParams[Feature], val counts:FeatureParams[Feature]) {
  type MyPoint = Point[Example,Feature]
  // Used by full optimization only
  val featureIndexer = new Indexer[Feature]
  def F = featureIndexer.size

  val points = new ArrayBuffer[MyPoint]
  var iter = -1
  var numUpdates = 0
  var changed = true

  def beginIteration(iter:Int) = {
    this.iter = iter
    require (points.size == 0)
    Record.begin("iteration", iter)
    Execution.putOutput("currIter", iter)
    changed = false
  }
  def addPoint(point:MyPoint) = {
    points += point
    if (points.size >= LO.miniBatchSize)
      updateParams
  }
  def endIteration = {
    updateParams
    Utils.writeLines(Execution.getFile("params."+iter), params.output)
    Utils.writeLines(Execution.getFile("counts."+iter), counts.output)
    Record.end
  }

  def permuteExamples(examples:ArrayBuffer[Example]) : Seq[Example] = {
    if (LO.permuteExamples) {
      val perm = fig.prob.SampleUtils.samplePermutation(LO.random, examples.size)
      perm.map(examples)
    }
    else
      examples
  }

  def updateParams : Unit = {
    if (points.size == 0) return
    numUpdates += 1

    // 1) Linearize: take just a gradient step
    // 2) Full: re-optimize regularizing to params
    LO.updateType match {
      case "gradient" => takeGradientStep(points)
      case "full" => fullyOptimize(points)
      case s => throw Utils.fails("Invalid updateType: %s", s)
    }
    points.clear
  }

  def scoresToTargetPredProbs(choices:Seq[Choice[Feature]], scores:Seq[Double]) = {
    val predProbs = scores.toArray.clone
    //(choices zip predProbs).foreach { case (choice, score) => dbgs("%s %s %s", score, choice.logTargetFactor, score + choice.logTargetFactor) }
    val targetProbs = (choices zip predProbs).map { case (choice, score) => score + choice.logTargetFactor }.toArray
    val logZ = Utils.expNormalizeLogZ_!(targetProbs) - Utils.expNormalizeLogZ_!(predProbs)
    //dbgs("T %s", fmt1(targetProbs))
    //dbgs("P %s", fmt1(predProbs))
    (targetProbs, predProbs, logZ)
  }

  def takeGradientStep(points:Seq[MyPoint]) = {
    val currStepSize = LO.initStepSize / math.pow(numUpdates, LO.stepSizeReductionPower)
    val logMarginProb = math.log(LO.marginProb)

    points.zipWithIndex.foreach { case (Point(_, choices), pi) => // For each point...
      // Compute the initial gradient
      var (targetProbs, predProbs, logZ) = scoresToTargetPredProbs(choices, choices.map(_.computeScore(params)))

      def updateInGradDirection(magnitude:Double) = {
        // Update in the direction
        choices.zipWithIndex.foreach { case (choice,i) => // For each choice...
          val d = magnitude * (targetProbs(i)-predProbs(i))
          choice.features.foreach { f =>
            //dbgs("UPDATE i=%s | %s : %s - %s = %s", i, f, targetProbs(i), predProbs(i), targetProbs(i)-predProbs(i))
            params.incr(f, d)
          }
        }
        changed = true
      }

      if (LO.passiveAggressive) {
        // Modify the step size so we get a certain probability
        //Utils.dbgs("POINT %s/%s: prob(correct) = %s", pi, points.size, math.exp(logZ))
        val origLogZ = logZ

        if (logZ < logMarginProb) { // Only update if we're not confident enough
          var ni = 0
          var alpha = 0.0 // Step size in the direction of the gradient
          while (ni < LO.maxNudgeSteps) {
            val sign = { if (logZ > logMarginProb) -1 else +1 } // Where to go next
            val newAlpha = math.max(alpha + sign * currStepSize / (ni+1), 0)

            updateInGradDirection(newAlpha-alpha)

            ni += 1
            alpha = newAlpha
            logZ = scoresToTargetPredProbs(choices, choices.map(_.computeScore(params)))._3 // Recompute prob(correct)
            //Utils.dbgs("NUDGE %s: alpha = %s, prob(correct) = %s", ni, alpha, math.exp(logZ))
          }
          Utils.logs("UPDATE: alpha = %s, prob(correct) = %s -> %s", alpha, math.exp(origLogZ), math.exp(logZ))
        }
      }
      else
        updateInGradDirection(currStepSize)
    }
  }

  def fullyOptimize(points:Seq[MyPoint]) : Unit = Utils.track("fullyOptimize(%s points)", points.size) {
    // Index all data
    val oldNumFeatures = F
    Utils.track("Indexing features for %s points", points.size) {
      points.foreach { point =>
        point.choices.foreach { choice =>
          choice.featureIndices = choice.features.map(featureIndexer.getIndex)
        }
      }
      Utils.logs("%s -> %s features", oldNumFeatures, F)
    }

    val maximizer = new LBFGSMaximizer(LO.btOpts, LO.lbfgsOpts)
    val state = new FunctionState {
      val theta = Utils.map(F, { f:Int => params.get(featureIndexer.getObject(f)) })
      val dtheta = new Array[Double](F)
      var objective = Double.NaN
      var dthetaValid = false
      var objectiveValid = false

      val mu = new Array[Double](F) // Expected counts: E_{p(y|x)}[\phi(x,y)]

      def point = theta
      def gradient = {
        if (!dthetaValid) compute(true)
        dtheta
      }
      def value = {
        if (!objectiveValid) compute(false)
        objective
      }
      def invalidate = {
        dthetaValid = false
        objectiveValid = false
      }

      def compute(updateGradient:Boolean) = {
        objective = 0
        Utils.set_!(dtheta, 0)
        if (updateGradient) Utils.set_!(mu, 0)
        points.foreach { case Point(_, choices) =>
          val (targetProbs, predProbs, logZ) = scoresToTargetPredProbs(choices, choices.map(_.computeScore(theta)))
          objective += logZ
          
          if (updateGradient) {
            choices.zipWithIndex.foreach { case (choice,i) =>
              val d = targetProbs(i) - predProbs(i)
              choice.featureIndices.foreach { f =>
                dtheta(f) += d
                mu(f) += predProbs(i)
              }
            }
          }
        }

        // Regularization
        Utils.foreach(F, { f:Int =>
          objective -= 0.5 * LO.regularization * theta(f)*theta(f)
          dtheta(f) -= LO.regularization * theta(f)
        })

        if (updateGradient) dthetaValid = true
        objectiveValid = true
      }
    }

    while (Utils.track("Maximizer.takeStep") {!maximizer.takeStep(state)}) {
      changed = true
    }

    // Put parameters back
    def copy(target:FeatureParams[Feature], source:Array[Double]) = {
      target.clear
      Utils.foreach(F, { f:Int => target.put(featureIndexer.getObject(f), source(f)) })
    }
    copy(params, state.theta)
    copy(counts, state.mu)
  }
}
