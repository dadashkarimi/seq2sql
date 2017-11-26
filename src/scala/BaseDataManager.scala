package dcs

import fig.exec.Execution
import fig.prob.SampleUtils
import java.util.Random
import scala.collection.mutable.ArrayBuffer
import tea.Utils

/*
Generic class which mostly deals with managing training/test examples, where we don't look inside each example.
*/
trait BaseDataManager[Example] {
  import tea.OptionTypes._
  // Data
  @Option(gloss="Paths to data to be split into training and test") var generalPaths = Array[String]()
  @Option(gloss="Paths to training data") var trainPaths = Array[String]()
  @Option(gloss="Paths to test data") var testPaths = Array[String]()
  @Option(gloss="Number of examples to load from generalPaths") var generalMaxExamples = Integer.MAX_VALUE
  @Option(gloss="Number of examples to load from trainPaths") var trainMaxExamples = Integer.MAX_VALUE
  @Option(gloss="Number of examples to load from testPaths") var testMaxExamples = Integer.MAX_VALUE
  @Option(gloss="Fraction of general examples to use as training") var trainFrac = 1.0
  @Option(gloss="Fraction of general examples to use as testing") var testFrac = Double.NaN // By default, 1-trainFrac
  @Option(gloss="Random seed used to split general examples") var random = new Random(1)
  @Option(gloss="Verbosity level") var verbose = 0
  @Option(gloss="Whether to permute general examples before dividing") var permuteExamples = false
  @Option(gloss="Start reading with this example") var generalExampleOffset = 0

  val generalExamples = new ArrayBuffer[Example]
  val trainExamples = new ArrayBuffer[Example]
  val testExamples = new ArrayBuffer[Example]

  def loadExamples : Unit = Utils.track_printAll("DataManager.loadExamples") {
    loadExamples("general", generalPaths, generalMaxExamples, generalExamples)
    loadExamples("train", trainPaths, trainMaxExamples, trainExamples)
    loadExamples("test", testPaths, testMaxExamples, testExamples)

    // Split general examples into training and test
    generalExamples.remove(0, generalExampleOffset)
    if (generalExamples.size > 0) {
      val perm = {
        if (permuteExamples) SampleUtils.samplePermutation(random, generalExamples.size)
        else Utils.map(generalExamples.size, { i:Int => i })
      }
      val numTrain = (trainFrac * generalExamples.size) max 1
      val numTest = {if (testFrac.isNaN) generalExamples.size-numTrain else (testFrac*generalExamples.size).toInt}
      Utils.logs("Randomly splitting %s general examples", generalExamples.size)
      perm.zipWithIndex.foreach { case (pi,i) =>
        if (i < numTrain) trainExamples += generalExamples(pi)
        if (i >= generalExamples.size-numTest) testExamples += generalExamples(pi)
      }
    }
    Utils.logs("%s training examples, %s test examples", trainExamples.size, testExamples.size)
    Execution.putOutput("numTrainExamples", trainExamples.size)
    Execution.putOutput("numTestExamples", testExamples.size)
  }

  def addTrainExamples(path:String) = {
    val n = trainExamples.size
    readExamples(path, true, {ex:Example => trainExamples += ex})
    trainExamples.size - n
  }

  def loadExamples(tag:String, paths:Seq[String], max:Int, examples:ArrayBuffer[Example]) = if (paths.size > 0) Utils.track("Reading %s", tag) {
    def continue = examples.size < max
    paths.foreach { path =>
      if (continue) {
        Utils.track("Path: %s", path) {
          readExamples(path, continue, {ex:Example => examples += ex})
        }
      }
    }
  }

  def readExamples(path:String, continue: =>Boolean, add:Example=>Any) : Unit
}
