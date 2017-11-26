package dcs

import fig.exec.Execution
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedHashMap
import tea.Utils
import LearnTypes._
import PhraseTypes.Phrase
import EntityTypes.Entity

class NuggetModel(U:Universe, DM:DataManager) {
  // Note: one set of parameters shared across all sessions
  val params = new FeatureParams[Feature](LO)
  val counts = new FeatureParams[Feature](LO)
  val learner = new MyLearner(LO, params, counts)

  // Oredered by predAcurracy
  class Performance(mode:String, val iter:Int) extends Ordered[Performance] {
    // Four possibilities
    // | correct chosen | correct in beam | insufficient beam | impossible
    // |  predAccuracy  |                 |                   |
    // |          oracleAccuracy          |                   |
    var numTrueCorrect = 0
    var numPredCorrect = 0
    var numImpossible = 0
    var sumErrors = 0
    var sumCorrectCandidates = 0
    var sumCandidates = 0
    var numTotal = 0

    def trueAccuracy = 1.0*numTrueCorrect/numTotal
    def predAccuracy = 1.0*numPredCorrect/numTotal
    def impossibleFrac = 1.0*numImpossible/numTotal
    def numErrors = 1.0*sumErrors/numTotal
    def numCorrectCandidates = 1.0*sumCorrectCandidates/numTotal
    def numCandidates = 1.0*sumCandidates/numTotal

    def compare(that:Performance) = this.predAccuracy compare that.predAccuracy

    def log(endOfIter:Boolean) = {
      Utils.logs("%sOracle accuracy: %s/%s = %s; pred accuracy: %s/%s = %s; %s/%s=%s impossible, %s errors/ex, %s correct candidates/ex, %s candidates/ex",
        {if (endOfIter) "FINAL " else ""},
        numTrueCorrect, numTotal, trueAccuracy,
        numPredCorrect, numTotal, predAccuracy,
        numImpossible, numTotal, impossibleFrac,
        numErrors, numCorrectCandidates, numCandidates)
      if (endOfIter || iter == 0) { // At the end of iteration or during the first iteration
        val pairs = 
          (mode+"OracleAccuracy", trueAccuracy) ::
          (mode+"PredAccuracy", predAccuracy) ::
          (mode+"ImpossibleFrac", impossibleFrac) ::
          (mode+"NumErrors", numErrors) ::
          (mode+"NumCorrectCandidates", numCorrectCandidates) ::
          (mode+"NumCandidates", numCandidates) ::
          Nil
        if (endOfIter) // At the end of iteration
          pairs.foreach { case (a,b) => Execution.putLogRec(a,b) }
        else
          pairs.foreach { case (a,b) => Execution.putOutput(a,b) }
      }
    }
  }

  def pass(mode:String, examples:ArrayBuffer[Example], iter:Int) = Utils.track("%s pass", mode) {
    val perf = new Performance(mode, iter)
    val ctxtId = mode+iter

    def getInferState(ex:Example) : InferState = ex match {
      case ex:BaseExample => new BaseExampleInferState(ex, learner, ctxtId)
      case ex:Basket => new BasketInferState(ex, learner, ctxtId)
      case _ => throw Utils.impossible
    }

    val orderedExamples = {
      if (mode == "train") learner.permuteExamples(examples)
      else examples
    }

    Utils.track("%s examples", orderedExamples.size) {
      orderedExamples.zipWithIndex.foreach { case (ex, i) =>
        Execution.putOutput("currExample", i)
        Utils.track("Example %s (%s/%s) [%s]", ex.id, i, orderedExamples.size, mode) {
          try {
            val state = getInferState(ex)
            if (state.oracleCorrect) perf.numTrueCorrect += 1
            if (state.predCorrect) perf.numPredCorrect += 1
            if (state.impossible) perf.numImpossible += 1
            perf.numTotal += 1
            perf.sumErrors += state.numErrorCandidates
            perf.sumCorrectCandidates += state.numCorrectCandidates
            perf.sumCandidates += state.numCandidates

            if (mode == "train") state.update
          } catch { case TimedOutException =>
            Utils.logs("Timed out - skipping example")
          }
        }
        perf.log(i == examples.size-1)
      }
    }
    perf
  }

  def loadParams(path:String) = Utils.track("loadParams(%s)", path) {
    params.clear
    Utils.foreachLine(path, { line:String =>
      val Array(f, w) = line.split("\t")
      params.put(f, w.toDouble)
      true
    })
  }
  def saveParams = Utils.track("saveParams") {
    if (MO.outParamsPath != null)
      Utils.writeLines(MO.outParamsPath, params.output)
  }

  def learn = {
    learner.changed = true
    val testPerfs = new ArrayBuffer[Performance]

    def loop(iter:Int) : Unit = if (iter < LO.numIters && learner.changed) {
      Utils.track("Iteration %s/%s", iter, LO.numIters) {
        U.initPredMap

        learner.beginIteration(iter)
        pass("train", DM.trainExamples, iter)
        testPerfs += pass("test", DM.testExamples, iter)
        learner.endIteration

        U.finishPredMap
        if (MO.learnLexicon) U.updateLexicon(counts, Execution.getFile("lexicon."+iter))

        val bestPerf = testPerfs.max
        Execution.putOutput("testBestIter", bestPerf.iter)
        Execution.putOutput("testBestPredAccuracy", bestPerf.predAccuracy)
        Utils.logs("|relCache| = %s, |canonicalDens| = %s", Denotation.relCache.size, Denotation.canonicalDens.size)
      }
      loop(iter+1)
    }
    loop(0)
  }

  def run = {
    Utils.track("NuggetModel.run") {
      if (MO.inParamsPath != null) loadParams(MO.inParamsPath)
      learn
      saveParams
    }
  }

  def interact = {
    Utils.logs("Entered interactive mode.")
    val session = new Session("stdin")
    Utils.foreachLine("/dev/stdin", { line:String =>
      try {
        Response.displayResponse(session.handleRequest(GeneralRequest(line)))
      } catch {
        case e => Utils.logs("ERROR: %s", e)
        if (MO.verbose >= 2)
          e.printStackTrace
      }
      true
    })
  }

  val sessions = new LinkedHashMap[String,Session]
  def getSession(sessionId:String) = sessions.synchronized {
    sessions.getOrElseUpdate(sessionId, new Session(sessionId))
  }

  object OutBasketsPathLock
  class Session(sessionId:String) {
    val basketStates = new LinkedHashMap[String,BasketInferState] // For interactive mode

    def saveExamples(examples:Iterable[Example]) = if (MO.outBasketsPath != null) { OutBasketsPathLock.synchronized {
      val out = fig.basic.IOUtils.openOutAppendHard(MO.outBasketsPath)
      examples.foreach {ex =>
        val z = ex.toStatementItem
        if (z != null) out.println(z)
      }
      out.close
      Utils.logs("Saved %s examples to %s", examples.size, MO.outBasketsPath)
    } }

    def addNewLexicalEntry(phrase:Phrase, handle:PredicateHandle) = {
      U.synchronized { U.addLexicalEntry(phrase, handle) }
      if (MO.outBasketsPath != null) OutBasketsPathLock.synchronized {
        val out = fig.basic.IOUtils.openOutAppendHard(MO.outBasketsPath)
        val body = handle match {
          case PredicateName(name) => StrItem(name) // Predicate name
          case PredicateConstant(pred) => new Answer(pred).toTermItem // Specifying a predicate
        }
        out.println(TermItem("_lex", StrItem(phrase.mkString(" "))::body::Nil)+".")
        out.close
        Utils.logs("Saved [%s -> %s] to %s", phrase.mkString(" "), body, MO.outBasketsPath)
      }
    }

    def getLastQuestion = {
      if (basketStates.size == 0) ""
      else basketStates.values.last.origQuestion
    }

    def withLastBasketState(f:BasketInferState=>Response) : Response = {
      if (basketStates.size == 0) MessageResponseItem("Please enter a query first.")
      else f(basketStates.values.last)
    }

    def addConstraint(mode:String, sentence:String) : Response = {
      withLastBasketState { basketState =>
        val response = basketState.addConstraint(mode, sentence)
        if (mode == "=") basketState.basket.setLastAsAnswer // Just assume that the last equality constraint added is the answer
        response
      }
    }

    def createDefinition(word:String, definition:String) : Response = {
      // Create new basket
      val basket = new Basket("def", U.currWorld)
      val basketState = new BasketInferState(basket, learner, "int")
      val response = basketState.addConstraint("=", definition)
      val ans = basketState.currPredAnswer
      if (ans == null)
        MessageResponseItem(Utils.fmts("Sorry, I can't interpret '%s'.", definition))
      else {
        val predName = U.toPredName(word.replaceAll(" ", "_"), 1)
        val pred = ans.pred.rename(predName)
        U.synchronized {
          U.currWorld.addPredicate(pred)
          U.addLexicalEntry(Phrase.parse(word), PredicateName(predName))
        }
        // FUTURE: save to baskets, but don't have a way of representing this right now
        Response(MessageResponseItem(Utils.fmts("Remembering '%s' means '%s'.", word, definition))::Nil) ++ response
      }
    }

    val LexPattern = """(.+)\s*=\s*(.+)""".r
    val DefPattern = """=\s*(.+)""".r
    val CommandPattern = """!(.+)""".r
    def handleRequest(request:Request) : Response = request match {
      case GeneralRequest(line) if line.size > DM.maxSentenceLength =>
        MessageResponseItem("Sorry, your input was too long.")
      case GeneralRequest(DefPattern(str)) =>
        withLastBasketState { basketState =>
          val ans = basketState.currPredAnswer
          addNewLexicalEntry(Phrase.parse(str), PredicateConstant(ans.pred.rename(str)))
          MessageResponseItem(Utils.fmts("'%s' maps to %s.", str, ans.humanRender))
        }
      case GeneralRequest(LexPattern(a,b)) =>
        U.matchPredicates(b) match {
          case Nil =>
            //createDefinition(a, b)
            MessageResponseItem(Utils.fmts("'%s' maps to no predicates.", b))
          case predName :: Nil =>
            addNewLexicalEntry(Phrase.parse(a), PredicateName(predName))
            MessageResponseItem(Utils.fmts("Ok, remembering that \"%s\" means %s.", a, b))
          case predNames =>
            MessageResponseItem(Utils.fmts("'%s' is ambiguous between %s.", b, predNames.mkString(", ")))
        }
      case GeneralRequest(CommandPattern(cmd)) => processCommand(cmd)
      case SetAnswerRequest(id) => withLastBasketState(_.setCurrPredAnswer(id))
      case AddExampleRequest =>
        // Create a training example out of what we learned on this basket
        withLastBasketState { basketState =>
          val question = basketState.origQuestion
          val answer = basketState.currPredAnswer
          if (answer == null)
            MessageResponseItem("No answers.")
          else {
            val ex = new BaseExample(DM.trainExamples.size.toString, SentenceTokenizer.tokenizeSentence(question), U.currWorld, answer, sessionId)
            saveExamples(ex::Nil)
            if (MO.autoUpdate) {
              val inferState = new BaseExampleInferState(ex, learner, "intex")
              val points = inferState.createPoints
              Utils.logs("Taking a gradient step on %s points", points.size)
              learner.synchronized { learner.takeGradientStep(points) }
            }
            val actionMessage = {
              if (MO.autoUpdate) "updated my parameters"
              else "will update my parameters later"
            }
            MessageResponseItem(Utils.fmts("Ok, remembering that the answer to \"%s\" is %s and %s.", question, answer.humanRender, actionMessage))
          }
        }
      case GeneralRequest(line) =>
        //else if (line.toLowerCase == "no") withLastBasketState(_.rejectCurrentAnswer)
        if (Constraint.isValidMode(line.slice(0, 1)))
          addConstraint(line.substring(0, 1), line.substring(1))
        else { // Ordinary question
          // Create new basket
          val basket = new Basket(DM.trainExamples.size.toString, U.currWorld)
          val basketState = new BasketInferState(basket, learner, "int")
          basketStates(basket.id) = basketState
          basketState.addConstraint("=", line)
        }
      case SentenceRequest(basketId, mode, sentence) =>
        basketStates.get(basketId) match {
          case None => MessageResponseItem("Internal error: invalid basket ID '"+basketId+"' received.")
          case Some(basketState) =>
            basketState.addConstraint(mode, sentence)
        }
    }

    // Warning: all of these commands are not synchronized (because assume multiple people won't execute them)
    def processCommand(line:String) : Response = {
      line.split(" ").toList match {
        case "verbose" :: s :: Nil =>
          MO.verbose = s.toInt
          MessageResponseItem("Verbosity set to "+MO.verbose+".")
        case "display" :: s :: Nil => {
          s match {
            case "types" => IO.displayTypes = !IO.displayTypes
            case "spans" => IO.displaySpans = !IO.displaySpans
            case "dens" => IO.displayDens = !IO.displayDens
            case _ =>
          }
          MessageResponseItem("Ok.")
        }
        case "load" :: path :: Nil => {
          if (path.endsWith(".params")) {
            loadParams(path)
            MessageResponseItem("Loaded %d parameters from %d.".format(params.numFeatures, path))
          }
          else if (path.endsWith(".dlog")) {
            val n = DM.addTrainExamples(path)
            MessageResponseItem(Utils.fmts("Loaded %s examples from %s.", n, path))
          }
          else
            MessageResponseItem(Utils.fmts("'%s' has invalid extension (must be .params or .dlog).", path))
        }
        case "save" :: Nil => {
          saveParams
          MessageResponseItem(Utils.fmts("%s parameters saved to disk.", params.numFeatures))
        }
        case "settings" :: Nil => {
          val l =
            ("Features", MO.features.mkString(" "), "Feature (templates) used by the current system.") ::
            ("Parameters", Utils.fmts("%s (%s => %s)", params.numFeatures, MO.inParamsPath, MO.outParamsPath), "Number of parameters, where parameters are initially read from, and where the updated parameters will be written.") ::
            ("Options", MO.dlogOptions.mkString(" "), "Options that determine the lexicon and any properties of the world/database.") ::
            ("Examples", Utils.fmts("%s train", DM.trainExamples.size), "These are examples that we've collected.") ::
            ("Training", Utils.fmts("%s iterations with %s update", LO.numIters, LO.updateType), "On !learn, this is what we do.") ::
            ("Timeout", MO.timeout, "Number of seconds allowed per query.") ::
            ("Beam size", MO.beamSize, "Truncate to this many hypotheses per span.") ::
            ("Auto-update", MO.autoUpdate, "When an example is added, do we make an update immediately?") ::
            ("Sessions", sessions.size, "Number of user sessions.") ::
            ("Verbosity", MO.verbose, "How much output (to the console)") ::
            Nil
          ListResponseItem("Settings", l.map{ case (k,v,tooltip) =>
            ListResponseItemElement(k+": "+v, tooltip=MessageResponseItem(tooltip))
          })
        }
        case "timeout" :: s :: Nil => {
          val oldTimeout = MO.timeout 
          MO.timeout = s.toInt
          MessageResponseItem(Utils.fmts("Changed timeout from %s to %s.", oldTimeout, MO.timeout))
        }
        case "beam" :: s :: Nil => {
          val oldBeamSize = MO.beamSize 
          MO.beamSize = s.toInt
          MessageResponseItem(Utils.fmts("Changed beam size from %s to %s.", oldBeamSize, MO.beamSize))
        }
        case "auto" :: s :: Nil => {
          if (s == "true") { MO.autoUpdate = true; MessageResponseItem("Enabled auto update.") }
          else{ MO.autoUpdate = false; MessageResponseItem("Disabled auto update.") }
        }
        case "learn" :: Nil =>
          learn
          MessageResponseItem("Model relearned.")
        case _ =>
          MessageResponseItem("Sorry, I cannot understand your command.")
      }
    }
  }
}
