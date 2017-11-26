package dcs

import fig.exec.Execution
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.mutable.LinkedHashMap
import scala.collection.mutable.LinkedHashSet
import PhraseTypes.Phrase
import EntityTypes.Entity
import tea.Utils
import LearnTypes.Feature

/*
The universe containsa set of worlds and some meta-information for mapping
natural language - like the lexical triggers (origPhraseMap).
*/
class Universe {
  val beginWord = "^"
  def toPredName(name:String, arity:Int) = name+"/"+arity

  //// Worlds
  val defaultWorld = new World("default", null, this)
  var currWorld = defaultWorld
  val worlds = new LinkedHashMap[String,World] // World ID -> world
  worlds(defaultWorld.id) = defaultWorld
  def getWorld(id:String) = worlds.getOrElseUpdate(id, new World(id, defaultWorld, this))

  type NormPhrase = String
  type NormType = String // Manner of normalization (full, prefix, suffix, abbreviation)
  val NT_EXPLICIT = "explicit" // Explicit lexical entry
  val NT_FULL = "full"
  def NT_SINGLE(i:Int) = "single"+i
  val NT_NUM = "num"

  //// For parsing language
  val rewrites = new HashMap[Phrase,Phrase] // e.g., when -> what time

  // predMap = normPhrase(origPhraseMap) + normPhrase(learnedPhraseMap)
  val origPhraseMap = new HashMap[Phrase,HashSet[PredicateHandle]] // word -> predicate handle
  val learnedPhraseMap = new HashMap[Phrase,HashSet[PredicateHandle]] // word -> predicate handle
  val predMap = new HashMap[NormPhrase,HashSet[PredicateHandle]] // normed phrase -> predicate handle

  val nameMap = new HashMap[NormPhrase,HashSet[String]] // normalized phrase -> names
  val usedNormPhrases = new HashSet[NormPhrase]
  // FUTURE: compute nameMap only for names that appear in the nameObj predicate.

  // Lexicon (lexMap)
  //   Word -> predicate
  //   POS -> predicate
  // Map each word to POS
  //   Type-level dictionary
  //   Token-level tagger (optional)
  // Parsing time: if word->predicate exists, use that; else, look at all possible POS tags and use that.
  // Grow lexicon time: form p(predicate|word) based on lexPred feature and truncate.  Set the word->predicate based on that.
  // For each word, keep track of whether to back off to POS
  val posTagDict = new HashMap[String, List[String]] // Word -> POS
  def loadPosTagDict = if (MO.posTagDictPath != null) {
    Utils.foreachLine(MO.posTagDictPath, { line:String =>
      val Array(word, tags) = line.split("\t")
      posTagDict(word) = tags.split(" ").toList
      true
    })
    Utils.logs("Loaded %s entries for the POS tagging dictionary", posTagDict.size)
  }
  def getPosTags(s:String) = posTagDict.getOrElse(s, Nil)

  def initPredMap = {
    usedNormPhrases.clear
    predMap.clear
    List(origPhraseMap, learnedPhraseMap).foreach { map =>
      map.foreach { case (phrase,handles) =>
        predMap.getOrElseUpdate(normPhrase(phrase), new HashSet[PredicateHandle]) ++= handles
      }
    }
  }
  def finishPredMap = {
    // Print out statistics on how we used the lexicon
    val origUsedNormPhrases = origPhraseMap.keys.map(normPhrase).toSet & usedNormPhrases
    Utils.logs("%s used original lexical entries: %s", origUsedNormPhrases.size, origUsedNormPhrases.toList.sortWith(_<_).mkString(" "))
    val usedPredicateHandles = new HashSet[PredicateHandle]
    usedNormPhrases.foreach { np =>
      predMap(np).foreach(usedPredicateHandles += _)
    }
    Utils.logs("%s used predicates: %s", usedPredicateHandles.size, usedPredicateHandles.map(_.name).toList.sortWith(_<_).mkString(" "))
  }

  val LexPredPattern = """lexpred:(.+):(.+/\d+)""".r
  def updateLexicon(counts:FeatureParams[Feature], outPath:String) = Utils.track("updateLexicon") {
    // For each word, compute distribution over predicates
    val probs = new HashMap[String, HashMap[String, Double]] // word -> distribution over predicates
    counts.foreachFeatureWeight { (f,v) => f match {
      case LexPredPattern(word, predName) =>
        probs.getOrElseUpdate(word, new HashMap[String,Double])(predName) = v+1e-10
      case _ =>
    } }
    probs.values.foreach(MyUtils.normalize)

    // Populate learned lexicon with this information
    learnedPhraseMap.clear
    probs.foreach { case (word,map) =>
      if (!origPhraseMap.contains(word::Nil)) {
        learnedPhraseMap(word::Nil) = HashSet[PredicateHandle]() ++ map.flatMap { case (predName,prob) =>
          if (prob < MO.lexiconPruneThreshold) Nil // Prune away
          else PredicateName(predName) :: Nil // Keep
        }
      }
    }
    Utils.logs("Learned lexicon size: %s", learnedPhraseMap.values.map(_.size).sum)

    // Write learned lexicon
    Utils.writeLines(outPath, { puts:(String=>Any) =>
      learnedPhraseMap.toList.sortWith(_._1.mkString(" ") < _._1.mkString(" ")).foreach { case (phrase,handles) =>
        puts(Rule(TermItem("_lex", StrItem(phrase.mkString(" ")) :: ListItem(handles.map{handle => StrItem(handle.name)}.toList) :: Nil), null).toString)
      }
    })
  }

  //// Normalized phrase
  def keyIsSpecial(s:String) = s.startsWith("-") && s.endsWith("-")
  def keyIsPos(s:String) = s.startsWith(":")
  def normPhrase(phrase:Phrase) : NormPhrase = { // When parsing, compute normalized phrase to look up predicates
    if (phrase.size == 1 && keyIsSpecial(phrase.head)) phrase.head // -BRIDGE-
    else if (phrase.size == 1 && keyIsPos(phrase.head)) phrase.head // :NN
    else {
      phrase.flatMap(_.split("[ _/]").toList)
        .map(MorphologicalAnalyzer.theMorpher.stem)
        .map(_.slice(0, MO.normPhrasePrefixLen))
        .mkString("").toLowerCase.replaceAll("""[_\-\.]""", "")
    }
  }

  def getPredHandles(phrase:Phrase) : Seq[(NormType,PredicateHandle)] = {
    def fromPred(p:Predicate) = (NT_NUM, PredicateConstant(p))
    def fromValue(v:Value) = (NT_NUM, PredicateConstant(SingletonPredicate(Entity(v))))
    val seq = phrase.mkString(" ")
    val np = normPhrase(phrase)
    if (predMap.contains(np)) usedNormPhrases += np // Mark lookup of lexical entry

    predMap.getOrElse(np, new HashSet[PredicateHandle]).map{handle => (NT_EXPLICIT, handle)}.toList ++ // Lexicon (e.g., traverse)
    NumberParser.parse(seq).map(fromValue) ++ // Number (e.g., thirty four)
    NumberParser.parseRange(seq).map { case (a,b) => fromPred(new NumRangePredicate(CountDomain, a, false, b, false)) } ++
    {if (MO.lexToTime) NumberParser.parseTime(seq).map(fromValue) else Nil} // Time (e.g., 4pm)
  }

  def log = {
    initPredMap
    Utils.logs("%s->%s lexical entries, %s->%s names",
      predMap.size, predMap.values.map(_.size).sum,
      nameMap.size, nameMap.values.map(_.size).sum)
    Utils.logs("%s worlds: %s", worlds.size, worlds.values.mkString(" "))

    Utils.writeLines(Execution.getFile("universe.txt"), { puts:(String=>Any) =>
      Domains.allDomains.foreach { dom =>
        puts(Array("DOM", dom).mkString("\t"))
      }
      puts("")
      val hit = new HashSet[PredicateHandle]
      predMap.foreach { case (np, handles) =>
        handles.foreach(hit += _)
        puts((Array("LEX", np) ++ handles.map(_.name)).mkString("\t"))
      }
      puts("")
      val predicates = new HashSet[Predicate]
      worlds.values.foreach { w => predicates ++= w.getPredicates }
      predicates.foreach { pred =>
        if (!hit(PredicateName(pred.name)))
          puts(Array("UNUSED_PRED", pred.name).mkString("\t"))
      }
      puts("")
      predicates.foreach { pred =>
        puts(Array("TYPE", pred.name, pred.abs.render, pred.size).mkString("\t"))
      }
      puts("")
      nameMap.foreach { case (np, names) =>
        puts((Array("NAME", np) ++ names).mkString("\t"))
      }
    })
  }

  def containsPredicate(s:String) = worlds.values.exists(_.containsPredicate(s))

  def matchPredicates(s:String) = {
    val names = s :: List(1, 2, 3, 4, 5).map{toPredName(s, _)}
    names.filter(containsPredicate)
  }

  def str2predicateHandle(s:String) : PredicateHandle = { // e.g., traverse/2
    if (!containsPredicate(s))
      throw Utils.fails("Predicate name %s not found in any world", s)
    PredicateName(s)
  }

  val morpher = MorphologicalAnalyzer.theMorpher

  def addRewrite(source:String, target:String) = {
    rewrites(Phrase.parse(source)) = Phrase.parse(target)
  }
  def rewriteSentence(sentence:List[String]) : Phrase = {
    var words = sentence

    // Apply arbitrary rewrites (this shouldn't be used very much)
    val maxPhraseLen = 4
    (1 to maxPhraseLen).reverse.foreach { len => // Try replacing longer phrases first
      var i = 0
      while (i+len <= words.size) {
        rewrites.get(words.slice(i, i+len)) match {
          case Some(subst) => words = words.patch(i, subst, len)
          case None =>
        }
        i += 1
      }
    }

    // Apply selective morphology (e.g., largest -> most large)
    words = words.flatMap { word =>
      val iw = morpher.decompose(word)
      if (iw.form == "") List(iw.stem)
      else List(iw.form, iw.stem)
    }

    words
  }

  def addLexicalEntry(str:String, handle:PredicateHandle) : Unit = {
    str.split("/") match {
      case Array(str) => addLexicalEntry(Phrase.parse(str), handle)
      case Array(stem, pos) => {
        val forms = pos match {
          case "N" => morpher.getNounForms(stem)
          case "V" => morpher.getVerbForms(stem)
          case _ => throw Utils.fails("Invalid POS (want N or V): %s", pos)
        }
        forms.foreach { form =>
          addLexicalEntry(form.surface::Nil, handle)
        }
      }
    }
  }
  def addLexicalEntry(phrase:Phrase, handle:PredicateHandle) : Unit = {
    require (handle != null, phrase)
    origPhraseMap.getOrElseUpdate(phrase, new HashSet[PredicateHandle]) += handle
    predMap.getOrElseUpdate(normPhrase(phrase), new HashSet[PredicateHandle]) += handle
  }

  def extractValues(pred:Predicate) : Unit = {
    val op = pred.enumerate
    if (op.possible) op.perform.foreach(extractValues)
  }

  def extractValues(e:Entity) : Unit = {
    e.foreach {
      case NameValue(name) =>
        val np = normPhrase(name::Nil)
        nameMap.getOrElseUpdate(np, new HashSet[String]) += name
      case _ =>
    }
  }
}
