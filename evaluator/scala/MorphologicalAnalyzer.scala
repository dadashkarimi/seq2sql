package dcs

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap

// English specific
// http://www.brighthub.com/education/languages/articles/51735.aspx
// http://abacus-es.com/sat/verbs.html
// http://abacus-es.com/sat/adjectives.html

// This is all very hacky and should be replaced with a nice clean general morphological analyzer.

case class InflectedWord(surface:String, stem:String, form:String)

class MorphologicalAnalyzer {
  private def prepare(stem:String, result:Seq[(String,String)]) = {
    result.map { case (surface, form) =>
      InflectedWord(surface, stem, form)
    }.toList
  }
  private def chop(s:String, n:Int) = s.substring(0, s.size-n)

  // pass -> pass, passes, passed, passing
  def getVerbForms(stem:String) : List[InflectedWord] = {
    val result = new ArrayBuffer[(String,String)]
    result += ((stem, "")) // infinitive
    if (stem.endsWith("s"))
      result += ((stem+"es", "s")) // 3sg
    else
      result += ((stem+"s", "s")) // 3sg
    if (stem.endsWith("e")) {
      result += ((stem+"d", "ed")) // past
      result += ((chop(stem, 1)+"ing", "ing")) // gerund
    }
    else {
      if (stem == "run")
        result += (("ran", "ed")) // past
      else
        result += ((stem+"ed", "ed")) // past
      if (stem.endsWith("n"))
        result += ((stem+"ning", "ing")) // gerund
      else
        result += ((stem+"ing", "ing")) // gerund
    }
    prepare(stem, result)
  }

  // city -> city, cities
  def getNounForms(stem:String) = {
    val result = new ArrayBuffer[(String,String)]
    result += ((stem, "sg"))
    if (stem.endsWith("s"))
      result += ((stem+"es", "pl"))
    else if (stem.endsWith("y"))
      result += ((chop(stem, 1)+"ies", "pl"))
    else if (stem == "person")
      result += (("people", "pl"))
    else
      result += ((stem+"s", "pl"))
    prepare(stem, result)
  }

  // large -> larger, largest
  def getAdjForms(stem:String) = {
    val result = new ArrayBuffer[(String,String)]
    if (stem == "big") {
      result += (("biggest", "most"))
      result += (("bigger", "more"))
    }
    else if (stem.endsWith("e")) {
      result += ((stem+"st", "most"))
      result += ((stem+"r", "more"))
    }
    else if (stem.endsWith("y")) {
      result += ((chop(stem,1)+"iest", "most"))
      result += ((chop(stem,1)+"ier", "more"))
    }
    else {
      result += ((stem+"est", "most"))
      result += ((stem+"er", "more"))
    }
    prepare(stem, result)
  }

  def pluralize(word:String) = {
    if (word == "foot") "feet"
    else if (word.endsWith("y")) word.substring(0, word.size-1)+"ies"
    else if (word.endsWith("s")) word+"es"
    else word+"s"
  }

  // largest -> most large
  // larger -> more large
  val surface2forms = new HashMap[String,InflectedWord]
  List("big", "large", "small", "great", "dense", "sparse", "long", "tall", "short", "high", "low", "late", "few", "early", "cheap").foreach { stem =>
    getAdjForms(stem).foreach { form =>
      surface2forms(form.surface) = form
    }
  }

  def decompose(surface:String) : InflectedWord = {
    surface2forms.getOrElse(surface, InflectedWord(surface, surface, ""))
  }

  def stem(s:String) : String = {
    val stemmer = new PorterStemmer
    s.toLowerCase.foreach(stemmer.add)
    stemmer.stem
    stemmer.toString
  }
}

object MorphologicalAnalyzer {
  val theMorpher = new MorphologicalAnalyzer
}
