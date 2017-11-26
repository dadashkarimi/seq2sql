package dcs

import edu.berkeley.nlp.tokenizer.PTBLineLexer

object SentenceTokenizer {
  def tokenizeSentence(origLine:String) = {
    // Replace "from30" with "from 30" (crappy data in countries)
    var buf = new StringBuilder
    var lastc : Char = 0
    origLine.foreach { c =>
      if (lastc.isLetter && c.isDigit) buf += ' '
      buf += c
    }

    val line = buf.toString
    (new PTBLineLexer).tokenize(line).toArray(Array[String]()).toList.map(_.replaceAll("""\\/""", "/"))
  }
  /*def tokenizeSentence(line:String) = {
    List("\\.", ",", ";", "\\?", "%").foldLeft(line) { case (line,c) => // Tokenize punctuation
      line.replaceAll(c, " "+c+" ")
    }.replaceAll("'", " '").trim.toLowerCase.split("\\s+").toList
  }*/
}
