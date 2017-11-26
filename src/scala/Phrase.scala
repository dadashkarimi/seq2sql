package dcs

object PhraseTypes {
  type Phrase = List[String]
}
object Phrase {
  def parse(s:String) = s.split(" ").toList
}
