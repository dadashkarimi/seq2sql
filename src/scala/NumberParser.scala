package dcs

object NumberParser {
  val countOnes = Map[String,Double](
    "zero" -> 0, "one" -> 1, "two" -> 2, "three" -> 3, "four" -> 4,
    "five" -> 5, "six" -> 6, "seven" -> 7, "eight" -> 8, "nine" -> 9)
  val countTeens = Map[String,Double](
    "ten" -> 10, "eleven" -> 11, "twelve" -> 12, "thirteen" -> 13, "fourteen" -> 14,
    "fifteen" -> 15, "sixteen" -> 16, "seventeen" -> 17, "eighteen" -> 18, "nineteen" -> 19)
  val countTens = Map[String,Double](
    "twenty" -> 20, "thirty" -> 30, "forty" -> 40, "fifty" -> 50,
    "sixty" -> 60, "seventy" -> 70, "eighty" -> 80, "ninety" -> 90)
  val countPowers = Map[String,Double](
    "hundred" -> 100, "thousand" -> 1e3, "million" -> 1e6, "billion" -> 1e9, "trillion" -> 1e12)
  val counts = countOnes ++ countTeens ++ countTens ++ countPowers

  val ordinalOnes = Map[String,Double](
    "zeroth" -> 0, "first" -> 1, "second" -> 2, "third" -> 3, "fourth" -> 4,
    "fifth" -> 5, "sixth" -> 6, "seventh" -> 7, "eighth" -> 8, "ninth" -> 9)
  val ordinalTeens = Map[String,Double](
    "tenth" -> 10, "eleventh" -> 11, "twelfth" -> 12, "thirteenth" -> 13, "fourteenth" -> 14,
    "fifteenth" -> 15, "sixteenth" -> 16, "seventeenth" -> 17, "eighteenth" -> 18, "nineteenth" -> 19)
  val ordinalTens = Map[String,Double](
    "twentieth" -> 20, "thirtieth" -> 30, "fortieth" -> 40, "fiftieth" -> 50,
    "sixtieth" -> 60, "seventieth" -> 70, "eightieth" -> 80, "ninetieth" -> 90)
  val ordinalPowers = Map[String,Double](
    "hundredth" -> 100, "thousandth" -> 1e3, "millionth" -> 1e6, "billionth" -> 1e9, "trillion" -> 1e12)
  val ordinals = ordinalOnes ++ ordinalTeens ++ ordinalTens ++ ordinalPowers

  val NumPattern = """(-?\d+(\.\d+)?)""".r
  val StrPattern = """([a-z]+)""".r

  val StrStrPattern = (StrPattern+"""[\- ]"""+StrPattern).r
  val StrStrStrPattern = (StrPattern+"""[\- ]"""+StrPattern+" "+StrPattern).r
  val NumStrPattern = (NumPattern+" ?"+StrPattern).r

  val DivPattern = """(\d+)/(\d+)""".r
  val RankPattern = """(\d+)(st|nd|rd|th)""".r
  val CommaNumPattern = """(\d+),(\d\d\d)""".r

  val RangePattern = (NumPattern+" ?(-|to|and) ?"+NumPattern).r

  val Time1Pattern = """(\d+)(am|pm)?""".r
  val Time2Pattern = """(\d+) o'clock""".r

  def ordinalValue(x:Double) = NumValue(OrdinalDomain, x)
  def countValue(x:Double) = NumValue(CountDomain, x)
  def fracValue(x:Double) = NumValue(FracDomain, x)
  def timeValue(x:Double) = NumValue(DateTimeDomain, x)

  // Return possible measurement values.
  // Need to break up the extractors, otherwise get a compiler error:
  // [error] Error running compile: java.lang.Error: ch.epfl.lamp.fjbg.JCode$OffsetTooBigException: offset too big to fit in 16 bits: 55767
  // https://lampsvn.epfl.ch/trac/scala/ticket/1133
  def parse(s:String) : List[NumValue] = s.toLowerCase match {
    case "half" => fracValue(0.5) :: Nil
    case NumPattern(a,_) => countValue(a.toDouble)::Nil // e.g., 6
    case DivPattern(a,b) => fracValue(a.toDouble/b.toDouble)::Nil // e.g., 5/6
    case RankPattern(a,b) => ordinalValue(a.toInt)::Nil // 6th
    case CommaNumPattern(a,b) => countValue(a.toDouble*1000+b.toDouble)::Nil // 6,200
    case StrPattern(a) =>
      if (counts.contains(a)) countValue(counts(a))::Nil // e.g., six
      else if (ordinals.contains(a)) ordinalValue(ordinals(a))::Nil // e.g., sixth
      else Nil
    case StrStrPattern(a,b) =>
      if (countTens.contains(a) && countOnes.contains(b)) countValue(counts(a)+counts(b))::Nil // twenty-six
      else if (countTens.contains(a) && ordinalOnes.contains(b)) ordinalValue(counts(a)+ordinals(b))::Nil // twenty-sixth
      else if (counts.contains(a) && countPowers.contains(b)) countValue(counts(a)*counts(b))::Nil // six million
      else if (counts.contains(a) && ordinals.contains(b)) fracValue(1.0*counts(a)/ordinals(b))::Nil // six fortieth
      else if (b.endsWith("s")) {
        val c = b.slice(0, b.size-1)
        if (counts.contains(a) && ordinals.contains(c)) fracValue(1.0*counts(a)/ordinals(c))::Nil // six fortieths
        else Nil
      }
      else Nil
    case s => parse2(s)
  }
  private def parse2(s:String) : List[NumValue] = s match {
    case StrStrStrPattern(a,b,c) =>
      if (countTens.contains(a) && countOnes.contains(b) && countPowers.contains(c)) countValue((counts(a)+counts(b))*counts(c))::Nil // e.g., twenty-six million
      else Nil
    case NumStrPattern(a,_,b) =>
      if (countPowers.contains(b)) countValue(a.toDouble*counts(b))::Nil // e.g., 6 million
      else Nil
    case _ => Nil
  }

  def parseRange(s:String) : List[(Double,Double)] = s match {
    case RangePattern(a,_,_,b,_) => (a.toDouble, b.toDouble) :: Nil
    case _ => Nil
  }

  def parseTime(s:String) : List[NumValue] = s match {
    case Time1Pattern(a,b) => // e.g., 5, 5am, 530pm
      val n = a.toInt
      val t = if (n < 100) n*100 else n
      if (b == "am") timeValue(t) :: Nil
      else if (b == "pm") timeValue(t+1200) :: Nil
      else timeValue(t) :: timeValue(t+1200) :: Nil // Both
    case Time2Pattern(a) => // e.g., 5 o'clock
      val t = a.toInt*100
      timeValue(t) :: timeValue(t+1200) :: Nil
    case _ => Nil
  }

  def main(args:Array[String]) = {
    println(parse(args.mkString(" ")))
  }
}
