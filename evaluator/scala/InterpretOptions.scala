package dcs

import java.util.Random

class InterpretOptions {
  import tea.OptionTypes._
  @Option var verbose = 0
  @Option var displayMaxSetSize = 1
  @Option var displayTypes = false
  @Option var displayDens = false
  @Option var displaySpans = false
  @Option var prettyMaxSetSize = 10
  @Option var crashOnTypeError = false
  @Option(gloss="Throw an error if cost exceeds this") var maxCost = 100000
  @Option var random = new Random(1)
  @Option var roundNumDigits = Integer.MAX_VALUE

  def displayAbsCon(isCon:Boolean) = (displayTypes && !isCon) || (displayDens && isCon)
}

object IO extends InterpretOptions
