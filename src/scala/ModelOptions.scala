package dcs

class ModelOptions {
  import tea.OptionTypes._
  @Option(gloss="Of the form x=3, interpreted as x(3).") var dlogOptions : Array[String] = Array()
  @Option(gloss="Features to use for learning") var features = Array[String]()
  @Option var useSyntax = true
  @Option var verbose = 0
  @Option var pruneEmptyDen = false
  @Option var pruneErrDen = false
  @Option(gloss="The final denotation of a sentence cannot be infinite") var pruneInfDen = false
  @Option var forceRightBranching = false
  @Option var forceParseSyntax = false
  @Option var beamSize = 100
  @Option var timeout = 0

  @Option(gloss="In interactive mode, whether to automatically learn") var autoUpdate = false
  @Option(gloss="Bayesian averaging to produce result") var useBayesianAveraging = false

  // Construction of trees
  @Option(gloss="Allow extraction in other places besides right before execute (if false, just enough to do simple superlatives/quantification)") var allowDelayedExtraction = false
  @Option(gloss="Allow insertion of implicit bridge predicates between arity 1 and arity k>1") var implicit1k = false
  @Option(gloss="Allow insertion of implicit bridge predicates between arity k>1 and arity 1") var implicitk1 = false
  @Option var allowNonProjectivity = false
  @Option(gloss="Try attaching things under the bridge") var allowTroll = false
  @Option(gloss="Only use POS predicates when nothing else matches") var usePosOnlyIfNoMatches = false
  @Option(gloss="Only use POS predicates when nothing matches in phraseMap") var usePosOnlyIfNoPhraseMatches = false

  @Option(gloss="Dictionary (word/POS pairs)") var posTagDictPath : String = null
  @Option(gloss="Add items to the lexicon each iteration") var learnLexicon = false
  @Option(gloss="Remove word-predicate entries from the lexicon with posterior probability less than this") var lexiconPruneThreshold = 1e-4

  @Option(gloss="Old version of attach") var oldAttach = false
  @Option(gloss="more long than => long more than") var hackComparatives = false

  // Preprocessing of sentences
  @Option var lexToName = false
  @Option var lexToSetWithName = false
  @Option var lexToObjWithName = true
  @Option var lexToTime = false
  @Option var normPhrasePrefixLen = Integer.MAX_VALUE

  @Option var inParamsPath : String = null
  @Option var outParamsPath : String = null
  @Option var outBasketsPath : String = null
}

object MO extends ModelOptions
