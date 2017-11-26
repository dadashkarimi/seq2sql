package dcs

import fig.exec.Execution
import tea.Utils

/*
Main entry point to the DCS system.
*/
object NuggetLearn extends Runnable {
  import tea.OptionTypes._
  @Option var debug = false
  @Option var interactive = false
  @Option var startServer = false
  @Option var launchParser = false

  val U = new Universe
  val DM = new DataManager(U)

  val EqPattern = """(\w+)=(.+)""".r

  def run = {
    // Incorporate preliminary Datalog options
    val dlog = new DatalogInterpreter(U) 
    MO.dlogOptions.foreach {
      case EqPattern(key,value) =>
        val valueItem = try { NumItem(value.toDouble) } catch { case _ => StrItem(value) }
        dlog.processStatement(null, Rule(TermItem(key, valueItem::Nil), null))
      case s => throw Utils.fails("Bad format, expected <key>=<value>: %s", s)
    }
    DM.loadExamples
  }

  def main(args:Array[String]) = {
    Execution.run(args, "main", this, "int", IO, "learn", LO, "model", MO, "data", DM, "server", SO, "parser", PO)
  }
}
