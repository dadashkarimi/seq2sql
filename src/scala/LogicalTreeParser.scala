package dcs

/*
Interpret files containing a sequence of commands (each command is a tree with LISP-style syntax)
for building a DCS tree (not really used because all our DCS trees are latent).

node ::= (<predicate>:<word> <rel> <node> ... <rel> <node>)
rel  ::= ++ | <i>-<j> | X | E | C | Q
*/

import scala.collection.mutable.HashMap
import tea.Utils
import scala.collection.mutable.ArrayBuffer
import EntityTypes.Entity

class LogicalTreeParser(U:Universe) {
  var on_? = true
  var verbose = 0

  def log = U.log

  // Instruction 
  class InstProcessor(inst:Array[Any]) {
    def strInst(i:Int) = inst(i).asInstanceOf[String]
    def arrInst(i:Int) = inst(i).asInstanceOf[Array[Any]]
    def hashInst(queryKey:String) = {
      var foundValue : Any = null
      inst.foreach {
        case Array(key, value) if key == queryKey => foundValue = value
        case _ =>
      }
      foundValue
    }
    def hashStrInst(queryKey:String) = hashInst(queryKey).asInstanceOf[String]
    def hashArrInst(queryKey:String) = hashInst(queryKey).asInstanceOf[Array[Any]]

    def process : Unit = inst.head match {
      case "verbose" => verbose = strInst(1).toInt
      case "int.verbose" => IO.verbose = strInst(1).toInt
      case "model.verbose" => MO.verbose = strInst(1).toInt
      case "log" => Utils.track("Universe") { log }
      case _ => throw Utils.fails("Invalid instruction: %s", Renderer.renderTree(inst))
    }
  }

  def newInstProcessor(inst:Array[Any]) = new InstProcessor(inst) // Override if desired

  def continue = true

  def readScript(inPath:String) = Utils.track("Reading %s", inPath) {
    TSU.foreachNode(inPath, { _inst:Any =>
      if (continue) {
        if (verbose >= 6) Utils.begin_track("Processing: %s", Renderer.renderTree(_inst))
        if (!_inst.isInstanceOf[Array[Any]])
          throw Utils.fails("Not an instruction: %s", Renderer.renderTree(_inst))
        val inst = _inst.asInstanceOf[Array[Any]]
        if (inst.size == 0) throw Utils.fails("Empty instruction")

        if (inst.head == "process") {
          inst(1) match {
            case "on" => on_? = true
            case "off" => on_? = false
            case _ => throw Utils.fails("Expected on|off")
          }
        }
        else if (on_?) {
          try {
            newInstProcessor(inst).process
          } catch {
            case e =>
              Renderer.logTree("Problem with:", inst)
              throw e
          }
        }
        if (verbose >= 6) Utils.end_track
        true
      }
      else
        false
    })
  }

  def parsePredicate(s:String) : List[Predicate] = {
    if (s == ".") return Nil
    NumberParser.parse(s) match {
      case v :: Nil => return SingletonPredicate(Entity(v)) :: Nil
      case _ =>
    }
    TypedValue.parse(s) match {
      case Some(v) => return SingletonPredicate(Entity(v)) :: Nil
      case None => 
    }
    U.currWorld.getPredicate(s) match {
      case pred if pred != null => return pred :: Nil
      case _ =>
    }
    throw Utils.fails("Unable to parse predicate from %s", s)
  }

  // Return the states (most recent first)
  def buildNode(tree:Any) : Node = tree match {
    case tree:Array[Any] =>
      if (tree.size % 2 != 1) throw Utils.fails("Expected (<predicate> <rel> <node> ...), but got %s", Renderer.renderTree(tree))
      var i = 1
      var edges : List[Edge] = Nil
      while (i < tree.size) {
        val e = Edge(Rel.parse(tree(i).asInstanceOf[String]), buildNode(tree(i+1)))
        edges = e :: edges
        i += 2
      } 
      Node.create(parsePredicate(tree(0).asInstanceOf[String]), edges.reverse)
    case s:String => buildNode(Array(s))
    case _ => throw Utils.fails("Expected string or array, but got %s", Renderer.renderTree(tree))
  }
}
