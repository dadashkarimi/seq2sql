package dcs

import scala.collection.mutable.HashMap
import tea.Utils
import EntityTypes.Entity

// Convert lambda calculus (Luke's format) to Datalog.

object HigherOrderFunctions {
  val superlatives = Map( // SPECIFIC
    "highest" -> ("max", "elevation"), "lowest" -> ("min", "elevation"),
    "longest" -> ("max", "len"), "shortest" -> ("min", "len"),
    "largest" -> ("max", "size"), "smallest" -> ("min", "size")
  )
  val funcNames1 = List("count", "the", "min", "max")
  val funcNames2 = List("argmin", "argmax", "sum")
  val mostFewFuncNames = List("most", "fewest")
  val lambdaFuncNames = List("answer", "lambda")
  val funcNames = Set[String]() ++ superlatives.keys ++ funcNames1 ++ funcNames2 ++ mostFewFuncNames ++ lambdaFuncNames
}

trait LFormula {
  def toStrTree : Any
  override def toString = Renderer.renderTree(toStrTree)
}
case class LConstant(value:Any) extends LFormula { override def toStrTree = value.toString }
case class LVariable(name:String) extends LFormula { override def toStrTree = name }

case class LNegation(arg:LFormula) extends LFormula { override def toStrTree = Array("not", arg.toStrTree) }
case class LConjunction(args:List[LFormula]) extends LFormula { override def toStrTree = Array("and") ++ args.map(_.toStrTree) }
case class LDisjunction(args:List[LFormula]) extends LFormula { override def toStrTree = Array("or") ++ args.map(_.toStrTree) }
case class LExists(varName:String, arg:LFormula) extends LFormula { override def toStrTree = Array("exists", varName, arg.toStrTree) }
case class LFunction(name:String, args:List[LFormula]) extends LFormula { override def toStrTree = Array(name) ++ args.map(_.toStrTree) }
case class LAbstraction(funcName:String, varNames:List[String], args:List[LFormula]) extends LFormula {
  override def toStrTree = Array(funcName, varNames.toArray) ++ args.map(_.toStrTree)
}

class LambdaCalculusParser(U:Universe) {
  val abbrevMap = {
    val map = new HashMap[String,String]
    val pred = U.currWorld.getPredicate(U.toPredName("domAbbrev", 2))
    pred.enumerate.perform.foreach {
      case Entity(NameValue(abbrev), NameValue(predName)) =>
        map(abbrev) = predName
    }
    map
  }

  def parse(node:Any) = {
    def recurse(env:Set[String], node:Any) : LFormula = {
      val f = node match {
        case s:String => {
          if (env.contains(s)) new LVariable(s)
          else {
            s.split(":") match {
              case Array(name, domAbbrev) => // SPECIFIC: denver:ci => cityid('denver')
                if (domAbbrev == "i")
                  LConstant(name.toDouble)
                else {
                  val value = {
                    try { name.toDouble }
                    catch { case _ => name }
                  }
                  LFunction(abbrevMap(domAbbrev), LConstant(value)::Nil)
                }
              case _ =>
                new LConstant(s)
            }
          }
        }
        case a:Array[Any] => a.toList match {
          case "not" :: expr :: Nil => new LNegation(recurse(env, expr))
          case "and" :: args => new LConjunction(args.map {arg => recurse(env, arg)})
          case "or" :: args => new LDisjunction(args.map {arg => recurse(env, arg)})
          case "exists" :: (varName:String) :: expr :: Nil => new LExists(varName, recurse(env + varName, expr))
          case "lambda" :: (varName1:String) :: t :: expr :: Nil =>
            expr match {
              case Array("lambda", (varName2:String), t, expr) =>
                expr match {
                  case Array("lambda", (varName3:String), t, expr) =>
                    expr match {
                      case Array("lambda", (varName4:String), t, expr) =>
                        new LAbstraction("lambda", varName1::varName2::varName3::varName4::Nil, recurse(env+varName1+varName2+varName3+varName4, expr)::Nil)
                      case _ =>
                        new LAbstraction("lambda", varName1::varName2::varName3::Nil, recurse(env+varName1+varName2+varName3, expr)::Nil)
                    }
                  case _ =>
                    new LAbstraction("lambda", varName1::varName2::Nil, recurse(env+varName1+varName2, expr)::Nil)
                }
              case _ =>
                new LAbstraction("lambda", varName1::Nil, recurse(env+varName1, expr)::Nil)
            }
          case (funcName:String) :: (varName:String) :: args if HigherOrderFunctions.funcNames.contains(funcName) =>
            new LAbstraction(funcName, varName::Nil, args.map{arg => recurse(env + varName, arg)})
          case (rawName:String) :: args => {
            val name = rawName.replaceAll(":.+$", "") // SPECIFIC: airline:e => airline
            new LFunction(name, args.map(recurse(env, _)))
          }
          case _ => throw Utils.fails("Invalid: %s", Renderer.renderTree(node))
        }
        case _ => throw Utils.fails("Invalid: %s", Renderer.renderTree(node))
      }
      f
    }
    recurse(Set[String](), node)
  }

  def to_dlog(formula:LFormula) = {
    val fact = new VarFactory("X")
    // Note that we maintain an environment from lambda calculus variables to (unique) Datalog variables,
    // because Datalog variables have more global scope than in lambda calculus.
    // In particular, everything in existential variables are treated the same.
    def convert(env:Map[String,IdentItem], formula:LFormula) : ExprItem = formula match {
      case LConstant(value) => value match {
        case value:String => StrItem(value)
        case value:Double => NumItem(value)
        case _ => throw Utils.fails("Bad: %s", value)
      }
      case LVariable(name) => env(name)
      case LExists(name, arg) => convert(env.updated(name, fact.freshVar), arg)
      case LAbstraction(funcName, names, args) => {
        var newEnv = env
        val vars = names.map { name =>
          val v = fact.freshVar
          newEnv = newEnv.updated(name, v)
          v
        }
        val formalSpec = {
          if (vars.size == 1) vars.head
          else ListItem(vars)
        }
        TermItem(funcName, formalSpec :: args.map { arg => convert(newEnv, arg) })
      }
      case LConjunction(args) => AndItem(args.map { arg => convert(env, arg) })
      case LDisjunction(args) => OrItem(args.map { arg => convert(env, arg) })
      case LNegation(arg) => NegItem(convert(env, arg))
      case LFunction(name, args) => TermItem(name, args.map { arg => convert(env, arg) })
      case _ => throw Utils.fails("Bad: %s", formula)
    }

    //dbgs("LAMBDA %s", formula)
    val result = convert(Map[String,IdentItem](), formula)
    //dbgs("DLOG %s", result)
    result
  }
}
