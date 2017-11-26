package dcs

import EntityTypes.Entity
import tea.Utils
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import scala.collection.mutable.HashMap
import java.io.File

/*
Essentially implements a crude non-recursive Datalog solver.
Does a bit of local query optimization.
Also handles special statements, which modify the world in other ways (such as updating the lexicon).
*/

class VarFactory(prefix:String) {
  var varId = 0
  def freshVar = { varId += 1; IdentItem(prefix+varId) }
}

class DatalogWorldInterpreter(world:World, verbose:Int) {
  val U = world.U

  def item2value(item:Item) : Value = item match {
    case NumItem(value) => NumValue(CountDomain, value)
    case StrItem(value) => NameValue(value)
    case IdentItem(name) => NameValue(name)
    case ListItem(elements) => { // Treat this as a set
      val pred = new ExplicitPredicate(1)
      elements.foreach { subitem =>
        pred += Entity(item2value(subitem))
      }
      SetValue(pred)
    }
    case _ => throw Utils.fails("Not an atomic value: %s", item)
  }

  def isVar(expr:ExprItem) = expr match {
    case IdentItem(s) if s(0) == '_' || s(0).isUpper => true
    case _ => false
  }
  def isVar(s:String) = s(0) == '_' || s(0).isUpper

  //type String = String
  /*type Env = HashMap[String,String]
  def getEnv(env:Env, name:String) = env.getOrElseUpdate(name, name)*/
  type Env = HashSet[String]
  def getEnv(env:Env, name:String) = { env += name; name }
  def envFreshVar(env:Env) = getEnv(env, "V"+env.size)

  def executeQuerySafe(expr:ExprItem) = {
    try { executeQuery(expr) }
    catch { case e:InterpretException => Predicate.error(e.message) }
  }
  def executeQuery(expr:ExprItem) : Predicate = expr match {
    case TermItem(funcName, specExpr :: rawBodyExpr :: Nil) if List("lambda", "answer").contains(funcName) => {
      val bodyExpr = flatten(rawBodyExpr)
      if (verbose >= 2) Utils.logs("QUERY %s", bodyExpr)
      val env = new Env
      val names = specExpr match {
        case IdentItem(name) => name :: Nil
        case ListItem(names) => names.map { case IdentItem(name) => name }
      }
      val labels = names.map(getEnv(env, _))
      val support = solve(env, bodyExpr, Support.init, names.toSet).perform.proj(labels)
      support.con
    }
    case TermItem(name, args) => {
      val (isPred, isFunc) = {
        if (HigherOrderFunctions.funcNames.contains(name)) (false, true)
        else {
          (world.containsPredicate(U.toPredName(name, args.size)),
           world.containsPredicate(U.toPredName(name, args.size+1)))
        }
      }
      //if (isPred && isFunc) throw Utils.fails("Can be either a predicate or a function, but don't know which: %s", name, expr)
      // Assume function by default
      if (isFunc) { // e.g. min(X1,...) => lambda(Q1,min(X1,...,Q1))
        val fact = new VarFactory("Q")
        val v = fact.freshVar
        executeQuery(TermItem("lambda", v :: TermItem(name, args ++ List(v)) :: Nil))
      }
      else if (isPred) // e.g., flight(X) => lambda([],flight(X)) [yes/no question]
        executeQuery(TermItem("lambda", ListItem(Nil) :: expr :: Nil))
      else throw Utils.fails("Not a predicate or a function of the right arity: %s in %s", name, expr)
    }
    case _ => throw Utils.fails("Unexpected: %s", expr)
  }

  def executeRule(rawRule:Rule) : Unit = {
    rawRule match {
      case Rule(target @ TermItem(name, args), null) => // Fact
        if (args.forall(!_.isInstanceOf[TermItem])) // Simple case
          return world.updatePredicate(U.toPredName(name, args.size), args.map(item2value))
        else {
          // Turn it into a rule
          /*val rule = flatten(fact.target) match {
            case AndItem((term:TermItem) :: extra) => Rule(term, AndItem(extra))
            case _ => throw Utils.fails("Bad: %s", fact.target)
          }
          executeRule(rule)*/
          return executeRule(Rule(target, AndItem(Nil)))
        }
      case _ =>
    }

    val rule = flatten(rawRule)
    val env = new Env
    if (verbose >= 2) Utils.logs("RULE %s", rule)

    // Target could consist of variables and values (e.g., f(A,3))
    var names = new ArrayBuffer[String]
    val args : List[Either[String,Value]] = rule.target.args.map {
      case IdentItem(name) if isVar(name) =>
        if (name == "_") throw Utils.fails("_ not allowed on the LHS of a rule")
        names += name
        Left(getEnv(env, name))
      case expr =>
        Right(item2value(expr))
    }

    // Solve the right hand side and projecting on to the variables on the left
    val labels = for (Left(label) <- args) yield label
    val fullSupport = solve(env, rule.source, Support.init, names.toSet).perform
    if (!labels.forall(fullSupport.labels.contains))
      throw Utils.fails("Executing %s failed: LHS labels not subset of RHS labels", rule)
    val support = fullSupport.proj(labels)
    val predName = U.toPredName(rule.target.name, rule.target.args.size)

    // Construct the final predicate by sticking the values (if any) in
    val finalPred = {
      if (args.forall(_.isInstanceOf[Left[String,Value]])) // All variables (use predicate as is)
        support.con.rename(predName)
      else { // Augment with values (if any)
        val pred = new ExplicitPredicate(args.size, predName)
        support.con.enumerate.perform.foreach { e =>
          pred += args.zipWithIndex.map {
            case (Left(label), i) => e(i)
            case (Right(x), i) => x
          }
        }
        pred
      }
    }
    world.updatePredicate(finalPred)
  }

  // Variables that can be accessible/constrained by the outside world
  // Note: this is different from the variables that are returned in an expression.
  def dependentVars(item:ExprItem) : Set[String] = item match {
    case TermItem(funcName, args) if HigherOrderFunctions.funcNames1.contains(funcName) || 
                                     HigherOrderFunctions.funcNames2.contains(funcName) => {
      args.last match {
        case IdentItem(varName) => Set(varName) // Only expose the last variable
        case _ => throw Utils.fails("Expected variable name in last position: %s", item)
      }
    }
    case TermItem(funcName, args) => dependentVars(args)
    case AndItem(args) => dependentVars(args)
    case OrItem(args) => dependentVars(args)
    case NegItem(arg) => dependentVars(arg)
    case IdentItem(varName) if isVar(varName) => if (varName == "_") Set() else Set(varName)
    case _:AtomItem => Set()
    case _ => throw Utils.fails("Unexpected: %s", item)
  }
  def dependentVars(args:List[ExprItem]) : Set[String] = {
    args.foldLeft(Set[String]()) { case (acc,arg) => acc ++ dependentVars(arg) }
  }

  // keepVarNames: variables that we must keep (other ones we can throw out); null means keep everything
  // Note that high-order functions are all computed inside their own environment (can't depend on external variables)
  def solve(env:Env, item:ExprItem, support:Support, keepVarNames:Set[String]) : Operation[Support] = {
    def keep(newSupport:Support) = {
      if (keepVarNames == null) newSupport
      else {
        val keepLabels = keepVarNames
        val labels = newSupport.labels.filter(keepLabels.contains)
        newSupport.proj(labels)
      }
    }
    def joinSupport(d_support:Support) = {
      val newSupport = (support join d_support).perform
      keep(newSupport)
    }

    if (IO.verbose >= 3) Utils.logs("solve env=%s, item=%s, support=%s, keep %s",
      env.mkString(","), item, support, if (keepVarNames == null) "all" else keepVarNames.mkString(","))
    item match {
      case TermItem(funcName, args) if HigherOrderFunctions.funcNames1.contains(funcName) => args match { // e.g., count(A,(major(A),city(A)),B)
        case IdentItem(mainVarName) :: expr :: IdentItem(outVarName) :: Nil =>
          new SimpleOperation[Support]({
            val newEnv = new Env
            val exprSupport = solve(newEnv, expr, Support.init, Set(mainVarName)).perform.proj(mainVarName::Nil) // Compute expr
            val (outType, outPred) : (Predicate,Predicate) = {
              if (funcName == "the") {
                //if (entities.size != 1) throw InterpretException("Uniqueness presupposition failed") // Too strict
                (exprSupport.abs, exprSupport.con)
              }
              else {
                val entities = exprSupport.con.enumerate.perform
                val (outType,outValue) : (Predicate,Value) = funcName match {
                  case "count" =>
                    (SingletonPredicate(CountDomain::Nil), NumValue(CountDomain, entities.size.toDouble))
                  case op => {
                    (exprSupport.abs, entities.foldLeft(null : NumValue) {
                      case (null, Entity(x:NumValue)) => x
                      case (acc:NumValue, Entity(x:NumValue)) => op match {
                        case "min" => acc min x
                        case "max" => acc max x
                        case _ => throw Utils.impossible
                      }
                      case _ => throw InterpretException(funcName + ": type error")
                    })
                  }
                }
                //if (out == null) throw InterpretException(funcName+" over empty set")
                (outType, {if (outValue == null) Predicate.empty(1) else SingletonPredicate(Entity(outValue)) })
              }
            }
            joinSupport(new Support(outType, outPred, getEnv(env, outVarName)::Nil))
          })
        case _ => throw Utils.fails("Invalid usage: %s", item)
      }
      case TermItem(funcName, IdentItem(mainVarName) :: arg :: TermItem(degFuncName, degArgs) :: IdentItem(outVarName) :: Nil) // argmax(A,state(A),population(A),B)
          if HigherOrderFunctions.funcNames2.contains(funcName) => {
        new SimpleOperation[Support]({
          // Reduce computing argmin/argmax to the problem of computing the complex superlative (not natural, but that code already exists).
          val fact = new VarFactory("D")
          val v = fact.freshVar // Variable that holds the degree
          val newEnv = new Env
          if (funcName == "sum") { // sum(A,state(A),population(A),B) => state(A),population(A,D), and then manually compute it
            val newExpr = AndItem(arg :: TermItem(degFuncName, degArgs++List(v)) :: Nil)
            val allSupport = solve(newEnv, newExpr, Support.init, Set(mainVarName,v.name)).perform.proj(mainVarName::v.name::Nil)
            var sum : NumValue = null
            allSupport.con.enumerate.perform.foreach {
              case arg :: (v:NumValue) :: Nil =>
                if (sum == null) sum = v
                else sum = sum + v
              case _ => throw Utils.impossible
            }
            if (sum == null) throw InterpretException("sum over empty set")
            joinSupport(new Support(allSupport.abs.proj(List(1)), SingletonPredicate(Entity(sum)), getEnv(env, outVarName)::Nil))
          }
          else {
            val newFuncName = funcName match {
              case "argmax" => "largest"
              case "argmin" => "smallest"
              case _ => throw Utils.fails("Unexpected: %s", funcName)
            }
            val newExpr = TermItem(newFuncName, v :: AndItem(arg :: TermItem(degFuncName, degArgs++List(v)) :: Nil) :: Nil)
            val newSupport = solve(newEnv, newExpr, support, Set(mainVarName)).perform.proj(mainVarName::Nil) // Solve and extract varName
            joinSupport(new Support(newSupport.abs, newSupport.con, getEnv(env, outVarName)::Nil)) // Dump it into outVarName
          }
        })
      }
      case TermItem(funcName, args) if HigherOrderFunctions.mostFewFuncNames.contains(funcName) => { // e.g., most(A,B,(state(A),loc(B,A)))
        new SimpleOperation[Support](args match {
          case IdentItem(mainVarName) :: IdentItem(subVarName) :: expr :: Nil if isVar(mainVarName) && isVar(subVarName) =>
            val newEnv = new Env
            val exprSupport = solve(newEnv, expr, Support.init, Set(mainVarName, subVarName)).perform.proj(mainVarName::subVarName::Nil)
            val entities = exprSupport.con.enumerate.perform
            val a2bs = new HashMap[Value,HashSet[Value]] // a -> set of b's
            entities.foreach {
              case Entity(a,b) => a2bs.getOrElseUpdate(a, new HashSet[Value]) += b
              case x => throw Utils.fails("Got non-pair inside the set: %s", x)
            }

            // Add nulls: (a,emptyset) : a satisfying predicates that only involve a (hack)
            expr match {
              case AndItem(args) =>
                val restrictExpr = AndItem(args.filter(arg => dependentVars(arg) == Set(mainVarName)))
                solve(newEnv, restrictExpr, Support.init, Set(mainVarName)).perform.proj(mainVarName::Nil).con.enumerate.perform.foreach {
                  case Entity(a) =>
                    if (!a2bs.contains(a)) a2bs(a) = new HashSet[Value]
                }
              case _ =>
            }

            val max = funcName match {
              case "most" => true
              case "fewest" => false
              case _ => throw Utils.impossible
            }
            val bestPred = SupPredicate.computeSup(max, a2bs.map {
              case (a,bs) => Entity(a, NumValue(CountDomain, bs.size))
            })
            joinSupport(new Support(exprSupport.abs.proj(List(0)), bestPred, getEnv(env, mainVarName)::Nil))
          case _ => throw Utils.fails("Invalid usage: %s", item)
        })
      }
      case TermItem(supName, args) if HigherOrderFunctions.superlatives.contains(supName) => { // highest(A,(mountain(A),loc(A,B),const(B,stateid(mississippi))))
        new SimpleOperation[Support](args match {
          case IdentItem(mainVarName) :: expr :: Nil if isVar(mainVarName) =>
            val (mode, degree) = HigherOrderFunctions.superlatives(supName)
            // Find all solutions of the expr (note that it's solved in isolation)
            var exprSupport = solve(env, expr, Support.init, null).perform
            var entities = exprSupport.con.enumerate.perform

            if (entities.size == 0)
              (support join new Support(exprSupport.abs, Predicate.empty(exprSupport.arity), exprSupport.labels)).perform
            else {
              // There are two cases
              //   mainVarName is the actual degree (if it's a number)
              //   mainVarName is the object, and we need to apply a degree
              val mainVarName_i = exprSupport.labels.indexOf(mainVarName)
              if (mainVarName_i == -1) throw Utils.fails("%s not used in %s", mainVarName, expr)

              var degreeLabel : String = null
              if (entities.head(mainVarName_i).isInstanceOf[NumValue]) // See if mainVarName corresponds to a number
                degreeLabel = mainVarName // Yes, that's the degree
              else {
                // No, need to apply degree
                degreeLabel = envFreshVar(env)
                val degPred = world.getPredicate(U.toPredName(degree, 2))
                exprSupport = (exprSupport join new Support(degPred.abs, degPred, List(mainVarName, degreeLabel))).perform
                entities = exprSupport.con.enumerate.perform
              }

              // Now, loop over all the configurations, and choose the ones that have the minimum/maximum according to degreeLabel
              val degree_i = exprSupport.labels.indexOf(degreeLabel)
              var bestPred = new ExplicitPredicate(exprSupport.arity)
              var bestDegree = -1.0/0
              val sign = {
                if (mode == "max") 1.0
                else if (mode == "min") -1.0
                else throw Utils.impossible
              }
              entities.foreach { e =>
                val degree = e(degree_i).asInstanceOf[NumValue].value * sign
                if (degree == bestDegree)
                  bestPred += e
                else if (degree > bestDegree) {
                  bestDegree = degree
                  //bestPred.clear
                  bestPred = new ExplicitPredicate(exprSupport.arity)
                  bestPred += e
                }
              }

              joinSupport(new Support(exprSupport.abs, bestPred, exprSupport.labels))
            }
          case _ => throw Utils.fails("Invalid usage: %s", item)
        })
      }
      case TermItem(name, args) => { // Normal predicate: e.g., f(A,B,_,3)
        // Arguments are either variables (kept), wild cards (marginalized away), or values (selected)
        val predName = U.toPredName(name, args.size)
        var pred = world.getPredicate(predName)
        if (pred == null) throw Utils.fails("Predicate not found: %s", predName)
        val abs = pred.abs
        var allLabels : List[String] = Nil
        var selIndices : List[Int] = Nil
        var sel_e : Entity = Nil
        args.zipWithIndex.reverse.foreach {
          case (IdentItem(name), i) if isVar(name) => {
            if (name == "_")
              allLabels = envFreshVar(env) :: allLabels
            else
              allLabels = getEnv(env, name) :: allLabels
          }
          case (expr, i) => {
            allLabels = envFreshVar(env) :: allLabels
            selIndices = i :: selIndices
            sel_e = item2value(expr) :: sel_e
          }
        }
        require (sel_e.size == selIndices.size && allLabels.size == args.size)
        if (selIndices.size > 0) pred = pred.select(selIndices, sel_e).perform
        val op = support join new Support(abs, pred, allLabels)
        op.extendLinear(keep)
      }
      case AndItem(args) => new SimpleOperation[Support]({
        // Go through all the term items and iteratively choose the cheapest one to join
        // Don't re-order past disjunction or negation

        var remArgs = args
        var newSupport = support
        def newKeepVarNames(arg:ExprItem) = {
          if (keepVarNames == null) null
          else keepVarNames ++ dependentVars(remArgs.filter(_ != arg)) 
        }
        while (remArgs != Nil) {
          val choiceArgs = remArgs.takeWhile { arg => arg.isInstanceOf[TermItem] } // Only choose among term items
          if (choiceArgs == Nil) { // No choice but to take the next arg
            val arg = remArgs.head
            newSupport = solve(env, arg, newSupport, newKeepVarNames(arg)).perform
            remArgs = remArgs.tail
          }
          else {
            if (IO.verbose >= 4)
              Utils.logs("=== %s choices", choiceArgs.size)
            val ops = choiceArgs.map { arg => // Here are my choices
              val op = solve(env, arg, newSupport, newKeepVarNames(arg))
              if (IO.verbose >= 4)
                Utils.logs("  COST %s %s %s", arg, op.retCost, op.outCost) // Here are my choices
              op
            }
            // Choose the operation with the lowest output cost, but favor those that have some variable overlap with support
            val best_i = Utils.argmin(ops.size, { i:Int =>
              val argLabels = dependentVars(choiceArgs(i))
              val overlaps = newSupport.labels.exists(argLabels.contains)
              //Utils.dbgs("  OVERLAP %s : %s & %s => %s", choiceArgs(i), argLabels, newSupport.labels, overlaps)
              ops(i).outCost + {if (overlaps) 0.0 else 1e10}
            })
            val cost = ops(best_i).outCost
            if (cost > IO.maxCost) throw InterpretException("Cost "+cost+" exceeds maximum of "+IO.maxCost)
            newSupport = ops(best_i).perform
            remArgs = remArgs.slice(0, best_i) ++ remArgs.slice(best_i+1, remArgs.size)
          }
        }
        newSupport
      })
      case OrItem(args) => new SimpleOperation[Support]({
        // For each arg, try to extend it
        val newKeepVarNames = {
          if (keepVarNames == null) null
          else keepVarNames ++ args.tail.foldLeft(dependentVars(args.head)) { case (acc,arg) =>
            acc & dependentVars(arg)
          }
        }
        val argSupports = args.map { arg => solve(env.clone, arg, support, newKeepVarNames).perform }
        // Project onto common labels
        val labels = argSupports(0).labels.filter { label => argSupports.forall(_.labels.contains(label)) }
        val abs = new ExplicitPredicate(labels.size)
        val con = new ExplicitPredicate(labels.size, argSupports.map(_.con).mkString("|"))
        argSupports.foreach { support =>
          val projSupport = support.proj(labels)
          abs ++= projSupport.abs
          con ++= projSupport.con.enumerate.perform
        }
        new Support(abs, con, labels)
      })
      case NegItem(arg) => new SimpleOperation[Support]({
        // Extend arg
        val negPred = solve(env.clone, arg, support, null).perform.proj(support.labels).con
        // Exclude all entities in negPred
        val out = new ExplicitPredicate(support.arity)
        support.con.enumerate.perform.foreach { e =>
          if (!negPred.contains(e).perform) out += e
        }
        keep(new Support(support.abs, out, support.labels))
      })
      case _ => throw Utils.fails("Unknown case: %s", item)
    }
  }

  def flatten(rule:Rule) = {
    val newRule = flattenFunctions(rule)
    new Rule(newRule.target, flattenConjunctions(newRule.source))
  }
  def flatten(expr:ExprItem) = flattenConjunctions(flattenFunctions(expr, "_E"))

  def flattenConjunctions(expr:ExprItem) : ExprItem = expr match {
    case item:AtomItem => item
    case TermItem(name, args) => TermItem(name, args.map(flattenConjunctions))
    case OrItem(args) => OrItem(args.map(flattenConjunctions))
    case NegItem(arg) => NegItem(flattenConjunctions(arg))
    case AndItem(args) => {
      AndItem(args.flatMap { arg =>
        flattenConjunctions(arg) match {
          case AndItem(subArgs) => subArgs // Flatten
          case item => item::Nil
        }
      })
    }
  }

  def flattenFunctions(rule:Rule) : Rule = {
    val newTarget = flattenFunctions(rule.target, "_T")
    val newSource = flattenFunctions(rule.source, "_S")
    newTarget match {
      case AndItem((term:TermItem) :: extra) => // Move extra predicates to right-hand side
        Rule(term, AndItem(newSource :: extra))
      case newTarget if newTarget == rule.target => Rule(rule.target, newSource)
      case _ => throw Utils.fails("Bad: %s", newTarget)
    }
  }

  // p(f(A,g(B)),C) => p(Z1,C), f(A,Z2,Z1), g(B,Z2)
  def flattenFunctions(expr:ExprItem, varPrefix:String) : ExprItem = {
    val fact = new VarFactory(varPrefix)

    def combine(pair:(ExprItem,List[ExprItem])) : ExprItem = {
      val (arg, extraArgs) = pair
      if (extraArgs == Nil) arg
      else AndItem(arg :: extraArgs)
    }

    // Return an expression and a list of expressions that need to be conjoined.
    def convert(expr:ExprItem, retSimple:Boolean) : (ExprItem,List[ExprItem]) = expr match {
      case _:AtomItem => (expr, Nil)
      case TermItem(funcName, args) => {
        val (newArgs, extraArgs) : (List[ExprItem],List[ExprItem]) = {
          if (HigherOrderFunctions.funcNames.contains(funcName)) // Each argument is self-contained
            (args.map { arg => combine(convert(arg, false)) }, Nil)
          else { // Extra args are pooled
            val extraArgs = new ArrayBuffer[ExprItem]
            val newArgs = args.map { arg =>
              val (newArg, d_extraArgs) = convert(arg, true)
              extraArgs ++= d_extraArgs
              newArg
            }
            (newArgs, extraArgs.toList)
          }
        }
        if (retSimple) {
          // Need to create a variable
          val v = fact.freshVar
          (v, TermItem(funcName, newArgs ++ List(v)) :: extraArgs)
        }
        else
          (TermItem(funcName, newArgs), extraArgs)
      }
      case NegItem(arg) => (NegItem(combine(convert(arg, false))), Nil)
      case AndItem(args) => (AndItem(args.map{arg => combine(convert(arg, false))}), Nil)
      case OrItem(args) => (OrItem(args.map{arg => combine(convert(arg, false))}), Nil)
      case _ => throw Utils.fails("Unknown: %s", expr)
    }

    //dbgs("%s => %s", expr, combine(convert(expr, false)))

    combine(convert(expr, false))
  }
}

class DatalogInterpreter(U:Universe) {
  var verbose = 0

  def currWorldInt = new DatalogWorldInterpreter(U.currWorld, verbose)

  def toString(item:Item) = item match {
    case IdentItem(name) => name
    case StrItem(value) => value
    case NumItem(value) => Utils.fmts("%s", value)
    case _ => throw Utils.fails("Cannot convert to string: %s", item)
  }

  def invalidUsage(x:Any) = throw Utils.fails("Invalid usage: %s", x)

  def addLex(phraseItem:ExprItem, predItem:ExprItem, statement:StatementItem) = {
    def handle(str:String) = predItem match {
      case StrItem(predName) => U.str2predicateHandle(predName)
      case TermItem("answer", args) => // answer(A, ...)
        PredicateConstant(currWorldInt.executeQuery(predItem).rename(str))
      case TermItem(name, args) => // stateid('oregon')
        val pred = U.currWorld.getPredicate(U.toPredName(name, args.size+1))
        val resultPred = pred.select(MyUtils.IntList(args.size), args.map(currWorldInt.item2value)).perform.proj(args.size::Nil)
        PredicateConstant(resultPred)
      case _ => invalidUsage(statement)
    }
    phraseItem match {
      case StrItem(str) => U.addLexicalEntry(str, handle(str))
      case ListItem(strItems) =>
        strItems.foreach {
          case StrItem(str) => U.addLexicalEntry(str, handle(str))
          case _ => invalidUsage(statement)
        }
      case _ => invalidUsage(statement)
    }
  }

  def processStatement(path:String, statement:StatementItem) : Unit = {
    statement match {
      case Rule(TermItem(name, args), source) if name.startsWith("_") =>
        if (source == null || currWorldInt.executeQuery(source).size > 0)
          processSpecialStatement(path, statement, name, args)
      case rule:Rule => currWorldInt.executeRule(rule)
      case _ => Utils.logs("Example FAILED TO PARSE")
    }
  }

  def processSpecialStatement(path:String, statement:StatementItem, name:String, args:List[ExprItem]) : Unit = name match {
    case "_logStatus" => U.log
    case "_include" => args match { // Load another file
      case StrItem(newPath) :: Nil =>
        process(if (newPath.startsWith("/")) newPath else (new java.io.File(path)).getParent+"/"+newPath, true)
      case _ => invalidUsage(statement)
    }
    case "_rewrite" => args match { // Create rewrite rule
      case StrItem(source) :: StrItem(target) :: Nil => U.addRewrite(source, target)
      case _ => invalidUsage(statement)
    }
    case "_lex" => args match { // Lexical entry
      case phraseItem :: ListItem(predItems) :: Nil => predItems.foreach{predItem => addLex(phraseItem, predItem, statement)}
      case phraseItem :: predItem :: Nil => addLex(phraseItem, predItem, statement)
      case _ => invalidUsage(statement)
    }
    case "_world" => args match { // Set the current world
      case StrItem(id) :: Nil => U.currWorld = U.getWorld(id)
      case _ => invalidUsage(statement)
    }
    case "_stdPred" => {
      def addStd(stdName:String, name:String) = {
        var pred = StandardLibrary.getPredicate(stdName, null) // Create the predicate just to get its arity
        pred = StandardLibrary.getPredicate(stdName, U.toPredName(name, pred.arity)) // Attach the arity
        U.currWorld.addPredicate(pred)
      }
      args match { // Create a predicate
        case StrItem(stdName) :: Nil => addStd(stdName, stdName)
        case StrItem(stdName) :: StrItem(name) :: Nil => addStd(stdName, name)
        case _ => invalidUsage(statement)
      }
    }
    case "_constructPred" => args match { // Create a predicate that creates a type (name, number of arguments)
      case StrItem(name) :: NumItem(value) :: Nil =>
        val k = value.toInt
        U.currWorld.addConstructPredicate(U.toPredName(name, k+1), k)
      case _ => invalidUsage(statement)
    }
    case "_expandPred" => args match { // Create a predicate from that turns count into repeated values of a domain
      case StrItem(countName) :: StrItem(expandedName) :: StrItem(domName) :: Nil =>
        val dom = TypedDomain(U.toPredName(domName, 1))
        //val expandedPred = new ExpandedPredicate(U.toPredName(expandedName, 2), dom, U.currWorld.getPredicate(U.toPredName(countName, 2)))
        val expandedPred = Predicate.expand(U.currWorld.getPredicate(U.toPredName(countName, 2)), dom).rename(U.toPredName(expandedName, 2))
        U.currWorld.addPredicate(expandedPred)
      case _ => invalidUsage(statement)
    }
    case _ => throw Utils.fails("Unknown command %s in %s", name, statement)
  }

  def process(path:String, continue: =>Boolean) : Unit = {
    val statements = Utils.track("Parsing %s", path) {
      if ((new File(path)).exists)
        DatalogParser.load(path)
      else {
        Utils.warnings("File doesn't exist: %s", path)
        Nil
      }
    }
    Utils.logs("%s statements", statements.size)
    statements.foreach { statement =>
      if (!continue) return
      if (verbose >= 2) Utils.logs("========= %s", statement)
      processStatement(path, statement)
    }
  }
}
