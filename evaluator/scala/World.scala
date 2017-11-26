package dcs

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashSet
import scala.collection.mutable.LinkedHashMap
import PhraseTypes._
import EntityTypes._
import tea.Utils

class World(val id:String, val parent:World, val U:Universe) { // A model/database
  override def toString = id+":"+predicates.size

  // Interpretation function: predicate name -> predicate
  private val predicates = new LinkedHashMap[String,Predicate]

  def getPredicates = predicates.values

  def getPredicatesOfPhrase(phrase:Phrase) : List[Predicate] = {
    val nameObjPred = getPredicate(U.toPredName("nameObj", 2))
    if (nameObjPred == null) {
      Utils.warnings("nameObj/2 not defined, can't match any constants from world")
      return Nil
    }
    val np = U.normPhrase(phrase)
    U.nameMap.getOrElse(np, new HashSet[String]).toList.flatMap { name:String =>
      val outPred = nameObjPred.select(List(0), List(NameValue(name))).perform.proj(List(1))
      //Utils.dbgs("GET %s %s %s", phrase, np, outPred)
      if (outPred.size == 0) Nil
      else {
        val preds = new ArrayBuffer[Predicate]
        if (MO.lexToName)
          preds += SingletonPredicate(Entity(NameValue(name))) // (1) name
        if (MO.lexToSetWithName) {
          if (outPred.size > 1)
            preds += outPred.rename("{"+name+"}") // (2) set of all objects with that name
        }
        if (MO.lexToObjWithName) {
          outPred.enumerate.perform.foreach { e =>
            preds += SingletonPredicate(e) // (3) an object with that name
          }
        }
        preds.toList
      }
    }
  }

  def getPredicate(name:String) : Predicate = predicates.getOrElse(name, {
    if (parent == null) null
    else parent.getPredicate(name)
  })
  def getExplicitPredicate(name:String) = getPredicate(name).asInstanceOf[ExplicitPredicate]

  def containsPredicate(name:String) : Boolean = predicates.contains(name) || (parent != null && parent.containsPredicate(name))

  def handle2pred(handle:PredicateHandle) : Predicate = handle match {
    case PredicateName(name) => getPredicate(name)
    case PredicateConstant(pred) => pred
  }

  def updatePredicate(name:String, e:Entity) = {
    //Utils.logs("KB: %s += %s", name, e)
    def createNew = {
      val pred = new ExplicitPredicate(e.size, name)
      pred.createIndex
      pred
    }
    U.extractValues(e)
    predicates.getOrElseUpdate(name, createNew).asInstanceOf[ExplicitPredicate] += e
  }

  def updatePredicate(pred:Predicate) = {
    if (IO.verbose >= 1) Utils.logs("Update: %s (%s): %s", pred.render, pred.size, pred.abs.render)
    if (IO.verbose >= 10) {
      val op = pred.enumerate
      if (op.possible) Utils.track_printAll("%s:", pred) {
        op.perform.foreach { e => Utils.logs("%s", e.mkString(",")) }
      }
    }
    U.extractValues(pred)

    def addTo(oldPred:ExplicitPredicate) = {
      val op = pred.enumerate
      if (!op.possible) throw Utils.fails("Tried to add non-enumerable predicate %s: %s", pred.name, pred)
      oldPred ++= op.perform
    }
    predicates.get(pred.name) match {
      case Some(oldPred) => oldPred match {
        case oldPred:ExplicitPredicate => addTo(oldPred)
        case RenamedPredicate(_,oldPred:ExplicitPredicate) => addTo(oldPred)
        case _ =>
          val op = oldPred.enumerate
          if (!op.possible) throw Utils.fails("Can't add to existing non-enumerable predicate %s: %s", pred.name, oldPred)
          val oldPredCopy = new ExplicitPredicate(oldPred.arity, pred.name)
          oldPredCopy ++= op.perform
          addTo(oldPredCopy)
          predicates(pred.name) = oldPredCopy
      }
      case None =>
        predicates(pred.name) = pred
        pred.createIndex
    }
  }

  def addConstructPredicate(name:String, k:Int) = {
    Domains.typedDomains.add(TypedDomain(name))
    predicates.get(name) match {
      case None => predicates(name) = new ConstructPredicate(name, k)
      case _ => // Don't wory about it if it already exists
    }
  }

  def addPredicate(pred:Predicate) = {
    pred.createIndex
    U.extractValues(pred)
    predicates.get(pred.name) match {
      case Some(_) => throw Utils.fails("Predicate %s already exists", pred.name)
      case None => predicates(pred.name) = pred
    }
  }

  def removePredicate(name:String) : Unit = {
    predicates.remove(name) 
    if (parent != null)
      parent.removePredicate(name)
  }
}
