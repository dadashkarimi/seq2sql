package dcs

import scala.collection.mutable.ArrayBuffer
import tea.Utils

object JoinMode {
  val firstOnly = "1" // p1
  val share = "1,2-1" // p1 p2\p1
  val disjoint = "1,2" // p1 p2
}

object PredicateOps {
  // Implements the core operation (join-and-project) for evaluating the semantics of DCS trees.
  def join(p1:Predicate, p2:Predicate, indices1:List[Int], indices2:List[Int], joinMode:String, isCon:Boolean) : Operation[Predicate] = {
    if (MO.verbose >= 5 && IO.displayAbsCon(isCon))
      Utils.dbgs("JOIN(%s) %s[%s] and %s[%s]", joinMode, p1.render, indices1.mkString(","), p2.render, indices2.mkString(","))

    require (indices1.forall(_<p1.arity))
    require (indices2.forall(_<p2.arity))

    // Try special case joins
    if (joinMode != JoinMode.disjoint) {
      if (p1.arity == 1 && p2.arity == 1) { // Sometimes predicates have special implicit way of doing join
        val op1 = p1.join(p2, 0)
        val op2 = p2.join(p1, 0)
        if (op1.possible || op2.possible) {
          if (op1.retCost <= op2.retCost) return op1
          else                            return op2
        }
      }
      else if (p1.arity == 1) {
        indices2 match {
          case i :: Nil =>
            val op2 = p2.join(p1, i)
            if (op2.possible)
              return op2.extendConst{p2:Predicate =>
                if (joinMode == JoinMode.share)
                  p2.proj(MyUtils.moveFront(MyUtils.IntList(p2.arity), i))
                else
                  p2.proj(i :: Nil)
              }
          case _ =>
        }
      }
      else if (p2.arity == 1) {
        indices1 match {
          case i :: Nil =>
            val op1 = p1.join(p2, i)
            if (op1.possible) return op1
          case _ =>
        }
      }
    }

    // Motivating example: argmax - 2,1 - countTopRanked
    // When joining p1 and p2 where...
    if (joinMode == JoinMode.firstOnly) { // we're going to marginalize out p2...
      (p2, indices2) match {
        case (ProdPredicate(p2a, p2b), i::Nil) => // and p2 is just a product where only one coordinate i matters...
          val (new_p2, new_i) = {
            if (i < p2a.arity) (p2a, i)
            else               (p2b, i-p2a.arity)
          }
          return join(p1, new_p2, indices1, new_i::Nil, joinMode, isCon) // Then just extract that coordinate
        case _ =>
      }
    }

    // Default implementation: call enumerate on one predicate and call select on the other
    // Might not be efficient if we ask select on the same thing over and over again.
    val enum1op = p1.enumerate; val sel1op = p1.select(indices1, null)
    val enum2op = p2.enumerate; val sel2op = p2.select(indices2, null)

    val retCost1 = enum1op.outCost * sel2op.retCost
    val retCost2 = enum2op.outCost * sel1op.retCost
    val outCost1 = enum1op.outCost * sel2op.outCost
    val outCost2 = enum2op.outCost * sel1op.outCost
    val reorder = retCost1 < retCost2 // Reorder if generating from p1
    val (retCost, outCost) = {
      if (reorder) (retCost1, outCost1)
      else         (retCost2, outCost2)
    }
    new ExplicitOperation[Predicate](retCost, outCost, /*Debug.track("s1"->s1,"s2"->s2,"t1"->t1,"t2"->t2,"indices1"->indices1,"indices2"->indices2)*/ {
      if (IO.verbose >= 2) {
        Utils.logs("  JOIN %s[%s] %s[%s] | COST: %s*%s=%s|%s versus %s*%s=%s|%s",
          p1.render, indices1.mkString(","), p2.render, indices2.mkString(","),
          enum1op.outCost, sel2op.retCost, retCost1, outCost1,
          enum2op.outCost, sel1op.retCost, retCost2, outCost2)
      }
      if (retCost == Operation.infCost)
        throw InterpretException("cannot join via enumerate/select:"+p1+" and "+p2)

      val generator     = if (reorder) p1 else p2
      val discriminator = if (reorder) p2 else p1

      val genJoinIndices = if (reorder) indices1 else indices2
      val disJoinIndices = if (reorder) indices2 else indices1
      val genBackIndices = MyUtils.IntList(generator.arity).filter(!genJoinIndices.contains(_))
      val disBackIndices = MyUtils.IntList(discriminator.arity).filter(!disJoinIndices.contains(_))

      val name = {
        if (p1.name == null || p1.arity == 0) p2.name
        else if (p2.name == null || p2.arity == 0) p1.name
        else "("+p1.name+"*"+p2.name+")"
      }

      val resultArity = (joinMode match {
        case JoinMode.firstOnly => if (reorder) generator.arity                       else discriminator.arity
        case JoinMode.share     => if (reorder) generator.arity + disBackIndices.size else discriminator.arity + genBackIndices.size
        case JoinMode.disjoint  => if (reorder) generator.arity + discriminator.arity else discriminator.arity + generator.arity
        case _ => throw Utils.impossible
      })
      val resultPreds = new ArrayBuffer[Predicate]
      val resultPred = new ExplicitPredicate(resultArity, name) // Store everything that's enumerable
      resultPreds += resultPred
      generator.enumerate.perform.foreach { gen_e =>
        val gen_back_e = genBackIndices.map(gen_e) // Condition
        if (IO.verbose >= 5)
          Utils.logs("GEN %s (%s)", Renderer.render(gen_e), Renderer.render(gen_back_e))
        if (IO.verbose >= 5) Utils.dbgs("GGG %s %s", discriminator, genJoinIndices.map(gen_e))
        val selected = discriminator.select(disJoinIndices, genJoinIndices.map(gen_e)).perform
        //require (selected.arity, discriminator.arity), discriminator + " => " + selected.arity)
        val op = selected.enumerate
        if (op.possible) { // Not clear if we always want to unpack everything
          selected.enumerate.perform.foreach { dis_e =>
            if (IO.verbose >= 5)
              Utils.logs("DIS %s", Renderer.render(dis_e))
            resultPred += (joinMode match {
              case JoinMode.firstOnly => if (reorder) gen_e                              else dis_e
              case JoinMode.share     => if (reorder) gen_e ++ disBackIndices.map(dis_e) else dis_e ++ gen_back_e
              case JoinMode.disjoint  => if (reorder) gen_e ++ dis_e                     else dis_e ++ gen_e
              case _ => throw Utils.impossible
            })
          }
        }
        else {
          resultPreds += (joinMode match {
            case JoinMode.firstOnly => if (reorder) SingletonPredicate(gen_e)
                                       else         selected
            case JoinMode.share     => if (reorder) ProdPredicate.create(SingletonPredicate(gen_e), selected.proj(disBackIndices))
                                       else         ProdPredicate.create(selected, SingletonPredicate(gen_back_e))
            case JoinMode.disjoint  => if (reorder) ProdPredicate.create(SingletonPredicate(gen_e), selected)
                                       else         ProdPredicate.create(selected, SingletonPredicate(gen_e))
          })
        }
      }

      if (MO.verbose >= 6 && IO.displayAbsCon(isCon))
        Utils.dbgs("JOIN(%s) %s[%s] and %s[%s] => %s", joinMode, p1.render, indices1.mkString(","), p2.render, indices2.mkString(","),
          resultPreds.map(_.render))

      UnionPredicate.create(resultPreds.toList)
    })
  }
}
