package dcs

import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.LinkedHashSet
import EntityTypes.Entity

object Domains {
  val numDomains = List(CountDomain, OrdinalDomain, FracDomain,
                        LengthDomain, AreaDomain, CountDensityDomain, TimeDurationDomain,
                        DateYearDomain, DateMonthDomain, DateDayDomain, DateTimeDomain, DateDomain,
                        MoneyDomain, SalaryDomain)
  val typedDomains = new LinkedHashSet[TypedDomain]
  typedDomains += NameDomain
  def allDomains = numDomains ++ typedDomains

  val units = new ArrayBuffer[NumUnit]
  numDomains.foreach { dom => units ++= dom.units }

  def predWithDomains(doms:Iterable[Domain], f:Domain=>Entity) = {
    val entities = doms.map(f).filter(_.forall(_ != null))
    val pred = new ExplicitPredicate(entities.head.size)
    pred ++= entities
    pred
  }
  def predWithDomains(doms1:Iterable[Domain], doms2:Iterable[Domain], f:(Domain,Domain)=>Entity) = {
    val entities = doms1.flatMap { dom1 =>
      doms2.map { dom2 => f(dom1, dom2) }
    }.filter(_.forall(_ != null))
    val pred = new ExplicitPredicate(entities.head.size)
    pred ++= entities
    pred
  }

  def predWithNumDomains(f:Domain=>Entity) = predWithDomains(numDomains, f)
  def predWithTypedDomains(f:Domain=>Entity) = predWithDomains(typedDomains, f)
}
