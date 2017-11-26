package dcs

// World-independent way of referencing predicates.
trait PredicateHandle {
  def name : String
}
case class PredicateName(name:String) extends PredicateHandle
case class PredicateConstant(pred:Predicate) extends PredicateHandle {
  def name = pred.name
}
