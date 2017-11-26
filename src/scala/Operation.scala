package dcs

// Reifies an operation (select, join)
// We can measure the cost of the operation first before performing it.
// Two types of cost:
//   retCost: time to perform the operation (and return an output)
//   outCost: time to scan through the output (size of the output).
trait Operation[A] {
  def retCost : Double
  def outCost : Double
  def perform : A
  def possible = retCost != Operation.infCost

  def extendConst[B](f:A=>B) = new ExplicitOperation[B](retCost, outCost, f(perform))
  def extendLinear[B](f:A=>B) = new ExplicitOperation[B](outCost, outCost, f(perform)) // Need to all of the output of f

  require (0 < retCost, retCost+" "+outCost)
  require (retCost != Operation.infCost || outCost == Operation.infCost, retCost+" "+outCost) // If retCost is infinite then so is outCost
}

class ExplicitOperation[A](val retCost:Double, val outCost:Double, _perform: =>A) extends Operation[A] {
  def perform = _perform
}
class LinearOperation[A](val retCost:Double, _perform: =>A) extends Operation[A] { // Output is linear in computation time
  def outCost = retCost
  def perform = _perform
}
class SimpleOperation[A](_perform: =>A) extends Operation[A] {
  def retCost = 1 
  def outCost = 1 
  def perform = _perform
}
case class ConstOperation[A](perform:A) extends Operation[A] {
  def retCost = 1 
  def outCost = 1 
}

object Operation {
  val infCost = Double.PositiveInfinity
  def totalRetCost[A](ops:List[Operation[A]]) = ops.foldLeft(0.0) { case (acc,op) => acc + op.retCost }
  def totalOutCost[A](ops:List[Operation[A]]) = ops.foldLeft(0.0) { case (acc,op) => acc + op.outCost }
}
