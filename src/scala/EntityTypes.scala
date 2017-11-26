package dcs

// An entity is a tuple (list) of values.
object EntityTypes {
  type Entity = List[Value]
  val Entity = List
  type EntityIterable = Iterable[_ <: Entity]
}
