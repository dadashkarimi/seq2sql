package dcs

case class Span(_1:Int, _2:Int) {
  def size = _2-_1
  override def toString = _1+"..."+_2
}

