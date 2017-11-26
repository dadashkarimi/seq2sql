package dcs

// Failed computation
case class InterpretException(message:String) extends Exception {
  override def toString = "Interpretation error: "+message
}

// Can't convert from one thing to another
case class ConvertException(message:String) extends Exception {
  override def toString = "Conversion error: "+message
}
