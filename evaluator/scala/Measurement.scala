package dcs

import scala.collection.mutable.ArrayBuffer
import tea.Utils

////////////////////////////////////////////////////////////
// A number is a quantity (e.g., length, area, etc.).

trait NumDomain extends Domain {
  def name : String // ABSTRACT
  def render = name
  def units : List[NumUnit] // ABSTRACT (units available)
}

case class ProdNumUnit(u1:NumUnit, u2:NumUnit) extends NumUnit {
  def name = if (u1 == u2) "squared "+u1.name else u1.name+" "+u2.name
  def abs = ProdNumDomain(u1.abs, u2.abs)
  def conversion = u1.conversion * u2.conversion
}
case class ProdNumDomain(dom1:NumDomain, dom2:NumDomain) extends NumDomain {
  def name = {
    if (dom1 == dom2) dom1.name+"^2"
    else dom1.name + " " + dom2.name
  }
  val units = {
    if (dom1 == dom2)
      dom1.units.map { u => ProdNumUnit(u, u) }
    else
      dom1.units.flatMap { u1 => dom2.units.map { u2 => ProdNumUnit(u1, u2) } }
  }
}
case class RatioNumUnit(u1:NumUnit, u2:NumUnit) extends NumUnit {
  def name = u1.name+"/"+u2.name
  def abs = RatioNumDomain(u1.abs, u2.abs)
  def conversion = u1.conversion / u2.conversion
}
case class RatioNumDomain(dom1:NumDomain, dom2:NumDomain) extends NumDomain {
  def name = dom1.name + "/" + dom2.name
  val units = dom1.units.flatMap { u1 => dom2.units.map { u2 => RatioNumUnit(u1, u2) } }
}

// value represents the num in the canonical units
object NumValue {
  lazy val roundFormatStr = "%."+(IO.roundNumDigits-1)+"e"
}
case class NumValue(abs:NumDomain, value:Double) extends Value with Ordered[NumValue] {
  val roundedValue = {
    if (IO.roundNumDigits == Integer.MAX_VALUE) value
    else NumValue.roundFormatStr.format(value)
  }

  def size = 1
  override def hashCode = roundedValue.hashCode
  override def equals(that:Any) = that match {
    case that:NumValue => this.abs == that.abs && this.roundedValue == that.roundedValue
    case _ => false
  }

  def assertCompatible(that:NumValue) = {
    if (this.abs != that.abs)
      throw InterpretException("Incompatible num values: "+this.abs+" and "+that.abs)
  }
  def min(that:NumValue) = { assertCompatible(that); NumValue(abs, this.value min that.value) }
  def max(that:NumValue) = { assertCompatible(that); NumValue(abs, this.value max that.value) }
  def +(that:NumValue) = { assertCompatible(that); NumValue(abs, this.value + that.value) }
  def -(that:NumValue) = { assertCompatible(that); NumValue(abs, this.value - that.value) }
  def *(c:Double) = NumValue(abs, value*c)
  def /(c:Double) = NumValue(abs, value/c)
  def *(that:NumValue) = NumValue(ProdNumDomain.create(this.abs, that.abs), this.value * that.value)
  def /(that:NumValue) = NumValue(RatioNumDomain.create(this.abs, that.abs), this.value / that.value)
  def unary_- = NumValue(abs, -value)
  def compare(that:NumValue) = {
    assertCompatible(that)
    this.value compare that.value
  }

  def absoluteValue = NumValue(abs, value.abs)

  def render = Utils.fmts("%s%s", roundedValue, abs.render)
  def humanRender = abs.units.head.humanRender(this)
}

trait NumUnit {
  def abs : NumDomain
  def name : String
  def conversion : Double
  require (conversion > 0)
  def applyUnit(value:Double) = NumValue(abs, value * conversion)
  def unapplyUnit(v:NumValue) = {
    require (abs == v.abs)
    v.value / conversion
  }

  def humanRender(n:Double) = {
    Utils.fmts("%s %s", Renderer.humanNum(n), Renderer.noun(name, n))
  }
  def humanRender(v:NumValue) : String = humanRender(unapplyUnit(v))
}
case class BaseNumUnit(abs:NumDomain, name:String, conversion:Double, abbrev:String=null) extends NumUnit {
  override def humanRender(n:Double) = {
    if (abbrev != null) Utils.fmts("%s%s", Renderer.humanNum(n), abbrev) else super.humanRender(n)
  }
}

case class SimpleNumDomain(name:String) extends NumDomain {
  val units = BaseNumUnit(this, name, 1, "") :: Nil
}

//// Generic
object CountDomain extends SimpleNumDomain("count")
object OrdinalDomain extends NumDomain {
  val name = "rank"
  val units = BaseNumUnit(this, "rank", 1) :: Nil
}
object FracDomain extends NumDomain {
  val name = "frac"
  val units = BaseNumUnit(this, "percent", 0.01, "%") :: Nil
}

//// Length, area, density
object LengthDomain extends NumDomain {
  val name = "length"
  val units =
    BaseNumUnit(this, "meter", 1, "m") ::
    BaseNumUnit(this, "kilometer", 1000, "km") ::
    BaseNumUnit(this, "inch", 0.0254, "in") ::
    BaseNumUnit(this, "foot", 0.3048, "ft") ::
    BaseNumUnit(this, "mile", 1609.344) ::
    Nil
}

//// Time duration
object TimeDurationDomain extends NumDomain { // In days
  val name = "time_duration"
  val units =
    BaseNumUnit(this, "second", 1.0/24/60/60) ::
    BaseNumUnit(this, "minute", 1.0/24/60) ::
    BaseNumUnit(this, "hour", 1.0/24) ::
    BaseNumUnit(this, "day", 1) ::
    BaseNumUnit(this, "week", 7) ::
    BaseNumUnit(this, "month", 365.0/12) ::
    BaseNumUnit(this, "working_hour", 40*7) ::
    BaseNumUnit(this, "year", 365) ::
    Nil
}

//// Date, time
object DateYearDomain extends SimpleNumDomain("date_year")
object DateMonthDomain extends SimpleNumDomain("date_month")
object DateDayDomain extends SimpleNumDomain("date_day")
object DateTimeDomain extends NumDomain {
  val name = "date_time" // minutes since midnight
  val units = new BaseNumUnit(this, name, 1) {
    override def humanRender(n:Double) = {
      val h = n / 60
      val m = n % 60
      "%d:%02d".format(h.toInt, m.toInt)
    }
  } :: Nil
}
object DateDomain extends NumDomain { // Seconds since Jan 1, 1970
  val name = "date"
  val dateFormat = new java.text.SimpleDateFormat("yyyy/MM/dd hh:mm")
  val units = new BaseNumUnit(this, name, 1) {
    override def humanRender(n:Double) = dateFormat.format(new java.util.Date(n.toInt))
  } :: Nil
}

// Money
object MoneyDomain extends NumDomain { // In dollars
  val name = "money"
  val units =
    new BaseNumUnit(this, "dollar", 1) {
      override def humanRender(n:Double) = {
        if (n >= 100) "$"+Renderer.humanNum(n)
        else "$%.2f".format(n)
      }
    } ::
    BaseNumUnit(this, "cent", 1.0/100) ::
    Nil
}

object AreaDomain extends ProdNumDomain(LengthDomain, LengthDomain)
object CountDensityDomain extends RatioNumDomain(CountDomain, AreaDomain)
object SalaryDomain extends RatioNumDomain(MoneyDomain, TimeDurationDomain)

object ProdNumDomain {
  def create(d1:NumDomain, d2:NumDomain) = {
    if (d1 == CountDomain) d2
    else if (d2 == CountDomain) d1
    else ProdNumDomain(d1, d2)
  }
  def lookup(d1:Domain, d2:Domain) = (d1, d2) match {
    case (LengthDomain, LengthDomain) => AreaDomain
    case _ => null
  }
}
object RatioNumDomain {
  def create(d1:NumDomain, d2:NumDomain) = {
    if (d2 == CountDomain) d1
    else if (d1 == d2) CountDomain
    else RatioNumDomain(d1, d2)
  }
  def lookup(d1:Domain, d2:Domain) = (d1, d2) match {
    case (CountDomain, AreaDomain) => CountDensityDomain
    case (MoneyDomain, TimeDurationDomain) => SalaryDomain
    case _ => null
  }
}
