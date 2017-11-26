package dcs

import tea.Utils

// Generic request/response messages for interfacing with the system.
// Mostly used by the Web interface.

//// Request
trait Request
case class GeneralRequest(line:String) extends Request
case class SentenceRequest(basketId:String, mode:String, sentence:String) extends Request
case class SetAnswerRequest(answerId:Int) extends Request
object AddExampleRequest extends Request

//// Response
case class Response(items:List[ResponseItem]) {
  def ++(that:Response) = Response(this.items ++ that.items)
}
trait ResponseItem
case class GroupResponseItem(items:List[ResponseItem]) extends ResponseItem
case class MessageResponseItem(message:String, main:Boolean=false) extends ResponseItem
case class ListResponseItem(label:String, elements:Seq[ListResponseItemElement]) extends ResponseItem // list of strings and their tooltips
case class SemTreeResponseItem(label:String, v:Node, state:BaseExampleInferState) extends ResponseItem
case class LexicalResponseItem(elements:Seq[WordInfo]) extends ResponseItem
case class WordInfo(word:String, tag:String, predInfos:List[(String,String)]) extends ResponseItem

case class ListResponseItemElement(main:String, link:String=null, tooltip:ResponseItem=null)

object Response {
  implicit def toResponse(item:ResponseItem) = Response(item::Nil)

  def displayResponse(response:Response) : Unit = response.items.foreach {
    case MessageResponseItem(message, _) => Utils.logs("%s", message)
    case GroupResponseItem(items) => displayResponse(Response(items)) 
    case ListResponseItem(_, elements) =>
      elements.zipWithIndex.foreach { case (ListResponseItemElement(elem,_,_),i) =>
        Utils.logs("%s%s", {if (i == 0) "=> " else "   "}, elem)
      }
    case _ => // Ignore
  }
}
