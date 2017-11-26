package dcs

import tea.Utils
import java.io.File
import java.io.PrintWriter
import java.io.FileInputStream
import java.io.BufferedOutputStream
import java.io.OutputStreamWriter
import java.net.InetSocketAddress
import java.net.URLEncoder
import java.net.URLDecoder
import java.net.HttpCookie
import com.sun.net.httpserver.HttpServer
import com.sun.net.httpserver.HttpHandler
import com.sun.net.httpserver.HttpExchange
import java.security.SecureRandom
import java.math.BigInteger
import java.util.Collections
import java.util.concurrent.Executors
import java.util.concurrent.ExecutorService
import java.awt.image.BufferedImage
import java.awt.Color
import java.awt.Graphics2D
import java.awt.Font
import java.awt.RenderingHints
import java.awt.FontMetrics
import java.awt.BasicStroke
import java.awt.AlphaComposite
import java.awt.Rectangle
import javax.imageio.ImageIO
import scala.xml.Elem
import scala.xml.NodeSeq
import scala.xml.NodeBuffer
import scala.collection.mutable.HashMap
import scala.collection.mutable.ArrayBuffer

/*
Starts a webserver that answers questions.
*/

object SecureIdentifiers {
  private val random = new SecureRandom
  def newId = new BigInteger(130, random).toString(32)
}

class ServerOptions {
  import tea.OptionTypes._
  @Option var port = 8400
  @Option var basePath = "www"
  @Option var logPath : String = null
  @Option var propertiesPath : String = null
  @Option var collapseDetails = false
  @Option var numThreads = 4
}

object SO extends ServerOptions

object Justify {
  val LEFT = -1
  val CENTER = 0
  val RIGHT = +1
  def spacing(justify:Int, w1:Double, w2:Double) : Double = justify match {
    case LEFT => 0
    case RIGHT => w2-w1
    case CENTER => (w2-w1)/2
    case _ => throw Utils.impossible
  }
}

case class DPoint(x:Double, y:Double) {
  def +(that:DPoint) = DPoint(this.x+that.x, this.y+that.y)
  def -(that:DPoint) = DPoint(this.x-that.x, this.y-that.y)
  def *(f:Double) = DPoint(x*f, y*f)
  def /(f:Double) = DPoint(x/f, y/f)
  def addx(d:Double) = DPoint(x+d,y)
  def addy(d:Double) = DPoint(x,y+d)
  def rx = (x+0.5).toInt
  def ry = (y+0.5).toInt
  override def toString = Utils.fmts("(%s,%s)", x,y)
}

trait DObj {
  def width = dim.x
  def height = dim.y
  def dim : DPoint
  def draw(p0:DPoint) : Unit
}

class DEnv(g:Graphics2D) {
  val predColor = Color.RED
  val wordColor = new Color(0, 100, 0) // Dark green
  val relColor = Color.BLUE

  val fontName = "sansserif"
  val fontSize = 14
  val smallFontSize = 11
  val smallerFontSize = 9
  val tinyFontSize = 7
  val smallFont = new Font(fontName, Font.PLAIN, smallFontSize)
  val smallerFont = new Font(fontName, Font.PLAIN, smallerFontSize)
  val tinyFont = new Font(fontName, Font.PLAIN, tinyFontSize)
  val defaultFont = new Font(fontName, Font.PLAIN, fontSize)
  val boldFont = new Font(fontName, Font.BOLD, fontSize)
  val italicFont = new Font(fontName, Font.ITALIC, fontSize)
  val smallItalicFont = new Font(fontName, Font.ITALIC, smallFontSize)
  val boldItalicFont = new Font(fontName, Font.ITALIC+Font.BOLD, fontSize)

  object DNull extends DObj {
    def dim = DPoint(0, 0)
    def draw(p0:DPoint) = { }
  }

  case class DStr(value:String, color:Color=Color.BLACK, font:Font=defaultFont) extends DObj {
    val fm = g.getFontMetrics(font)
    val dim = DPoint(fm.stringWidth(value), fm.getHeight)
    def draw(p0:DPoint) = {
      //Utils.dbgs("DRAW %s at %s", value, p0)
      g.setColor(color)
      g.setFont(font)
      val p = p0.addy(fm.getAscent)
      g.drawString(value, p.rx, p.ry)
    }
  }

  case class DBox(margin:DPoint, obj:DObj, color:Color=Color.BLACK) extends DObj {
    def dim = obj.dim + margin*2
    def draw(p0:DPoint) = {
      val p1 = p0+dim
      g.setColor(color)
      g.drawRect(p0.rx, p0.ry, p1.rx, p1.ry)
      obj.draw(p0+margin)
    }
  }

  case class DrawNode(label:DObj, edges:List[DrawEdge], xmargin:Int=20, ymargin:Int=20) extends DObj {
    val childrenWidth = edges.map(_.c.width).sum + (edges.size-1)*xmargin
    val dim = DPoint(
      childrenWidth max label.width,
      {if (edges.size == 0) 0 else edges.map(_.c.height).max} + ymargin + label.height)

    def draw(p0:DPoint) = {
      label.draw(p0.addx((width-label.width) / 2))
      val rootp = p0 + DPoint(width/2, label.height)
      var childp = p0 + DPoint((label.width - childrenWidth)/2 max 0, label.height+ymargin)
      edges.foreach { edge =>
        edge.c.draw(childp)
        edge.draw(rootp, childp.addx(edge.c.width/2))
        childp = childp.addx(edge.c.width+xmargin)
      }
    }
  }
  case class DrawEdge(labels:List[DObj], c:DrawNode, color:Color=Color.BLACK, thickness:Double=1, endPaddingFrac:Double=0.1) {
    def draw(p1:DPoint, p2:DPoint) = {
      g.setColor(color)
      g.setStroke(new BasicStroke(thickness.toFloat))
      g.drawLine(p1.rx, p1.ry, p2.rx, p2.ry)

      def drawAt(f:Double, label:DObj) = {
        val p = (p1*(1-f) + p2*f) - label.dim/2
        DRect(label.dim, color=Color.WHITE, fill=true).draw(p) // White out the part for the label
        label.draw(p) // Draw label
      }

      // Draw labels evenly spaced over the edge
      labels match {
        case Nil =>
        case label :: Nil => drawAt(0.5, label) // One label => center it
        case labels => labels.zipWithIndex.foreach { case (label,i) =>
          drawAt(endPaddingFrac + i*(1-2*endPaddingFrac)/(labels.size-1), label)
        }
      }
    }
  }

  case class DRect(dim:DPoint, color:Color=Color.BLACK, fill:Boolean=false) extends DObj {
    def draw(p0:DPoint) = {
      g.setColor(color)
      if (fill) g.fillRect(p0.rx, p0.ry, dim.rx, dim.ry)
      else      g.drawRect(p0.rx, p0.ry, dim.rx, dim.ry)
    }
  }

  case class DXTable(items:List[DObj], margin:Double=10, n:Int=(-1), justify:Int=Justify.LEFT) extends DObj {
    val realItems = if (n == -1) items else items.slice(0, n)
    val dim = DPoint(realItems.map(_.width).sum+margin*(realItems.size-1), realItems.map(_.height).max)
    def draw(p0:DPoint) = {
      var p = p0
      items.foreach { item =>
        item.draw(p.addy(Justify.spacing(justify, item.height, height)))
        p = p.addx(item.width+margin)
      }
    }
  }
  case class DYTable(items:List[DObj], margin:Double=10, justify:Int=Justify.LEFT) extends DObj {
    val dim = DPoint(items.map(_.width).max, items.map(_.height).sum+margin*(items.size-1))
    def draw(p0:DPoint) = {
      var p = p0
      items.foreach { item =>
        item.draw(p.addx(Justify.spacing(justify, item.width, width)))
        p = p.addy(item.height+margin)
      }
    }
  }
}

trait DrawInfo {
  def getImage : BufferedImage
}
case class SemTreeDrawInfo(v:Node, state:BaseExampleInferState) extends DrawInfo {
  val words = state.words
  val U = state.U
  def getImage = {
    def make(width:Int, height:Int, draw:Boolean) = {
      val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
      val g = image.createGraphics

      if (draw) {
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 1.0f))
        g.setPaint(Color.BLUE)
        g.fillRect(0, 0, width, height)
        g.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f))
        g.setPaint(Color.BLACK)
      }

      val env = new DEnv(g) {
        def renderPred(v:Node) = {
          if (v.predicates == Nil) "*"
          else Renderer.simplifyPredName(v.pred.name)
        }
        def convertDRS(v:Node, nullSpan:Span) : DrawNode = { // null span is between v and its parent
          val predStr = DStr(renderPred(v), font=boldFont, color=predColor)
          val str = {
            if (v.predicates != Nil) {
              val lexStr = {
                if (v.hasAnchoredPred)
                  DStr(words.slice(v.predSpan._1, v.predSpan._2).mkString(" "), font=smallItalicFont, color=wordColor)
                else if (v.hasImplicitPred) {
                  if (nullSpan == null || nullSpan.size == 0) DNull
                  else DStr("("+words.slice(nullSpan._1, nullSpan._2).mkString(" ")+")", font=smallItalicFont, color=wordColor)
                }
                else throw Utils.impossible
              }
              DYTable(predStr :: lexStr :: Nil, margin=0, justify=Justify.CENTER)
            }
            else
              predStr
          }
          DrawNode(str, v.edges.flatMap { case Edge(r, c) =>
            val cc = convertDRS(c, v.nullSpan(c).getOrElse(null))
            val sigma = "\u03a3"
            r match {
              case CollectRel => DrawEdge(DStr(sigma, font=smallFont, color=relColor)::Nil, cc, thickness=2) :: Nil
              case r:MarkerRel => DrawEdge(DStr(r.toString, font=smallerFont, color=relColor)::Nil, cc, thickness=2) :: Nil
              case ExecuteRel(is) => DrawEdge(DXTable(DStr("X", font=smallerFont, color=relColor)::
                                                      DStr(is.map(_+1).mkString(""), font=tinyFont, color=relColor)::Nil, Justify.RIGHT)::Nil,
                                              cc, thickness=2) :: Nil
              case JoinRel(i,j) => DrawEdge(DStr((i+1).toString, font=smallerFont, color=relColor)::
                                            DStr((j+1).toString, font=smallerFont, color=relColor)::Nil, cc, thickness=2) :: Nil
              case _ => throw Utils.impossible
            }
          })
        }

        def convertParseTree(tree:ParseTree) : DrawNode = {
          DrawNode(
            DStr(tree.tag, font={if (tree.isLeaf) smallItalicFont else smallFont}),
            tree.children.map{c => new DrawEdge(Nil, convertParseTree(c))},
            xmargin=16, ymargin=16)
        }

        val numStart = Utils.map(state.N, { i:Int => 0 })
        val numEnd = Utils.map(state.N+1, { i:Int => 0 })
        val hit = Utils.map(state.N, { i:Int => false })
        state.transitions.zipWithIndex.foreach { case (l,i) =>
          l.foreach { case (_,j) =>
            (i to j-1).foreach(hit(_) = true)
            numStart(i) += 1
            numEnd(j) += 1
          }
        }

        // Draw everything
        val fig = DXTable(convertDRS(v, null) :: {
          if (state.parseInfo != null && MO.features.contains("syntax")) convertParseTree(state.parseInfo.tree)::Nil else Nil
        }, margin=100)
        val obj = DYTable(DXTable(DNull :: fig :: Nil, margin=40) :: Nil, margin=40)
      }
      if (draw) env.obj.draw(DPoint(0,0))
      (image, env.obj.dim)
    }

    val (_, dim) = make(1, 1, false)
    val (image, _) = make(dim.rx, dim.ry, true)
    image
  }
}

object QueryBuilder {
  def query(question:String=null, answer:String=null, add:String=null) = {
    val s = new StringBuilder
    s.append("/")
    var first = true
    (("question", question) :: ("answer", answer) :: ("add", add) :: Nil).foreach { case (k,v) =>
      if (v != null) {
        s.append({if (first) "?" else "&"})
        s.append(k+"="+URLEncoder.encode(v, "UTF-8"))
        first = false
      }
    }
    s.toString
  }
}

class Handler(model:NuggetModel) extends HttpHandler {
  val mimeTypes = Map("html" -> "text/html", "jpeg" -> "image/jpeg", "gif" -> "image/gif", "css" -> "text/css")
  def getMimeType(path:String) = mimeTypes.getOrElse(path.split("\\.").last, "text/plain")

  val drawCache = new HashMap[String,DrawInfo]
  def addToDrawCache(info:SemTreeDrawInfo) = drawCache.synchronized {
    val id = ""+drawCache.size
    drawCache(id) = info
    id
  }

  val properties : Map[String,String] = {
    if (SO.propertiesPath == null) Map()
    else Utils.readLines(SO.propertiesPath).map(_.split("=", 2) match {case Array(k,v) => (k,v)}).toMap
  }

  def makeTooltip(main:NodeSeq, aux:NodeSeq, link:String="") = <a href={link} class="info">{main}<span class="tooltip">{aux}</span></a>

  def renderPred(pred:String) = <span class="predicate">{Renderer.simplifyPredName(pred)}</span>
  def renderItem(item:ResponseItem) : NodeSeq = item match {
    case MessageResponseItem(message, _) => <span>{message}</span>
    case GroupResponseItem(items) =>
      <table class="groupResponse">
        <tr>{items.map{item => <td>{renderItem(item)}</td>}}</tr>
      </table>
    case ListResponseItem(header, elements) => {
      <div class="listResponse">
        <span class="listHeader">{header}</span>
        <ul>{elements.map{ case ListResponseItemElement(main, link, tooltip) =>
          <li>{if (tooltip == null) main else makeTooltip(<span>{main}</span>, renderItem(tooltip), link)}</li>}
        }</ul>
      </div>
    }
    case LexicalResponseItem(elements) => {
      val predRow = elements.map { case WordInfo(word, tag, predInfos) =>
        predInfos match {
          case Nil => <td/>
          case (pred,_)::rest => {
            val extra = if (rest == Nil) "" else "..."
            val predInfoTable : Elem = {
              <table class="predInfo">{predInfos.map { case (pred,info) =>
                <tr><td>{renderPred(pred)}</td><td>{info}</td></tr>
              }}</table>
            }
            <td>{makeTooltip(<span>{renderPred(pred)}{extra}</span>, predInfoTable)}</td>
          }
        }
      }
      val wordRow = elements.map { case WordInfo(word, tag, predInfos) =>
        <td>{makeTooltip(<span class="word">{word}</span>, <span class="tag">POS: {tag}</span>)}</td>
      }

      <div class="lexicalResponse">
        <span class="listHeader">Lexical Triggers</span>
        <table>
          <tr>{predRow}</tr>
          <tr>{wordRow}</tr>
        </table>
      </div>
    }
    case SemTreeResponseItem(header, v, state) => {
      val id = addToDrawCache(SemTreeDrawInfo(v, state))
      val img = {<img class="semTree" src={"/img?id="+id}/>}
      <div class="semTreeResponse">
        {if (header == null) Nil else <div class="listHeader">{header}</div>}
        <div>{img}</div>
        <div class="semTreeDetails">(beam &ge; {v.maxBeamPos})</div>
      </div>
    }
  }

  def handle(exchange:HttpExchange) = {
    val uri = exchange.getRequestURI
    val remoteHostName = exchange.getRemoteAddress.getHostName

    val uriPath = uri.getPath
    val reqParams : Map[String,String] = try { // Don't use uri.getQuery: it can't distinguish between '+' and '-'
      uri.toString.split("""\?""") match {
        case Array(_) => Map()
        case Array(_,query) => query.split("&").map { s =>
          s.split("=", 2) match {
            case Array(k,v) =>
              //Utils.dbgs("DECODE %s => %s", v, URLDecoder.decode(v, "UTF-8"))
              (URLDecoder.decode(k, "UTF-8"), URLDecoder.decode(v, "UTF-8"))
            case _ => throw Utils.impossible
          }
        }.toMap 
      }
    } catch { case e =>
      Utils.logs("Invalid query: %s", uri.getQuery)
      e.printStackTrace
      Map()
    }

    //Utils.logs("%s", reqParams)

    val (cookie, isNewSession) = {
      val c = exchange.getRequestHeaders.getFirst("Cookie")
      if (c != null) (HttpCookie.parse(c).get(0), false) // Cookie already exists
      else (new HttpCookie("sessionId", SecureIdentifiers.newId), true) // Create a new cookie
    }
    val sessionId = cookie.getValue
    //Utils.dbgs("Cookie: %s", cookie)
    Utils.logs("GET %s from %s (%ssessionId=%s)", uri, remoteHostName, {if (isNewSession) "new " else ""}, sessionId)

    def setHeaders(mimeType:String) = {
      val headers = exchange.getResponseHeaders
      headers.set("Content-Type", mimeType)
      if (isNewSession)
        headers.set("Set-Cookie", cookie.toString)
      exchange.sendResponseHeaders(200, 0)
    }

    def withWriter(f:PrintWriter =>Any) = {
      val writer = new PrintWriter(new OutputStreamWriter(exchange.getResponseBody))
      f(writer)
      writer.close
    }

    def getImage = try {
      reqParams.get("id").flatMap(drawCache.get) match {
        case None =>
          Utils.logs("ID '%s' not in the cache (%s items)", reqParams.get("id").getOrElse(""), drawCache.size)
        case Some(drawInfo) =>
          val image = drawInfo.getImage
          val out = new BufferedOutputStream(exchange.getResponseBody)
          setHeaders(getMimeType("png"))
          ImageIO.write(image, "png", out)
          out.close
      }
    } catch {
      case e => e.printStackTrace
      throw e
    }

    def getFile = {
      val path = SO.basePath+uriPath
      if (new File(path).exists) {
        setHeaders(getMimeType(path))
        //Utils.logs("Sending %s", path)
        val out = new BufferedOutputStream(exchange.getResponseBody)
        val buf = new Array[Byte](16384)
        val in = new FileInputStream(path)
        def readAll : Unit = {
          val n = in.read(buf)
          if (n > 0) { out.write(buf, 0, n); readAll }
        }
        readAll
        in.close
        out.close
      }
      else {
        //Utils.logs("File not found: %s", path)
        exchange.sendResponseHeaders(404, 0)
      }
    }

    val header = {
      <title>{properties.getOrElse("title", "A Question-Answering System using Dependency-Based Compositional Semantics")}</title>
      <link rel="stylesheet" type="text/css" href="main.css"/>
      <script src="main.js"></script>
      <div class="description">{properties.getOrElse("description", "")}</div>
    }

    var collapsableId = 0
    def collapsable(label:String, contents:NodeSeq) = {
      collapsableId += 1 
      <p>
        <div id={"_back"+collapsableId} style="display:none">{contents}</div>
        <a href={"javascript:copy(_front"+collapsableId+", _back"+collapsableId+")"}>{label}</a>
        <div id={"_front"+collapsableId}></div>
      </p>
    }

    def questionPart(question:String=null) = {
      <div>
        <form action="/">
          <input class="question" type="text" autofocus="true" size="50" name="question" value={if (question == "") properties.getOrElse("exampleQuestion", "") else question}/>
          <input class="ask" type="submit" value="Ask"/>
        </form>
      </div>
    }
    def answerPart(answer:String) = {
      <table><tr>
        <td><span class="answer">{answer}</span></td>
        <td>{makeTooltip(<span class="correctButton">[Correct]</span>,
                         <div class="bubble">If this answer is correct, click to add as a new training example!</div>,
                         link=QueryBuilder.query(question=null, answer=null, add="1"))}</td>
        <td>{makeTooltip(<span class="wrongButton">[Wrong]</span>,
                         <div class="bubble">If this is wrong, click on the correct answer from the list of candidate answers below.</div>)}</td>
      </tr></table>
    }
    /*def answerPart(answer:String) = {
      makeTooltip(<span class="answer">{answer}</span>,
                  <div>
                    <div class="nowrap">If this is correct, click to add it as a new training example!</div>
                    <div class="nowrap">Otherwise, find the correct answer from the candidate answers below.</div>
                  </div>,
                  link=QueryBuilder.query(question=null, answer=null, add="1"))
    }*/

    def detailsPart(details:NodeSeq) = {
      val div = <div class="details">{details}</div>
      if (SO.collapseDetails) collapsable("[Details]", div) else div
    }

    def doMain = withWriter { writer:PrintWriter =>
      setHeaders("text/html")
      header.foreach(writer.println)
      val session = model.getSession(sessionId)

      val question = reqParams.getOrElse("question", "").trim
      val answer = reqParams.getOrElse("answer", "").trim
      val add = reqParams.getOrElse("add", "").trim

      writer.println(questionPart(if (question != "") question else session.getLastQuestion))

      if (question != "" || answer != "" || add != "") {
        val response : Response = try {
          val req = {
            if (add != "") AddExampleRequest
            else if (answer != "") SetAnswerRequest(answer.toInt) // Set the temporary answer
            else GeneralRequest(question)
          }
          session.handleRequest(req)
        } catch {
          case TimedOutException =>
            MessageResponseItem("Sorry, your request took too long.")
          case e =>
            e.printStackTrace
            MessageResponseItem("Sorry, there was an internal error.")
        }

        // Log the interaction
        if (SO.logPath != null) {
          val answers = response.items.map {
            case MessageResponseItem(message, _) => message
            case _ =>
          }
          val answer = if (answers.size == 0) "" else answers.head
          val logOut = fig.basic.IOUtils.openOutAppendHard(SO.logPath)
          val row = Array(
            fig.basic.Fmt.formatEasyDateTime((new java.util.Date).getTime), remoteHostName+":"+sessionId, question, answer)
          logOut.println(row.mkString("\t"))
          logOut.close
        }

        // Write stuff to the user
        response.items.foreach { // Write messages first
          case MessageResponseItem(message, main) =>
            writer.println(if (main) answerPart(message) else <div class="message">{message}</div>)
          case _ =>
        }
        
        writer.println(detailsPart(response.items.filter(!_.isInstanceOf[MessageResponseItem]).flatMap(renderItem)))
      }
    }

    if (uriPath == "/") doMain
    else if (uriPath == "/img") getImage
    else getFile

    exchange.close
  }
}

class Server(model:NuggetModel) {
  System.setProperty("java.awt.headless", "true") // Don't need X to create images
  val hostname = fig.basic.SysInfoUtils.getHostName
  val server = HttpServer.create(new InetSocketAddress(SO.port), 10)
  val pool = Executors.newFixedThreadPool(SO.numThreads)
  server.createContext("/", new Handler(model))
  server.setExecutor(pool)
  server.start
  Utils.logs("Server started at http://%s:%s", hostname, SO.port)
  Utils.logs("Press Ctrl-D to terminate.")
  while (readLine != null) { }
  model.saveParams
  Utils.logs("Shutting down server...")
  server.stop(0)
  Utils.logs("Shutting down executor pool...")
  pool.shutdown
}
