package dcs

import java.io.File
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.HashMap
import tea.Utils
import EntityTypes.Entity
import fig.basic.Indexer
import fig.exec.Execution

class DataManager(U:Universe) extends BaseDataManager[Example] {
  import tea.OptionTypes._
  @Option var maxSentenceLength = Integer.MAX_VALUE

  def fieldPred = U.currWorld.getPredicate(U.toPredName("field", 4))

  // This is not used for learning but creating an HTML interface to get data from Mechanical Turk.
  // This is here because the information is stored in the Datalog knowledgebase.
  def generateDBHtml(description:String, path:String) = Utils.track("generateDBHtml: %s to %s", description, path) {
    case class Field(i:Int, predName:String, displayName:String, unitName:String) {
      val pred = U.currWorld.getPredicate(U.toPredName(predName, 2))
      val munit = if (unitName == "") null else U.currWorld.getPredicate(U.toPredName(unitName, 2)).asInstanceOf[UnitConversionPredicate].munit
      // Display value, sort value, return value
      def humanRender(v:Value) : String = v match {
        case NameValue(value) => value
        case TypedValue(Entity(NameValue(value)), _) => value
        case NumValue(CountDomain, value) => Renderer.humanNum(value)
        case mv @ NumValue(FracDomain, value) => FracDomain.units.head.humanRender(mv)
        case mv @ NumValue(MoneyDomain, value) => MoneyDomain.units.head.humanRender(mv)
        case v:NumValue =>
          if (munit == null) throw Utils.fails("No unit for %s, but have value %s", this, v)
          Renderer.humanNum(munit.unapplyUnit(v))
        case _ => throw Utils.fails("Unexpected: %s", v)
      }
      def sortkey(v:Value) : String = v match {
        case NameValue(value) => value.toLowerCase
        case TypedValue(Entity(NameValue(value)), _) => value.toLowerCase
        case NumValue(CountDomain, value) => value.toString
        case mv @ NumValue(FracDomain, value) => FracDomain.units.head.unapplyUnit(mv).toString
        case mv @ NumValue(MoneyDomain, value) => MoneyDomain.units.head.unapplyUnit(mv).toString
        case v:NumValue =>
          if (munit == null) throw Utils.fails("No unit for %s, but have value %s", this, v)
          munit.unapplyUnit(v).toString
        case _ => throw Utils.fails("Unexpected: %s", v)
      }
      def display = displayName + {if (unitName == "") "" else "<br>("+Renderer.noun(munit.name, 0)+")"}
    }
    case class Record(val id:TypedValue, values:List[List[Value]])

    val allFields = fieldPred.enumerate.perform.map {
      case Entity(NumValue(CountDomain, i), NameValue(predName), NameValue(displayName), NameValue(unitName)) =>
        Field(i.toInt, predName, displayName, unitName)
      case x => throw Utils.fails("Unexpected field specification: %s", x)
    }.toList.sortWith(_.i < _.i)

    def generate(fields:List[Field], path:String) = {
      val ids = fields(0).pred.enumerate.perform.map {
        case (id:TypedValue) :: rest => id
        case x => throw Utils.fails("First argument should be id, but got: %s", x)
      }.toList
      val records = ids.map { id =>
        val values = fields.map { field =>
          field.pred.select(List(0), List(id)).perform.enumerate.perform.map {
            case Entity(_, v) => v
          }.toList
        }
        Record(id, values)
      }.sortWith(_.values.head.head.toString < _.values.head.head.toString)

      //def bold(s:String) = "<b>"+s+"</b>"
      //def row(l:List[String]) = "<tr>"+l.map("<td>"+_+"</td>").mkString(" ")+"</tr>"
      def block(tag:String, attrs:List[(String,String)], body:String) = {
        "<"+tag+attrs.map{case (k,v) => " "+k+"=\""+v+"\""}.mkString("")+">"+body+"</"+tag+">"
      }
      Utils.writeLines(path, { puts:(String=>Any) =>
        def putsBlock(tag:String, attrs:List[(String,String)]=Nil)(f: =>Any) = {
          puts("<"+tag+attrs.map{case (k,v) => " "+k+"=\""+v+"\""}.mkString("")+">")
          f
          puts("</"+tag+">")
        }
        puts("""<script src="table.js"></script>""")
        puts("""<link rel="stylesheet" type="text/css" href="table.css"/>""")

        if (description != "all") puts(block("h1", Nil, description))

        val classStr = "example table-autosort table-autofilter table-stripeclass:alternate table-autopage:10000 table-page-number:t1page table-page-count:t1pages table-filtered-rowcount:t1filtercount table-rowcount:t1allcount"
        putsBlock("table", List("id" -> "t1", "class" -> classStr)) {
          // Header
          putsBlock("thead") { putsBlock("tr", List("valign" -> "bottom")) {
            fields.zipWithIndex.foreach { case (field,j) =>
              MyUtils.findMap(records, { rec:Record =>
                rec.values(j).find(_ => true)
              }) match {
                case None => puts("<th nowrap>"+field.display+"</th>") // Uncommon (no valid values)
                case Some(v) =>
                  val classes = "table-filterable" :: {v match {
                    case _:NumValue => "table-sortable:numeric"
                    case _ => "table-sortable:default"
                  }} :: Nil
                  puts("<th nowrap class=\"%s\" valuetype=\"%s\">%s</th>".format(classes.mkString(" "), v.abs.render, field.display))
                }
              }
            }

            putsBlock("tr") {
              puts("""<td colspan="10000"><b>(<span id="t1filtercount"></span> matches)</b></td>""")
            }
          }

          putsBlock("tbody") {
            records.zipWithIndex.foreach { case (rec, i) =>
              val contents = (rec.values zip fields).map { case (vs,f) =>
                val sortkey = vs.map { v => f.sortkey(v) }.mkString(" ")
                val displayText = vs.map{v => f.humanRender(v)}.mkString(", ")
                block("td", List("sortkey" -> sortkey.toString), displayText)
              }.mkString("")
              puts(block("tr", if (i%2 == 0) Nil else List("class" -> "alternate"), contents))
            }
          }
        }
      })
    }

    if (description == "all") {
      val subdomains = new ArrayBuffer[String]
      allFields.slice(1, allFields.size).foreach { f1 =>
        allFields.slice(1, allFields.size).foreach { f2 =>
          if (f1.i < f2.i) {
            val subdomain = f1.predName+"-"+f2.predName
            generate(allFields.head :: f1 :: f2 :: Nil, path+subdomain+".html")
            subdomains += subdomain
          }
        }
      }
      Utils.writeLines((new File(path)).getParent+"/subdomains.list", { puts:(String=>Any) =>
        subdomains.foreach(puts)
      })
    }
    else
      generate(allFields, path)
  }

  def log = Utils.track("DataManager") {
    val sentenceLenFig = new fig.basic.FullStatFig
    val wordCounts = new HashMap[String,Double]
    def print(tag:String, examples:ArrayBuffer[Example]) = {
      if (examples.size > 0) {
        var numEmpty = 0
        var numErr = 0
        def processEx(ex:BaseExample) = {
          if (ex.trueAnswer != null) {
            if (ex.trueAnswer.size == 0) numEmpty += 1
            if (ex.trueAnswer.isError) numErr += 1
          }
          sentenceLenFig.add(ex.words.size)
          ex.words.foreach { w =>
            wordCounts(w) = wordCounts.getOrElse(w, 0.0) + 1
          }
        }
        examples.foreach {
          case ex:BaseExample => processEx(ex)
          case basket:Basket => basket.constraints.map(_.ex).foreach(processEx)
        }
        Utils.logs("%s %s examples: %s have empty denotations, %s have errors", examples.size, tag, numEmpty, numErr)
      }
    }
    print("general", generalExamples)
    print("train", trainExamples)
    print("test", testExamples)
    Utils.logs("Overall: %s word types, sentence length = %s", wordCounts.size, sentenceLenFig)

    Utils.writeLines(Execution.getFile("words.txt"), { (puts:String=>Any) =>
      wordCounts.toList.sortWith(_._2 > _._2).foreach { case (w,c) =>
        puts(Utils.fmts("%s\t%s", w, c))
      }
    })
  }

  def readExamples(path:String, continueFunc: =>Boolean, add:Example=>Any) = {
    def printEx(ex:Example) = {
      Utils.logs("Example %s: %s", ex.id, ex.summary)
    }
    var id = -1
    def newId = { id += 1; new java.io.File(path).getName+":"+id }
    if (path.endsWith(".dlog")) {
      val dlog = new DatalogInterpreter(U) {
        override def processSpecialStatement(path:String, statement:StatementItem, name:String, args:List[ExprItem]) = name match {
          case "_parse" => args match {
            case ListItem(wordItems) :: trueExpr :: rest =>
              val groupId = rest match {
                case StrItem(id) :: _ => id
                case _ => null
              }
              val words = wordItems.map(toString)
              if (words.size <= maxSentenceLength) {
                try { 
                  val ex = new BaseExample(newId, words, U.currWorld, new Answer(currWorldInt.executeQuerySafe(trueExpr)), groupId)
                  ex.trueExpr = trueExpr
                  printEx(ex)
                  add(ex)
                } catch {
                  case e: Exception => Utils.logs("Example FAILED TO EXECUTE")
                }
              }
            case _ => invalidUsage(statement)
          }
          case "_basket" => {
            val basket = new Basket(newId, U.currWorld)
            args.zipWithIndex.foreach {
              case (ListItem(StrItem(mode)::StrItem(sentence)::Nil), i) =>
                basket.addConstraint(mode, sentence)
              case _ => invalidUsage(statement)
            }
            basket.setLastAsAnswer
            add(basket)
          }
          case "_flushFields" => args match {
            case StrItem(description) :: StrItem(fileName) :: Nil =>
              generateDBHtml(description, new File(path).getParent+"/"+fileName)
              //U.currWorld.getExplicitPredicate(U.toPredName("field", 4)).clear
              U.currWorld.removePredicate(U.toPredName("field", 4))
            case _ => invalidUsage(statement)
          }
          case "_flushAllFieldPairs" => args match {
            case StrItem(filePrefix) :: Nil =>
              generateDBHtml("all", new File(path).getParent+"/"+filePrefix)
              //U.currWorld.getExplicitPredicate(U.toPredName("field", 4)).clear
              U.currWorld.removePredicate(U.toPredName("field", 4))
            case _ => invalidUsage(statement)
          }
          case _ => super.processSpecialStatement(path, statement, name, args)
        }
      }
      dlog.verbose = verbose
      dlog.process(path, continueFunc)
    }
    else if (path.endsWith(".nat")) { // Natural supervision: format: each line has a question and an answer, separated by a tab
      Utils.foreachLine(path, { line:String =>
        val basket = new Basket(newId, U.currWorld)
        line.split("\t").foreach { sentence =>
          basket.addConstraint("=", sentence)
        }
        basket.setLastAsAnswer
        add(basket)
        continueFunc
      })
    }
    else if (path.endsWith(".tree")) {
      val tdrt = new LogicalTreeParser(U) {
        override def continue = continueFunc
        override def newInstProcessor(inst:Array[Any]) = new InstProcessor(inst) {
          override def process = inst.head match {
            case "example" =>
              var words = hashArrInst("sentence").map(_.asInstanceOf[String]).toList
              val trueTree = buildNode(hashInst("tree"))
              if (words.size <= maxSentenceLength) {
                val id = if (hashInst("id") == null) newId else hashStrInst("id")
                val ex = new BaseExample(id, words, U.currWorld, new Answer(trueTree.computeConDen.main))
                ex.trueTree = trueTree
                Renderer.logTree("Tree", trueTree.toStrTree)
                printEx(ex)
                add(ex)
              }
            case _ => super.process
          }
        }
      }
      tdrt.verbose = verbose
      tdrt.readScript(path)
    }
    else if (path.endsWith(".lambda")) {
      // Format: lines alternating between sentence and lambda expression
      var sentence : String = null
      val dlog = new DatalogInterpreter(U)
      val lambdaParser = new LambdaCalculusParser(U)
      Utils.foreachLine(path, { line:String =>
        if (line != "") {
          if (sentence == null)
            sentence = line
          else {
            val formula = lambdaParser.parse(TSU.stringToNodes(line).head)
            val (trueExpr, ans) = try {
              val trueExpr = lambdaParser.to_dlog(formula)
              val pred = dlog.currWorldInt.executeQuerySafe(trueExpr)
              (trueExpr, new Answer(pred))
            } catch { case _:ConvertException =>
              (null, new Answer(null))
            }

            val ex = new BaseExample(newId, sentence.split(" ").toList, U.currWorld, ans)
            ex.trueExpr = trueExpr
            printEx(ex)
            add(ex)
            sentence = null
          }
        }
        continueFunc
      })
    }
    else
      throw Utils.fails("Invalid extension: %s", path)
  }
}
