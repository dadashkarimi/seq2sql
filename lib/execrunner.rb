#!/usr/bin/ruby

"""
{09/09/11}
The execrunner library provides an easy way to manage a set of commands which share a lot of arguments.
For example, suppose you wanted to run the following
  my_program -numIters 5 -greedy false
  my_program -numIters 5 -greedy true
  my_program -numIters 10 -greedy false
  my_program -numIters 10 -greedy true

You can write the following snippet of Ruby:
  require 'execrunner'
  env!(
    run(
      'my_program',
      selo(nil, 'numIters', 5, 10),
      selo(nil, 'greedy', false, true),
    nil),
  nil)
Call this script r.

Type ./r -n to print out the four commands.
Type ./r to execute all of them sequentially.
Type ./r <additional arguments> to append additional arguments.

The general usage to try all settings is:
  selo(nil, argument, value1, ..., valuen)
To choose the i-th particular setting:
  selo(i, argument, value1, ..., valuen)

There are more advanced features, but this is just the basics.
"""

# Options:
# -n (print out the command that would be executed)
# -d (debug); otherwise, run it remotely using wq
# -t: run a thunk
# @<tag>: run commands only with the environment set

# Special variables (set via, for example, let(:tag, 'foobar'))
# tag: used to match the tag that's passed in via options
# easy: if command fails, press on
# appendArgs: arguments to put at very end

require 'myutils'
require 'runfig'

class Prog
  attr_accessor :x
  def initialize(x); @x = x; end
end
class Let
  attr_accessor :var, :value, :append
  def initialize(var, value, append); @var = var; @value = value; @append = append; end
end
class Prod
  attr_accessor :choices
  def initialize(choices); @choices = choices end
end
class Stop; end

class Env
  attr_accessor :list
  def initialize(isTerminal, list)
    @isTerminal = isTerminal
    @list = standarizeList(list.flatten)
  end

  def getRuns(runner, list=@list, args=[], bindings={})
    if list.size == 0
      runner.runArgs(args, bindings) if @isTerminal
      return
    end

    x, *rest = list
    case x
    when Array then
      getRuns(runner, x+rest, args, bindings)
    when Let then # Temporarily modify bindings
      oldvalue = bindings[x.var]
      #puts "Add #{oldvalue.inspect} #{x.value.inspect}"
      bindings[x.var] = x.append && oldvalue ? oldvalue+x.value : x.value
      getRuns(runner, rest, args, bindings)
      bindings[x.var] = oldvalue
    when Symbol # Substitute bindings with something else
      raise "Variable not bound: '#{x}'" unless bindings[x]
      getRuns(runner, [bindings[x]]+rest, args, bindings)
    when Prod then # Branch on several choices (each choice is a list)
      x.choices.each { |choice|
        getRuns(runner, choice+rest, args, bindings)
      }
    when Env then # Branch
      x.getRuns(runner, x.list, args, bindings)
      getRuns(runner, rest, args, bindings)
    when Prog then # Set the program
      getRuns(runner, rest, args+[x], bindings) # Just add to arguments
    when Proc then # Lazy value
      # Two ways to deal with lazy values: (example: run(lambda{f}, sel(nil,a,b)))
      #  1) evaluate f once for a and once for b (in this case, f can only contain primitive data)
      #  2) evaluate f once before the sel (f can contain complex things like another product)
      #     getRuns(runner, rest, args+[x], bindings) # Just add to arguments - evaluate later
      # We chose 2.
      getRuns(runner, [x.call(bindings)]+rest, args, bindings)
    else # String
      getRuns(runner, rest, args+[x.to_s], bindings) # Just add to arguments
    end
  end

  def to_s; "env(#{@list.size})" end
end

class ExecRunner
  attr :extraArgs

  def initialize(prog, extraArgs)
    @prog = prog
    setExtraArgs(extraArgs)
  end
  def setExtraArgs(extraArgs)
    @debug = extraArgs.member?("-d") # Just pass %debug to runfig
    @thunk = extraArgs.member?("-t") # Just pass %thunk to runfig
    @pretend = extraArgs.member?("-n") # Print out what would be passed to runfig (less detail than %pretend)
    @specifiedTags = extraArgs.map { |x| x =~ /^@(.+)$/ ? $1 : nil }.compact

    # Remove the options and tags that we just extracted
    @extraArgs = extraArgs.clone.delete_if { |x|
      x =~ /^-[dnt]$/ || x =~ /^@/
    }
  end

  # Specify tag to run a command
  def requireTags(v=true); @requireTags = v; nil end

  def memberOfAny(a, b)
    return false unless b
    b.each { |x|
      return true if a.member?(x)
    }
    false
  end

  # Run the command with the arguments
  def runArgs(args, bindings)
    # Skip if user specified some tags but none of the current tags
    # match any of the specified tags
    if @requireTags || @specifiedTags.size > 0
      return if (not memberOfAny(@specifiedTags, bindings[:tag]))
    end

    args = args.clone
    if @debug    then args << "%debug"
    elsif @thunk then args << "%thunk"
    end
    args += @extraArgs unless bindings[:ignoreExtraArgs]
    args = bindings[:prependArgs] + args if bindings[:prependArgs]
    args = args + bindings[:appendArgs] if bindings[:appendArgs]

    prog = args.reverse.find { |x| x.is_a?(Prog) }
    args.delete_if { |x| x.is_a?(Prog) }
    args = args.map { |x| x.is_a?(Proc) ? x.call : x } # Evaluate lazy items
    if prog
      if @pretend
        puts prog.x.inspect + " " + args.join(" ")
      else
        prog.x[:argv] = args
        runfig(prog.x)
      end
    else
      # If no explicit program given, just concatenate arguments together
      # (assume first argument is the program)
      if @pretend then
        puts args.join(' ') # Not quoted
      else
        success = ProcessManager::system(*args)
        if (not success) && (not bindings[:easy])
          puts "Command failed: #{args.join(' ')}"
          exit 1
        end
      end
    end
  end

  def execute(e); e.getRuns(self) end 
end

############################################################
# How to use:
# env!(
#   run(prog('grape'), o('maxIters', 4)),
#   run(prog(:className => 'grape.Main'), o('maxIters', 4)),
# nil)

$globalExecRunner = ExecRunner.new(nil, ARGV)

def env(*list); Env.new(false, list) end
def run(*list); Env.new(true, list) end
def env!(*x); $globalExecRunner.execute(env(*x)) end

def prog(*x); $globalExecRunner.prog(*x) end
def tag(v); let(:tag, [v], true) end
def easy(v=true); let(:easy, v) end
def ignoreExtraArgs(v=true); let(:ignoreExtraArgs, v) end
def requireTags(v=true); $globalExecRunner.requireTags(v) end
def appendArgs(*v); let(:appendArgs, v) end
def prependArgs(*v); let(:prependArgs, v) end
def note(*x); a('note', *x) end
def misc(*x); a('miscOptions', *x) end
def tagstr; lambda { |e| e[:tag].join(',') } end
def l(*list); standarizeList(list) end

# Options
def o(key, *values); optAppendOrNot(false, key, *values) end
def a(key, *values); optAppendOrNot(true, key, *values) end
def optAppendOrNot(append, key, *values)
  lambda { |e|
    values = standarizeList(values.flatten).map { |value| value && envEval(e, value).to_s }
    values = ['---']+values+['---'] if values.map { |x| x =~ /^-/ ? x : nil }.compact.size > 0 # Quote values: -x -0.5   =>   -x --- -0.5 ---
    ["#{append ? '+' : '-'}#{key}"] + values
  }
end

# Selection functions
def sel(i, *list)
  list = standarizeList(list)
  map = toMap(list)
  i == nil ? prod(*map.values) : lambda {|e| map[envEval(e,i)]}
end
def selo(i, name, *list); general_sel(i, name, list, false, lambda{|*z| o(*z)}) end
def selotag(i, name, *list); general_sel(i, name, list, true, lambda{|*z| o(*z)}) end
def sellet(i, name, *list); general_sel(i, name, list, false, lambda{|*z| let(*z)}) end
def sellettag(i, name, *list); general_sel(i, name, list, true, lambda{|*z| let(*z)}) end
def general_sel(i, name, list, useTags, baseFunc)
  # baseFunc is one of {opt,let}
  list = standarizeList(list)
  map = toMap(list)
  if i == nil # Product over all possible values
    values = isHash(list) ? map.values : list
    prod(*values.map {|v| l(baseFunc.call(name, v), useTags ? tag("#{name}=#{v}") : nil)})
  else
    lambda { |e|
      key = envEval(e,i)
      raise "Unknown key: #{envEval(e,i)}" unless map.has_key?(key)
      v = map[key]
      l(baseFunc.call(name, v), useTags ? tag("#{name}=#{v}") : nil)
    }
  end
end
def isHash(list); list.size == 1 && list[0].is_a?(Hash) end
def toMap(list)
  if isHash(list) then list[0]
  else h = {}; list.compact.each_with_index { |x,i| h[i] = x }; h
  end
end

def prog(x); Prog.new(x) end
def let(var, value, append=false); Let.new(var, value, append) end
def prod(*choices)
  Prod.new(standarizeList(choices).map {|choice| choice.class == Array ? choice : [choice]})
end
def stop; Stop.new end

# To use the same execrunner script on many machines,
# we need to dynamically figure out the view to add an execution to
# Assume $CLUSTER is set
# Example usages: view(3)
#                 view('s' => 3, 'r' => 5)
def view(map)
  if map.is_a?(Hash)
    x = map[ENV['CLUSTER']]
    x && o('addToView', x)
  else
    o('addToView', map)
  end
end
def tagview(x); l(view(x), tag(x)) end

# Helper function: evaluate v in environment e
def envEval(e,v)
  case v
  when Proc then envEval(e, v.call(e))
  when Symbol then envEval(e, e[v])
  else v
  end
end

# Helper function: compact the list of arguments
# Also, remove anything after stop object
def standarizeList(list)
  hasStopped = false
  list.map { |x|
    hasStopped = true if x.is_a?(Stop) 
    hasStopped ? nil : x
  }.compact
end
