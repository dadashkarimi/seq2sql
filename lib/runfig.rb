# Provides a wrapper around a Java or Scala program that uses the fig Java libraries.
# Relevant environment variables: JAVA_OPTS
# Special arguments (which are parsed by this script):
#  - %nojar (disable jar files)
#  - %pretend (don't actually run)
#  - %help (display help)
#  - %prof, %hprof (run profile)
#  - %q (use q to run the job)
# {06/18/11}: vastly simplified this script (at the expense of backward compatability).

# Additional jar files will be prepended before fig.jar and <package>.jar
def runfig(options)
  className = options[:className] or raise "No class name" # Main entry point
  lang = options[:lang] || 'java'
  argv = options[:argv] || ARGV
  jarFiles = options[:jarFiles] || [] # These are added to the classpath and pre-loaded (so they can be replaced without crashing running programs)
  javaOpts = options[:javaOpts]
  mem = options[:mem]

  #ENV['CLASSPATH'] += ':'+ENV['SCALA_HOME']+"/lib/scala-library.jar" if options[:lang] == 'scala'

  # Java opts
  javaArgs = []
  javaArgs << "-Xprof" if argv.member?("%prof")
  javaArgs << "-Xrunhprof:cpu=samples,depth=20" if argv.member?("%hprof")
  #javaArgs << "-Xrunhprof:cpu=samples" if argv.member?("%hprof")
  #javaArgs << "-Xrunhprof:heap=all" if argv.member?("%hprof")
  javaArgs += ["-ea", "-server", '-Xss8m']
  javaArgs += ['-Xmx'+mem] if mem
  javaArgs += javaOpts.split if javaOpts

  # Build arguments
  args = ['java'] + javaArgs
  args += ["-cp", (jarFiles + (ENV['CLASSPATH']||"").split).join(":")]
  args << className
  args += ["-create", "-monitor"] if not argv.member?('%debug')
  if argv.member?('%q')
    args += ["-execDir", "//_OUTPATH_", "-overwrite"]
  else
    args += ["-useStandardExecPoolDirStrategy"]
  end
  args += ["-jarFiles"] + jarFiles[0...1] # Assume that only the first one is interesting and subject to change...
  if argv.member?('%q')
    memArgs = mem ? ['-mem', mem] : []
    args = ['q'] + memArgs + ['-add', '---'] + args
  elsif argv.member?('%wq') # Run remotely using workqueue
    args = ['wq'] + args
  end

  args += argv.select{|x| x !~ /^%/} # Remove the script options

  # Build command; quote arguments that contain spaces or other weird characters
  def quote(a); a.map { |s| s =~ /[^\w:,\.+-\/]/ ? "'"+s+"'" : s }.join(" ") end
  cmd = quote(args)

  # Execute!
  if argv.member?('%pretend')
    puts cmd
  else
    system cmd
  end
end
