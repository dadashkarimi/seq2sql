"""Check if two params files have all the same values."""
import numpy
import sys

import main as util  # Get all the requisite imports here

TOL=1e-10

def read(filename):
  spec = util.specutil.load(filename)
  params = spec.get_params()
  vals = [x.get_value() for x in params]
  return vals

def check(f1, f2):
  p1 = read(f1)
  p2 = read(f2)
  all_equal = True
  for x1, x2 in zip(p1, p2):
    diffs = numpy.abs(x1 - x2)
    is_equal = numpy.all(diffs < TOL)
    if not is_equal:
      all_equal = False
      print 'Not equal:'
      print x1
      print x2
  if all_equal:
    print 'All equal! (within %g)' % TOL

def main():
  if len(sys.argv) == 1:
    print >> sys.stderr, 'Usage: %s params_1 params_2' % sys.argv[0]
    sys.exit(1)
  check(*sys.argv[1:])

if __name__ == '__main__':
  main()
