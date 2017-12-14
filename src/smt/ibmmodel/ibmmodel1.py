#! /usr/bin/env python
# coding:utf-8

from operator import itemgetter
import collections
from smt.utils import utility
import decimal
from decimal import Decimal as D

# set deciaml context
decimal.getcontext().prec = 4
decimal.getcontext().rounding = decimal.ROUND_HALF_UP


def _constant_factory(value):
    '''define a local function for uniform probability initialization'''
    #return itertools.repeat(value).next
    return lambda: value


def _train(corpus, loop_count=100):
    f_keys = set()
    for (es, fs) in corpus:
        for f in fs:
            f_keys.add(f)
    # default value provided as uniform probability)
    t = collections.defaultdict(_constant_factory(D(1.0/(0.0001+len(f_keys)))))
    
    # loop
    for i in range(loop_count):
        count = collections.defaultdict(D)
        total = collections.defaultdict(D)
        s_total = collections.defaultdict(D)
        for (es, fs) in corpus:
            # compute normalization
            for e in es:
                s_total[e] = D()
                for f in fs:
                    s_total[e] += t[(e, f)]
            for e in es:
                for f in fs:
                    count[(e, f)] += t[(e, f)] / s_total[e]
                    total[f] += t[(e, f)] / s_total[e]
                    #if e == u"に" and f == u"always":
                    #    print(" BREAK:", i, count[(e, f)])
        # estimate probability
        for (e, f) in count.keys():
            #if count[(e, f)] == 0:
            #    print(e, f, count[(e, f)])
            t[(e, f)] = count[(e, f)] / total[f]

    return t


def train(sentences, loop_count=100):
    corpus = utility.mkcorpus(sentences)

    return _train(corpus, loop_count)


def _pprint(tbl):
    for (e, f), v in sorted(tbl.items(), key=itemgetter(1), reverse=True):
        print(u"p({e}|{f}) = {v}".format(e=e, f=f, v=v))


def test_train_loop1():
    sent_pairs = [("the house", "das Haus"),
                  ("the book", "das Buch"),
                  ("a book", "ein Buch"),
                  ]
    #t0 = train(sent_pairs, loop_count=0)
    t1 = train(sent_pairs, loop_count=1)

    loop1 = [(('house', 'Haus'), D("0.5")),
             (('book', 'ein'), D("0.5")),
             (('the', 'das'), D("0.5")),
             (('the', 'Buch'), D("0.25")),
             (('book', 'Buch'), D("0.5")),
             (('a', 'ein'), D("0.5")),
             (('book', 'das'), D("0.25")),
             (('the', 'Haus'), D("0.5")),
             (('house', 'das'), D("0.25")),
             (('a', 'Buch'), D("0.25"))]
    # assertion
    # next assertion doesn't make sence because
    # initialized by defaultdict
    #self.assertEqual(self._format(t0.items()), self._format(loop0))
    assert set(t1.items()) == set(loop1)


def test_train_loop2():
    sent_pairs = [("the house", "das Haus"),
                  ("the book", "das Buch"),
                  ("a book", "ein Buch"),
                  ]
    #t0 = train(sent_pairs, loop_count=0)
    t2 = train(sent_pairs, loop_count=2)

    loop2 = [(('house', 'Haus'), D("0.5713")),
             (('book', 'ein'), D("0.4284")),
             (('the', 'das'), D("0.6367")),
             (('the', 'Buch'), D("0.1818")),
             (('book', 'Buch'), D("0.6367")),
             (('a', 'ein'), D("0.5713")),
             (('book', 'das'), D("0.1818")),
             (('the', 'Haus'), D("0.4284")),
             (('house', 'das'), D("0.1818")),
             (('a', 'Buch'), D("0.1818"))]
    # assertion
    # next assertion doesn't make sence because
    # initialized by defaultdict
    #self.assertEqual(self._format(t0.items()), self._format(loop0))
    assert set(t2.items()) == set(loop2)


if __name__ == '__main__':
    import sys

    fd = open(sys.argv[1]) if len(sys.argv) >= 2 else sys.stdin
    sentences = [line.strip().split('|||') for line in fd.readlines()]
    
    t = train(sentences, loop_count=3)
    for (e, f), val in t.items():
        print("{} {}\t{}".format(e, f, val))
