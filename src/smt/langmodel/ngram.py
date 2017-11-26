#! /usr/bin/env python
# coding:utf-8

from __future__ import division, print_function
import itertools


class NgramException(Exception):
    pass


def ngram(sentences, n):
    s_len = len(sentences)
    if s_len < n:
        raise NgramException("the sentences length is not enough:\
                             len(sentences)={} < n={}".format(s_len, n))
    xs = itertools.tee(sentences, n)
    for i, t in enumerate(xs[1:]):
        for _ in xrange(i+1):
            next(t)
    return itertools.izip(*xs)


if __name__ == '__main__':
    pass
