#! /usr/bin/env python
# coding:utf-8

from __future__ import division, print_function


def mkcorpus(sentences):
    """
    >>> sent_pairs = [("僕 は 男 です", "I am a man"),
                      ("私 は 女 です", "I am a girl"),
                      ("私 は 先生 です", "I am a teacher"),
                      ("彼女 は 先生 です", "She is a teacher"),
                      ("彼 は 先生 です", "He is a teacher"),
                      ]
    >>> pprint(mkcorpus(sent_pairs))
    [(['\xe5\x83\x95',
       '\xe3\x81\xaf',
       '\xe7\x94\xb7',
       '\xe3\x81\xa7\xe3\x81\x99'],
      ['I', 'am', 'a', 'man']),
     (['\xe7\xa7\x81',
       '\xe3\x81\xaf',
       '\xe5\xa5\xb3',
       '\xe3\x81\xa7\xe3\x81\x99'],
      ['I', 'am', 'a', 'girl']),
     (['\xe7\xa7\x81',
       '\xe3\x81\xaf',
       '\xe5\x85\x88\xe7\x94\x9f',
       '\xe3\x81\xa7\xe3\x81\x99'],
      ['I', 'am', 'a', 'teacher']),
     (['\xe5\xbd\xbc\xe5\xa5\xb3',
       '\xe3\x81\xaf',
       '\xe5\x85\x88\xe7\x94\x9f',
       '\xe3\x81\xa7\xe3\x81\x99'],
      ['She', 'is', 'a', 'teacher']),
     (['\xe5\xbd\xbc',
       '\xe3\x81\xaf',
       '\xe5\x85\x88\xe7\x94\x9f',
       '\xe3\x81\xa7\xe3\x81\x99'],
      ['He', 'is', 'a', 'teacher'])]
    """
    return [(es.split(), fs.split()) for (es, fs) in sentences]


def matrix(
        m, n, lst,
        m_text = None,
        n_text =None):
    """
    m: row
    n: column
    lst: items

    >>> print(_matrix(2, 3, [(1, 1), (2, 3)]))
    |x| | |
    | | |x|
    """
    print("Example:",[" ".join(m_text)],"\t",["".join(n_text)])
    fmt = ""
    if n_text:
        #fmt += "     {}\n".format(" ".join(n_text))
        fmt += "     {}\n".format("".join([" {0:<3.3}".format(x) for x in n_text]))
    for i in range(1, m+1):
        if m_text:
            fmt += "{:<4.4} ".format(m_text[i-1])
        fmt += "|"
        for j in range(1, n+1):
            if (i, j) in lst:
                fmt += " x |"
            else:
                fmt += "   |"
        fmt += "\n"
    return fmt
