#! /usr/bin/env python
# coding:utf-8


def phrase_extract(es, fs, alignment):
    ext = extract(es, fs, alignment)
    ind = {((x, y), (z, w)) for x, y, z, w in ext}
    es = tuple(es)
    fs = tuple(fs)
    return {(es[e_s-1:e_e], fs[f_s-1:f_e])
            for (e_s, e_e), (f_s, f_e) in ind}


def extract(es, fs, alignment):
    """
    caution:
        alignment starts from 1 - not 0
    """
    phrases = set()
    len_es = len(es)
    for e_start in range(1, len_es+1):
        for e_end in range(e_start, len_es+1):
            # find the minimally matching foreign phrase
            f_start, f_end = (len(fs), 0)
            for (e, f) in alignment:
                if e_start <= e <= e_end:
                    f_start = min(f, f_start)
                    f_end = max(f, f_end)
            phrases.update(_extract(es, fs, e_start,
                                    e_end, f_start,
                                    f_end, alignment))
    return phrases


def _extract(es, fs, e_start, e_end, f_start, f_end, alignment):
    if f_end == 0:
        return {}
    for (e, f) in alignment:
        if (f_start <= f <= f_end) and (e < e_start or e > e_end):
            return {}
    ex = set()
    f_s = f_start
    while True:
        f_e = f_end
        while True:
            ex.add((e_start, e_end, f_s, f_e))
            f_e += 1
            if f_e in list(zip(*alignment))[1] or f_e > len(fs):
                break
        f_s -= 1
        if f_s in list(zip(*alignment))[1] or f_s < 1:
            break
    return ex


def available_phrases(fs, phrases):
    """
    return:
        set of phrase indexed tuple like
            {((1, "I"), (2, "am")),
             ((1, "I"),)
             ...}
    """
    available = set()
    for i, f in enumerate(fs):
        f_rest = ()
        for fr in fs[i:]:
            f_rest += (fr,)
            if f_rest in phrases:
                available.add(tuple(enumerate(f_rest, i+1)))
    return available


def test_phrases():
    from smt.utils.utility import mkcorpus
    from smt.phrase.word_alignment import symmetrization

    sentenses = [("僕 は 男 です", "I am a man"),
                 ("私 は 女 です", "I am a girl"),
                 ("私 は 先生 です", "I am a teacher"),
                 ("彼女 は 先生 です", "She is a teacher"),
                 ("彼 は 先生 です", "He is a teacher"),
                 ]

    corpus = mkcorpus(sentenses)
    es, fs = ("私 は 先生 です".split(), "I am a teacher".split())
    alignment = symmetrization(es, fs, corpus)
    ext = phrase_extract(es, fs, alignment)
    ans = ("は 先生 です <-> a teacher",
           "先生 <-> teacher"
           "私 <-> I am"
           "私 は 先生 です <-> I am a teacher")
    for e, f in ext:
        print("{} {} {}".format(' '.join(e), "<->", ' '.join(f)))

    ## phrases
    fs = "I am a teacher".split()
    phrases = available_phrases(fs, [fs_ph for (es_ph, fs_ph) in ext])
    print(phrases)
    ans = {((1, 'I'), (2, 'am')),
           ((1, 'I'), (2, 'am'), (3, 'a'), (4, 'teacher')),
           ((4, 'teacher'),),
           ((3, 'a'), (4, 'teacher'))}

    phrases = available_phrases(fs, [fs_ph for (es_ph, fs_ph) in ext])
    assert ans == phrases


if __name__ == '__main__':

    # test2
    from smt.utils.utility import mkcorpus
    from word_alignment import alignment
    from smt.ibmmodel import ibmmodel2
    import sys

    delimiter = ","
    # load file which will be trained
    modelfd = open(sys.argv[1])
    sentenses = [line.rstrip().split(delimiter) for line
                 in modelfd.readlines()]
    # make corpus
    corpus = mkcorpus(sentenses)

    # train model from corpus
    f2e_train = ibmmodel2._train(corpus, loop_count=10)
    e2f_corpus = list(zip(*reversed(list(zip(*corpus)))))
    e2f_train = ibmmodel2._train(e2f_corpus, loop_count=10)

    # phrase extraction
    for line in sys.stdin:
        _es, _fs = line.rstrip().split(delimiter)
        es = _es.split()
        fs = _fs.split()

        f2e = ibmmodel2.viterbi_alignment(es, fs, *f2e_train).items()
        e2f = ibmmodel2.viterbi_alignment(fs, es, *e2f_train).items()
        align = alignment(es, fs, e2f, f2e)  # symmetrized alignment

        # output matrix
        #from smt.utils.utility import matrix
        #print(matrix(len(es), len(fs), align, es, fs))

        ext = phrase_extract(es, fs, align)
        for e, f in ext:
            print("{}{}{}".format(''.join(e), delimiter, ''.join(f)))
