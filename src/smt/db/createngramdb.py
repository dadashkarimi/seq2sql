#! /usr/bin/env python
# coding:utf-8

from __future__ import division, print_function
import collections
import sqlite3
# import SQLAlchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
# smt
from smt.db.tables import Tables
from smt.langmodel.ngram import ngram
import math


def _create_ngram_count_db(lang, langmethod=lambda x: x,
                           n=3, db="sqilte:///:memory:"):
    engine = create_engine(db)
    # create session
    Session = sessionmaker(bind=engine)
    session = Session()

    Sentence = Tables().get_sentence_table()
    query = session.query(Sentence)

    ngram_dic = collections.defaultdict(float)
    for item in query:
        if lang == 1:
            sentences = langmethod(item.lang1).split()
        elif lang == 2:
            sentences = langmethod(item.lang2).split()
        sentences = ["</s>", "<s>"] + sentences + ["</s>"]
        ngrams = ngram(sentences, n)
        for tpl in ngrams:
            ngram_dic[tpl] += 1

    return ngram_dic


def create_ngram_count_db(lang, langmethod=lambda x: x,
                          n=3, db="sqilte:///:memory:"):
    engine = create_engine(db)
    # create session
    Session = sessionmaker(bind=engine)
    session = Session()

    # trigram table
    tablename = 'lang{}trigram'.format(lang)
    Trigram = Tables().get_trigram_table(tablename)
    # create table
    Trigram.__table__.drop(engine, checkfirst=True)
    Trigram.__table__.create(engine)

    ngram_dic = _create_ngram_count_db(lang, langmethod=langmethod, n=n, db=db)

    # insert items
    for (first, second, third), count in ngram_dic.items():
        print(u"inserting {}, {}, {}".format(first, second, third))
        item = Trigram(first=first,
                       second=second,
                       third=third,
                       count=count)
        session.add(item)
    session.commit()


def create_unigram_count_db(lang, langmethod=lambda x: x,
                            db="sqilte:///:memory:"):
    engine = create_engine(db)
    # create session
    Session = sessionmaker(bind=engine)
    session = Session()

    # trigram table
    tablename = 'lang{}unigram'.format(lang)
    Sentence = Tables().get_sentence_table()
    Unigram = Tables().get_unigram_table(tablename)
    # create table
    Unigram.__table__.drop(engine, checkfirst=True)
    Unigram.__table__.create(engine)

    query = session.query(Sentence)
    ngram_dic = collections.defaultdict(int)
    for item in query:
        if lang == 1:
            sentences = langmethod(item.lang1).split()
        elif lang == 2:
            sentences = langmethod(item.lang2).split()
        ngrams = ngram(sentences, 1)
        for tpl in ngrams:
            ngram_dic[tpl] += 1

    # insert items
    for (first,), count in ngram_dic.items():
        print(u"inserting {}: {}".format(first, count))
        item = Unigram(first=first,
                       count=count)
        session.add(item)
    session.commit()


# create views using SQLite3
def create_ngram_count_without_last_view(lang, db=":memory:"):
    # create phrase_count table
    fromtablename = "lang{}trigram".format(lang)
    table_name = "lang{}trigram_without_last".format(lang)
    # create connection
    con = sqlite3.connect(db)
    cur = con.cursor()
    try:
        cur.execute("drop view {0}".format(table_name))
    except sqlite3.Error:
        print("{0} view does not exists.\n\
              => creating a new view".format(table_name))
    cur.execute("""create view {}
                as select first, second, sum(count) as count from
                {} group by first, second order by count
                desc""".format(table_name, fromtablename))
    con.commit()


def create_ngram_prob(lang,
                      db=":memory:"):

    # Create connection in sqlite3 to use view
    table_name = "lang{}trigram_without_last".format(lang)
    # create connection
    con = sqlite3.connect(db)
    cur = con.cursor()

    trigram_tablename = 'lang{}trigram'.format(lang)
    trigramprob_tablename = 'lang{}trigramprob'.format(lang)
    trigramprobwithoutlast_tablename = 'lang{}trigramprob_without_last'\
        .format(lang)

    # tables
    Trigram = Tables().get_trigram_table(trigram_tablename)
    TrigramProb = Tables().get_trigramprob_table(trigramprob_tablename)
    TrigramProbWithoutLast = Tables().get_trigramprobwithoutlast_table(
        trigramprobwithoutlast_tablename)

    # create connection in SQLAlchemy
    sqlalchemydb = "sqlite:///{}".format(db)
    engine = create_engine(sqlalchemydb)
    # create session
    Session = sessionmaker(bind=engine)
    session = Session()
    # create table
    TrigramProb.__table__.drop(engine, checkfirst=True)
    TrigramProb.__table__.create(engine)
    TrigramProbWithoutLast.__table__.drop(engine, checkfirst=True)
    TrigramProbWithoutLast.__table__.create(engine)

    # calculate total number
    query = session.query(Trigram)
    totalnumber = len(query.all())

    # get trigrams
    query = session.query(Trigram)
    for item in query:
        first, second, third = item.first, item.second, item.third
        count = item.count

        cur.execute("""select * from {} where \
                    first=? and\
                    second=?""".format(table_name),
                    (first, second))
        one = cur.fetchone()
        # if fetch is failed, one is NONE (no exceptions are raised)
        if not one:
            print("not found correspont first and second")
            continue
        else:
            alpha = 0.00017
            c = count
            n = one[2]
            v = totalnumber
            # create logprob
            logprob = math.log((c + alpha) / (n + alpha * v))
            print(u"{}, {}, {}:\
                  log({} + {} / {} + {} + {}) = {}".format(first,
                                                           second,
                                                           third,
                                                           c,
                                                           alpha,
                                                           n,
                                                           alpha,
                                                           v,
                                                           logprob))
            trigramprob = TrigramProb(first=first,
                                      second=second,
                                      third=third,
                                      prob=logprob)
            session.add(trigramprob)
            # for without last
            logprobwithoutlast = math.log(alpha / (n + alpha * v))
            print(u"{}, {}, {}:\
                  log({} / {} + {} + {}) = {}".format(first,
                                                      second,
                                                      third,
                                                      alpha,
                                                      n,
                                                      alpha,
                                                      v,
                                                      logprobwithoutlast))
            probwl = TrigramProbWithoutLast(first=first,
                                            second=second,
                                            prob=logprobwithoutlast)
            session.add(probwl)
    session.commit()


def create_unigram_prob(lang, db=":memory:"):

    unigram_tablename = 'lang{}unigram'.format(lang)
    unigramprob_tablename = 'lang{}unigramprob'.format(lang)

    # tables
    Unigram = Tables().get_unigram_table(unigram_tablename)
    UnigramProb = Tables().get_unigramprob_table(unigramprob_tablename)

    # create engine
    sqlalchemydb = "sqlite:///{}".format(db)
    engine = create_engine(sqlalchemydb)
    # create session
    Session = sessionmaker(bind=engine)
    session = Session()
    # create table
    UnigramProb.__table__.drop(engine, checkfirst=True)
    UnigramProb.__table__.create(engine)

    # calculate total number
    query = session.query(Unigram)
    sm = 0
    totalnumber = 0
    for item in query:
        totalnumber += 1
        sm += item.count

    # get trigrams
    query = session.query(Unigram)
    for item in query:
        first = item.first
        count = item.count

        alpha = 0.00017
        c = count
        v = totalnumber
        # create logprob
        logprob = math.log((c + alpha) / (sm + alpha * v))
        print(u"{}:\
              log({}+{} / {} + {}*{}) = {}".format(first,
                                                   c,
                                                   alpha,
                                                   sm,
                                                   alpha,
                                                   v,
                                                   logprob))
        unigramprob = UnigramProb(first=first,
                                  prob=logprob)
        session.add(unigramprob)
    session.commit()


def create_ngram_db(lang, langmethod=lambda x: x,
                    n=3, db=":memory:"):

    sqlalchemydb = "sqlite:///{}".format(db)
    create_ngram_count_db(lang=lang, langmethod=langmethod,
                          n=n,
                          db=sqlalchemydb)
    create_ngram_count_without_last_view(lang=lang, db=db)
    create_ngram_prob(lang=lang, db=db)

    create_unigram_count_db(lang=lang, langmethod=langmethod,
                            db=sqlalchemydb)
    create_unigram_prob(lang=lang, db=db)


if __name__ == '__main__':
    pass
