#! /usr/bin/env python
# coding:utf-8

from __future__ import division, print_function
# import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, TEXT, REAL, INTEGER


class Tables(object):

    def get_sentence_table(self, tablename="sentence"):

        class Sentence(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            lang1 = Column(TEXT)
            lang2 = Column(TEXT)

        return Sentence

    def get_wordprobability_table(self, tablename):

        class WordProbability(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            transto = Column(TEXT)
            transfrom = Column(TEXT)
            prob = Column(REAL)

        return WordProbability

    def get_wordalignment_table(self, tablename):

        class WordAlignment(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            from_pos = Column(INTEGER)
            to_pos = Column(INTEGER)
            to_len = Column(INTEGER)
            from_len = Column(INTEGER)
            prob = Column(REAL)

        return WordAlignment

    def get_phrase_table(self, tablename="phrase"):

        class Phrase(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            lang1p = Column(TEXT)
            lang2p = Column(TEXT)

        return Phrase

    def get_transphraseprob_table(self, tablename="phraseprob"):

        class TransPhraseProb(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            lang1p = Column(TEXT)
            lang2p = Column(TEXT)
            p1_2 = Column(REAL)
            p2_1 = Column(REAL)

        return TransPhraseProb

    def get_trigram_table(self, tablename):

        class Trigram(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            first = Column(TEXT)
            second = Column(TEXT)
            third = Column(TEXT)
            count = Column(INTEGER)

        return Trigram

    def get_trigramprob_table(self, tablename):

        class TrigramProb(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            first = Column(TEXT)
            second = Column(TEXT)
            third = Column(TEXT)
            prob = Column(REAL)

        return TrigramProb

    def get_trigramprobwithoutlast_table(self, tablename):

        class TrigramProbWithoutLast(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            first = Column(TEXT)
            second = Column(TEXT)
            prob = Column(REAL)

        return TrigramProbWithoutLast

    def get_unigram_table(self, tablename):

        class Unigram(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            first = Column(TEXT)
            count = Column(INTEGER)

        return Unigram

    def get_unigramprob_table(self, tablename):

        class UnigramProb(declarative_base()):
            __tablename__ = tablename
            id = Column(INTEGER, primary_key=True)
            first = Column(TEXT)
            prob = Column(REAL)

        return UnigramProb
