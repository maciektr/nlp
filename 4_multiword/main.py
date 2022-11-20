import math
import os
import pickle
import random
import re
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from multiprocessing import Pool
from pprint import pprint
from typing import Dict, List, Optional, Set

import morfeusz2
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Document, Search, Text, analyzer, connections
from Levenshtein import distance
from matplotlib import pyplot as plt
from spacy.lang.pl import Polish
from spacy.tokenizer import Tokenizer

N_PROCESSES = 32


class Corpus:
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)

    def list_paths(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                yield os.path.join(self.data_dir, filename)


class TextProcessor:
    def __init__(self) -> None:
        self.nlp = Polish()
        self.tokenizer = Tokenizer(self.nlp.vocab)

    def get_tokens(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            tokens = self.tokenizer(f.read())
            tokens = [str(token).lower() for token in tokens]
            return tokens


class ClarinTextProcessor:
    def __init__(self) -> None:
        pass

    def get_tokens(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
            root = ET.fromstring(text)
            tokens = []
            for r in root.findall(".//*/lex"):
                base = str(r.find("base").text).lower()
                ctag = str(r.find("ctag").text).lower()
                if ctag:
                    ctag = ctag.split(":")[0]
                tokens.append(f"{base}:{ctag}")
            return tokens


class CorpusProcessor:
    NOT_POLISH_CHAR_RE = re.compile(r".*[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ].*")

    @classmethod
    def polish_is_alpha(cls, token):
        return not cls.NOT_POLISH_CHAR_RE.match(token)

    def __init__(self, corpus, text_processor=None) -> None:
        self.corpus = corpus
        self.text_processor = text_processor
        if not text_processor:
            self.text_processor = TextProcessor()

        self.tokens = None

    def get_tokens(self):
        if self.tokens is not None:
            return self.tokens

        cache_file_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, self.corpus.data_dir)
        tokens_pickle_filename = f".{cache_file_uuid}.cache.pkl"

        if os.path.isfile(tokens_pickle_filename):
            print("Loading tokens from cache")
            with open(tokens_pickle_filename, "rb") as f:
                self.tokens = pickle.load(f)
            return self.tokens

        with Pool(N_PROCESSES) as pool:
            docs_tokens = pool.map(
                self.text_processor.get_tokens, self.corpus.list_paths()
            )

            self.tokens = sum(docs_tokens, start=[])
            with open(tokens_pickle_filename, "wb") as f:
                pickle.dump(self.tokens, f)

            return self.tokens

    @classmethod
    def filter_tokens(cls, key):
        def filter_word(word):
            word_split = str(word).split(":")
            if len(word_split) > 1:
                word = word_split[-2]
            return cls.polish_is_alpha(word) and len(word) > 1

        return filter_word(key[0]) and filter_word(key[1])

    def get_bigram(self):
        tokens = self.get_tokens()
        bigram = Counter(list(zip(tokens, tokens[1:])))
        bigram = {k: v for k, v in bigram.items() if self.__class__.filter_tokens(k)}
        return bigram

    def get_word_freq(self):
        tokens = self.get_tokens()
        word_freq = Counter(tokens)
        return word_freq

    def get_pmi(self, min_occurrences=0):
        bigram = self.get_bigram()
        word_freq = self.get_word_freq()
        pmi = {}

        for (w1, w2), freq in bigram.items():
            if freq < min_occurrences:
                continue

            pmi[(w1, w2)] = math.log2(freq / (word_freq[w1] * word_freq[w2]))

        return pmi

    def get_pmi_top10(self, n=10, min_occurrences=0):
        pmi = self.get_pmi(min_occurrences)
        pmi_top10 = sorted(pmi.items(), key=lambda x: x[1], reverse=True)[:n]
        return pmi_top10


if __name__ == "__main__":
    print("Spacy:")
    processor = CorpusProcessor(corpus=Corpus("../datasets/ustawy"))
    tokens = processor.get_tokens()
    print(len(tokens))
    # print(processor.get_bigram())
    print("Top 10 PMI: \n", processor.get_pmi_top10())
    print(
        "Top 10 PMI, occurrences >= 5: \n", processor.get_pmi_top10(min_occurrences=5)
    )

    print("\nClarin:")
    processor = CorpusProcessor(
        text_processor=ClarinTextProcessor(),
        corpus=Corpus("../datasets/ustawy_tagged_clarin"),
    )
    tokens = processor.get_tokens()
    print(len(tokens))
    # print(processor.get_bigram())
    print("Top 10 PMI: \n", processor.get_pmi_top10())
    print(
        "Top 10 PMI, occurrences >= 5: \n", processor.get_pmi_top10(min_occurrences=5)
    )
