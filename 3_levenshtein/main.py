import os
import random
import re
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
    def __init__(self):
        self.data_dir = "../datasets/ustawy"

    def list_paths(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                yield os.path.join(self.data_dir, filename)


class TextProcessor:
    NOT_POLISH_CHAR_RE = re.compile(r".*[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ].*")

    def __init__(self) -> None:
        self.nlp = Polish()
        self.tokenizer = Tokenizer(self.nlp.vocab)

    @classmethod
    def polish_is_alpha(cls, token):
        return not cls.NOT_POLISH_CHAR_RE.match(token)

    @classmethod
    def filter_tokens(cls, tokens):
        return [
            str(token).lower()
            for token in tokens
            if cls.polish_is_alpha(str(token)) and len(str(token)) > 3
        ]

    def get_word_frequency(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            tokens = self.tokenizer(f.read())
            tokens = self.filter_tokens(tokens)
            return Counter(tokens)


class CorpusProcessor:
    def __init__(self) -> None:
        self.corpus = Corpus()
        self.text_processor = TextProcessor()
        self.words_freq = None

    @staticmethod
    def calculate_term_rank(word_freq: Counter) -> List:
        items = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)
        return [(word, freq, i) for i, (word, freq) in enumerate(items)]

    @classmethod
    def plot_term_ranks(cls, word_ranks: List):
        ranks = [rank for _word, _freq, rank in word_ranks]
        freqs = [freq for _word, freq, _rank in word_ranks]
        plt.plot(ranks, freqs, scalex="log", scaley="log")
        plt.xlabel("Rank")
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def words_not_in_dictionary(word_freq: Counter):
        morf = morfeusz2.Morfeusz()
        analysed = morf.analyse(" ".join(word_freq.keys()))
        words = [word for (_, _, (word, _, wid, _, _)) in analysed if wid == "ign"]
        return set(words)

    @staticmethod
    def top_ranked_words(words: Set[str], ranks: List, n: int = 30):
        ranks = [(word, freq, rank) for (word, freq, rank) in ranks if word in words]
        ranks = sorted(ranks, key=lambda x: x[2])
        if not ranks:
            return []
        return [word for word, _freq, _rank in ranks[:n]]

    @staticmethod
    def random_words(words: Set[str], ranks: List, n: int = 30, occurrences: int = 5):
        ranks = [
            (word, freq, rank)
            for (word, freq, rank) in ranks
            if word in words and freq == occurrences
        ]
        random.shuffle(ranks)
        if not ranks:
            return []
        return [word for word, _freq, _rank in ranks[:n]]

    def find_corrections(self, words):
        words_freq = self.word_frequency()
        not_in_dict = self.words_not_in_dictionary(words_freq)
        predictions = {}
        for word in words:
            res_dist = distance(word, "")
            res_word = ""
            for w in words_freq.keys():
                if w in not_in_dict:
                    continue
                d = distance(word, w)
                if d < res_dist:
                    res_dist = d
                    res_word = w
            predictions[word] = res_word
        return predictions

    def word_frequency(self):
        if self.words_freq is not None:
            return self.words_freq

        with Pool(N_PROCESSES) as pool:
            docs_freq = pool.map(
                self.text_processor.get_word_frequency, self.corpus.list_paths()
            )

            self.words_freq = sum(docs_freq, start=Counter())
            return self.words_freq

    def words_to_correct(self):
        word_freq = self.word_frequency()
        ranks = self.__class__.calculate_term_rank(word_freq)
        # self.plot_term_ranks(ranks)

        not_in_dict = self.words_not_in_dictionary(word_freq)
        print(f"Words not in dictionary: {len(not_in_dict)}")
        top_ranked_words = self.top_ranked_words(not_in_dict, ranks)
        print("Top ranked to correct: ", top_ranked_words)
        random_words = self.random_words(not_in_dict, ranks)
        print("Random to correct: ", random_words)

        return set(top_ranked_words + random_words)


class Word(Document):
    """
    Document and Index definition.
    """

    @staticmethod
    def create_analyzer():
        return analyzer(
            "polish_analyzer",
            tokenizer="standard",
            filter=["lowercase", "morfologik_stem"],
        )

    body = Text(
        analyzer=create_analyzer(), term_vector="with_positions_offsets_payloads"
    )

    class Index:
        name = "word_index"
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }


class IndexClient:
    def __init__(self) -> None:
        self.connect()
        Word.init()

    def get_documents_count(self):
        return Search().count()

    def get_words(self):
        path = "../datasets/sgjp.tab"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                word, *_ = line.split("\t")
                yield Word(body=word).to_dict(include_meta=True)

    def connect(self):
        # Create connection for elasticsearch_dsl client
        connections.create_connection(hosts=["localhost"])
        # Create connection for elasticsearch client
        self.es = Elasticsearch(["localhost:9200"])

    def load_words(self, force=False):
        doc_count = self.get_documents_count()
        if doc_count == 0 or force:
            print("Loading documents.")
            bulk(self.es, self.get_words())

            # from collections import deque
            # deque(parallel_bulk(...), maxlen=0)

    def find_corrections(self, words: List[str]) -> Dict[str, str]:
        def fuzzy_query(word: str):
            return {"query": {"fuzzy": {"body": {"value": word}}}}

        def build_search(word: str):
            return Search().from_dict(fuzzy_query(word)).execute()

        res = {}
        for word in words:
            resp = build_search(word)
            res[word] = list(set([hit.body for hit in resp]))
        return res


if __name__ == "__main__":
    processor = CorpusProcessor()
    to_correct = processor.words_to_correct()
    levenshtein_corrections = processor.find_corrections(to_correct)

    index_client = IndexClient()
    index_client.load_words()

    elastic_corrections = index_client.find_corrections(to_correct)

    def merge_dicts(*corrs):
        res = defaultdict(lambda: dict())
        for name, corr in corrs:
            for key, value in corr.items():
                res[key][name] = value
        return res

    corrections = merge_dicts(
        ("levenshtein", levenshtein_corrections), ("elastic", elastic_corrections)
    )
    pprint(corrections)
