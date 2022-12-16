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
import nltk
from nltk.collocations import *
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Document, Search, Text, analyzer, connections
from Levenshtein import distance
from matplotlib import pyplot as plt
from spacy.lang.pl import Polish
from spacy.tokenizer import Tokenizer
from lpmn_client import download_file, upload_file
from lpmn_client import Task
from pprint import pprint
from matplotlib import pyplot as plt

N_PROCESSES = 32


def is_top_50(filename: str) -> bool:
    # ! ls -laSh ../datasets/ustawy | head -n 51 | tail -n 50 | awk '{print $NF}'
    bill_files = [
        "1994_195.txt",
        "1994_591.txt",
        "1996_110.txt",
        "1996_460.txt",
        "1996_465.txt",
        "1996_496.txt",
        "1996_561.txt",
        "1997_117.txt",
        "1997_153.txt",
        "1997_349.txt",
        "1997_553.txt",
        "1997_555.txt",
        "1997_557.txt",
        "1997_714.txt",
        "1997_926.txt",
        "1998_1118.txt",
        "1998_602.txt",
        "1999_930.txt",
        "1999_95.txt",
        "2000_1104.txt",
        "2000_1186.txt",
        "2000_1268.txt",
        "2000_1315.txt",
        "2000_136.txt",
        "2000_696.txt",
        "2000_991.txt",
        "2001_1070.txt",
        "2001_1188.txt",
        "2001_1229.txt",
        "2001_1368.txt",
        "2001_1381.txt",
        "2001_1438.txt",
        "2001_1444.txt",
        "2001_1545.txt",
        "2001_475.txt",
        "2001_499.txt",
        "2001_627.txt",
        "2001_628.txt",
        "2001_906.txt",
        "2001_92.txt",
        "2002_1689.txt",
        "2003_1750.txt",
        "2003_2256.txt",
        "2003_2277.txt",
        "2003_423.txt",
        "2004_1693.txt",
        "2004_177.txt",
        "2004_2065.txt",
        "2004_2533.txt",
        "2004_880.txt",
    ]
    return filename in bill_files or filename.split("%")[-1] in bill_files


class Corpus:
    def __init__(self, data_dir: str):
        self.data_dir = os.path.abspath(data_dir)

    def top_50_largest(self, file_name: str) -> bool:
        return is_top_50(file_name)

    def list_paths(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt") and self.top_50_largest(filename):
                yield os.path.join(self.data_dir, filename)


class ClarinTextProcessor:
    NOT_POLISH_CHAR_RE = re.compile(r".*[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ].*")

    def __init__(self) -> None:
        pass

    @classmethod
    def filter_word(cls, word):
        return not cls.NOT_POLISH_CHAR_RE.match(word) and len(word) > 1

    def get_tokens(self, filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
            root = ET.fromstring(text)
            tokens = []
            for s in root.findall(".//*/sentence"):
                sentence = []
                for r in s.findall("tok"):
                    #    <tok>
                    #     <orth>Polskiej</orth>
                    #     <lex disamb="1"><base>polski</base><ctag>adj:sg:gen:f:pos</ctag></lex>
                    #    </tok>
                    orth = str(r.find("orth").text)
                    lex = r.find("lex")
                    base = str(lex.find("base").text).lower()
                    # ctag = str(r.find("ctag").text).lower()
                    # if ctag:
                    #     ctag = ctag.split(":")[0]
                    # sentence.append(f"{orth}:{base}")
                    if self.__class__.filter_word(base):
                        sentence.append((orth, base))
                tokens.append(sentence)
            return tokens


class CorpusProcessor:
    def __init__(self, corpus, text_processor=None) -> None:
        self.corpus = corpus
        self.text_processor = text_processor
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

    def find_entries(self):
        tokens = self.get_tokens()
        result = []
        for sentence in tokens:
            entries = []
            for word, base in sentence[1:]:
                if word[0].isupper():
                    entries.append((word, base))
                elif len(entries) > 0:
                    result.append(entries)
                    entries = []
            if len(entries) > 0:
                result.append(entries)
        return result

    def get_entries_freq(self):
        entries = self.find_entries()
        entries = [[base for _word, base in sentence] for sentence in entries]
        entries = [tuple(sentence) for sentence in entries]
        return Counter(entries)


class NerClient:
    output_dir = "./liner_out"

    def __init__(self) -> None:
        self.task = Task(lpmn='any2txt|wcrft2|liner2({"model":"n82"})')
        self.task.email = "mtratnow@student.agh.edu.pl"
        self.token_classes = None

        os.makedirs(self.__class__.output_dir, exist_ok=True)

    def list_output_files(self):
        output_files = list(os.listdir(self.__class__.output_dir))
        dirname = output_files[0].split(".")[0]
        dirpath = os.path.abspath(os.path.join(self.__class__.output_dir, dirname))
        filenames = [
            filename for filename in os.listdir(dirpath) if is_top_50(filename)
        ]
        return [os.path.join(dirpath, filename) for filename in filenames]

    def process(self, corpus_path: str):
        if len(self.list_output_files()) > 0:
            print("Liner output already downloaded.")
            return
        print("Calling liner2 API.")
        file_id = upload_file(corpus_path)
        output_file_id = self.task.run(file_id)
        download_file(output_file_id, self.__class__.output_dir)
        print("Liner output downloaded.")

    @staticmethod
    def process_token_classes_file(filename: str):
        with open(filename, "r", encoding="utf-8") as f:
            text = f.read()
            root = ET.fromstring(text)
            classes = []

            # for s in root.findall(".//*/ann"):
            #     classes.append(s.attrib["chan"])

            for token in root.findall(".//*/tok"):
                entity = defaultdict(lambda: [])
                #    <tok>
                #     <orth>Minister</orth>
                #     <lex disamb="1"><base>minister</base><ctag>subst:sg:nom:m1</ctag></lex>
                #     <ann chan="nam_org_institution" head="1">1</ann>
                #    </tok>
                #    <tok>
                #     <orth>Finansów</orth>
                #     <lex disamb="1"><base>finanse</base><ctag>subst:pl:gen:n</ctag></lex>
                #     <ann chan="nam_org_institution">1</ann>
                #    </tok>
                #    <ns/>
                #    <tok>
                #     <orth>,</orth>
                #     <lex disamb="1"><base>,</base><ctag>interp</ctag></lex>
                #     <ann chan="nam_org_institution">0</ann>
                #    </tok>
                orth = str(token.find("orth").text)
                lex = token.find("lex")
                base = str(lex.find("base").text).lower()

                for ann in token.findall("ann"):
                    chan = ann.attrib["chan"]
                    value = int(ann.text)

                    if value == 0:
                        if len(entity[chan]) > 0:
                            classes.append((chan, entity[chan]))
                            entity[chan] = []
                        continue

                    entity[chan].append(orth)

                for chan, values in entity.items():
                    if len(values) > 0:
                        classes.append((chan, " ".join(entity[chan])))
                # tokens.append(sentence)
            return classes

    def get_token_classes(self):
        if self.token_classes is not None:
            return self.token_classes

        cache_file_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, "liner_token_classes")
        tokens_pickle_filename = f".{cache_file_uuid}.cache.pkl"

        if os.path.isfile(tokens_pickle_filename):
            print("Loading token classes from cache")
            with open(tokens_pickle_filename, "rb") as f:
                self.token_classes = pickle.load(f)
            return self.token_classes

        with Pool(N_PROCESSES) as pool:
            docs_tokens = pool.map(
                self.__class__.process_token_classes_file, self.list_output_files()
            )

            self.token_classes = sum(docs_tokens, start=[])
            with open(tokens_pickle_filename, "wb") as f:
                pickle.dump(self.token_classes, f)

            return self.token_classes


class ClassesProcessor:
    def __init__(self, values: List):
        self.values = values

    @staticmethod
    def coarse_grained(classname):
        return "_".join(classname.split("_")[:2])

    def plot_histogram(self, bins=50):
        def do_plot(values, suffix=""):
            plt.figure(figsize=(20, 15), dpi=200)
            plt.hist(values, bins=bins)
            plt.ylabel("Count")
            plt.xlabel("Value")
            plt.title("Token classes histogram")
            plt.xticks(rotation="vertical")
            plt.tight_layout()
            output = f"hist{suffix}.png"
            if not os.path.isfile(output):
                plt.savefig(output)
            # plt.show()

        values = self.get_classnames()
        do_plot(values, "1")
        values = [self.__class__.coarse_grained(v) for v in values]
        do_plot(values, "2")

    @staticmethod
    def group(values):
        res = defaultdict(lambda: [])
        for classname, entity in values:
            res[classname].append(entity)
        return res

    def get_classnames(self):
        return [v for v, _e in self.values]

    def get_coarse_grained(self):
        return [(self.__class__.coarse_grained(v), e) for v, e in self.values]

    def group_coarse_grained(self):
        return self.__class__.group(self.get_coarse_grained())

    def count_coarse_grained(self):
        grouped = self.group_coarse_grained()
        return {classname: Counter(entities) for classname, entities in grouped.items()}

    def print_top_k_coarse_grained(self, k=10):
        count_coarse_grained = self.count_coarse_grained()
        print(f"Top {k} classes:")
        for classname, counter in count_coarse_grained.items():
            print(classname)
            pprint(counter.most_common(k))

    def count_entities(self):
        return Counter(self.get_coarse_grained())

    def print_top_k_entities(self, k=50):
        count_entities = self.count_entities()
        print(f"Top {k} entities:")
        pprint(count_entities.most_common(k))


if __name__ == "__main__":
    print("\nClarin:")
    processor = CorpusProcessor(
        text_processor=ClarinTextProcessor(),
        corpus=Corpus("../datasets/ustawy_tagged_clarin"),
    )
    tokens = processor.get_tokens()
    print("Tokens count", len(tokens))
    # print(tokens[:100])
    entries = processor.find_entries()
    print("Entries count", len(entries))
    # print(entries[:100])
    entries_freq = processor.get_entries_freq()
    print("Most common entries:")
    pprint(entries_freq.most_common(50))

    ner_client = NerClient()
    ner_client.process("../datasets/ustawy.zip")

    classes = ner_client.get_token_classes()
    print(classes[:50])

    # plot_histogram([chan for chan, _entity in classes])

    classes_processor = ClassesProcessor(classes)
    classes_processor.plot_histogram()
    classes_processor.print_top_k_coarse_grained(10)
    classes_processor.print_top_k_entities()
