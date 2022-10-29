import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import (
    Document,
    Search,
    Text,
    analyzer,
    connections,
    token_filter,
)


class Act(Document):
    """
    Document and Index definition.
    """

    @staticmethod
    def create_analyzer():
        synonym_filter = token_filter(
            "polish_synonym_filter",
            "synonym",
            synonyms=[
                "kpk => kodeks postępowania karnego",
                "kpc => kodeks postępowania cywilnego",
                "kk => kodeks karny",
                "kc => kodeks cywilny",
            ],
        )

        return analyzer(
            "polish_analyzer",
            tokenizer="standard",
            filter=["lowercase", synonym_filter, "morfologik_stem"],
        )

    body = Text(
        analyzer=create_analyzer(), term_vector="with_positions_offsets_payloads"
    )

    class Index:
        name = "act_index"
        settings = {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }


class Corpus:
    def __init__(self):
        self.data_dir = os.path.abspath("./../datasets/ustawy")

    def list_all(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                yield os.path.join(self.data_dir, filename)

    def size(self):
        return len([*self.list_all()])

    def list_articles(self):
        for filename in self.list_all():
            with open(filename, "r") as f:
                yield Act(body=f.read()).to_dict(include_meta=True)


class IndexClient:
    def __init__(self) -> None:
        self.corpus = Corpus()
        self.connect()
        Act.init()

    def connect(self):
        # Create connection for elasticsearch_dsl client
        connections.create_connection(hosts=["localhost"])
        # Create connection for elasticsearch client
        self.es = Elasticsearch(["localhost:9200"])

    def get_documents_count(self):
        return Search().count()

    def load_files(self, force=False):
        doc_count = self.get_documents_count()
        if doc_count == 0 or force:
            print("Loading documents.")
            bulk(self.es, self.corpus.list_articles())

    def get_term_vector(self, body: str):
        req_body = {
            "doc": {"body": body},
            "offsets": True,
            "payloads": True,
            "positions": True,
            "term_statistics": True,
            "field_statistics": True,
        }
        data = index_client.es.termvectors(index="act_index", body=req_body)

        return data


if __name__ == "__main__":
    index_client = IndexClient()
    index_client.load_files()

    corpus_size = index_client.corpus.size()

    def count_as_percentage(count):
        return round(count / corpus_size * 100, 2)

    ustawa_count = Search().query("match", body="ustawa").count()
    print(
        'Number of acts containing the word "ustawa": ',
        ustawa_count,
        "; ",
        count_as_percentage(ustawa_count),
        "%",
    )

    kpc_count = (
        Search().query("match_phrase", body="kodeks postępowania cywilnego").count()
    )
    print(
        'Number of acts containing the phrase "kodeks postepowania cywilnego": ',
        kpc_count,
        "; ",
        count_as_percentage(kpc_count),
        "%",
    )

    def match_phrase_with_slop(query, slop):
        return {"query": {"match_phrase": {"body": {"query": query, "slop": slop}}}}

    wchodzi_w_zycie_count = (
        Search().from_dict(match_phrase_with_slop("wchodzi w życie", 2)).count()
    )

    print(
        'Number of acts containing the phrase "wchodzi w życie": ',
        wchodzi_w_zycie_count,
        "; ",
        count_as_percentage(wchodzi_w_zycie_count),
        "%",
    )

    konstytucja = (
        Search()
        .query("match", body="konstytucja")
        .highlight("body", fragment_size=50)
        .execute()
    )

    print("-" * 15)
    print('Ten most relevant documents for the word "konstytucja"')
    for hit in konstytucja:
        print("Document id: ", hit.meta.id)
        print("Score: ", hit.meta.score)
        print("Excerpts: ", hit.meta.highlight.body[:3])
        print("-" * 15)

    data = index_client.get_term_vector("ustawa")
    ustawa_occ_count = data["term_vectors"]["body"]["terms"]["ustawa"]["ttf"]
    print('Number of occurrences of the word "ustawa": ', ustawa_occ_count)

    data = index_client.get_term_vector("ustaw")
    ustaw_occ_count = (
        data["term_vectors"]["body"]["terms"]["ustawa"]["ttf"]
        + data["term_vectors"]["body"]["terms"]["ustawić"]["ttf"]
    )
    print(
        'Number of occurrences of the word "ustaw" in any inflectional form: ',
        ustaw_occ_count,
    )
