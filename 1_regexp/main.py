import os
import re
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List

import pandas as pd
from matplotlib import pyplot as plt

N_PROCESSES = 32


class Corpus:
    def __init__(self):
        self.data_dir = "./ustawy"

    def list_all(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                yield os.path.join(self.data_dir, filename)


class Patterns:
    UNITS = [
        "rozdziału",
        "ustawy",
        "art",
        "lit",
        "lit",
        "pkt",
        "poz",
        "punkt",
        "rozdział",
        "Rozdział",
        "dział",
        "tiret",
        "ust",
        "usta",
        "ustawa",
        "preambułę",
        "§",
    ]

    YEAR_LOOKUP = r"z\s+dnia\s+\d\d?\s+\w+\s+(\d{4})"
    UNITS_REGEX = r"(" + "|".join(UNITS) + r")\.?"
    UNIT_ID = r"\s*\d*\w*(\{\w*\d*\w*\})?\s*"

    @classmethod
    def addition(cls) -> str:
        return r"dodaje\s+się\s+(" + cls.UNITS_REGEX + r")"

    @classmethod
    def removal(cls) -> str:
        removal_base = r"((po\s+wyrazach\s+\".*\"\s*)?skreśla\s+się)"
        unit_in_front = cls.UNITS_REGEX + cls.UNIT_ID + removal_base
        unit_in_back = removal_base + r"\s+" + cls.UNITS_REGEX
        return r"(" + unit_in_front + r")|(" + unit_in_back + r")"

    @classmethod
    def change(cls) -> str:
        unit = r"(" + cls.UNITS_REGEX + cls.UNIT_ID + r")"
        base = r"otrzymuje\s+brzmienie.*"
        loc = r"\s*(zdanie\s+\w+)?(\w+\s+zdanie)?(w\s+dziale\s+\w+)?\s*"
        return unit + loc + base


class YearLookup:
    def __init__(self):
        self.regex = re.compile(Patterns.YEAR_LOOKUP, re.IGNORECASE)

    def __call__(self, text: str) -> str:
        return self.regex.search(text).groups(1)[0]


class ReCounter:
    def __init__(self, raw_re: str):
        self.expr = re.compile(raw_re, re.IGNORECASE) if raw_re else None

    def findall(self, text: str):
        if not self.expr:
            return []
        return self.expr.findall(text)

    def __call__(self, text: str):
        all = self.findall(text)
        return len(all)


@dataclass
class BillStats:
    year: str
    additions: int = 0
    removals: int = 0
    changes: int = 0

    def add(self, stats: "BillStats"):
        self.additions += stats.additions
        self.removals += stats.removals
        self.changes += stats.changes


class BillProcessor:
    year_lookup = YearLookup()
    addition_counter = ReCounter(Patterns.addition())
    removal_counter = ReCounter(Patterns.removal())
    change_counter = ReCounter(Patterns.change())

    def __init__(
        self,
        text: str,
    ):
        self.text = text

    def year(self):
        return self.__class__.year_lookup(self.text)

    def additions(self):
        return self.__class__.addition_counter(self.text)

    def removals(self):
        return self.__class__.removal_counter(self.text)

    def changes(self):
        return self.__class__.change_counter(self.text)

    def stats(self) -> BillStats:
        return BillStats(
            year=self.year(),
            additions=self.additions(),
            removals=self.removals(),
            changes=self.changes(),
        )


def process_file(filename: str) -> BillStats:
    with open(filename, "r") as f:
        return BillProcessor(f.read()).stats()


class CorpusProcessor:
    class StatsByYear:
        def __init__(self):
            self.stats = {}

        def add(self, stats: BillStats):
            year = stats.year
            if year not in self.stats:
                self.stats[year] = stats
                return
            self.stats[year].add(stats)

        def get_all(self) -> List[BillStats]:
            return list(self.stats.values())

        def plot(self):
            df = pd.DataFrame(self.get_all())
            df = df.set_index("year")
            df = df.div(df.sum(axis=1), axis=0)
            df = df.reset_index()
            df.plot(x="year")
            plt.show()

    def __init__(self) -> None:
        self.corpus = Corpus()
        self.stats_by_year = self.__class__.StatsByYear()

    def process(self):
        with Pool(N_PROCESSES) as pool:
            stats = pool.map(process_file, self.corpus.list_all())
            for bill_stats in stats:
                self.stats_by_year.add(bill_stats)
            print(self.stats_by_year.get_all())
            self.stats_by_year.plot()


if __name__ == "__main__":
    CorpusProcessor().process()
