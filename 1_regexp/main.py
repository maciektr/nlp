import os
import re
from dataclasses import dataclass
from multiprocessing import Pool
from pprint import pprint
from typing import List, Optional

import pandas as pd
from matplotlib import pyplot as plt

N_PROCESSES = 16


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
    ACT = [
        "ustaw",
        "ustawa",
        "ustawą",
        "ustawach",
        "ustawami",
        "ustawę",
        "ustawie",
        "ustawo",
        "ustawom",
        "ustawy",
    ]

    FROM_DAY = r"\s+z\s+dnia\b"
    YEAR_LOOKUP = r"z\s+dnia\s+\d\d?\s+\w+\s+(\d{4})"
    UNIT_ID = r"\s*\d*\w*(\{\w*\d*\w*\})?\s*"
    UNITS_REGEX = r"(" + "|".join(UNITS) + r")\.?"
    ACT_REGEX = r"\b(" + "|".join(ACT) + r")\b"

    @classmethod
    def addition(cls) -> str:
        return r"\bdodaje\s+się\s+(" + cls.UNITS_REGEX + r")\b"

    @classmethod
    def removal(cls) -> str:
        removal_base = r"((po\s+wyrazach\s+\".*\"\s*)?skreśla\s+się)"
        unit_in_front = cls.UNITS_REGEX + cls.UNIT_ID + removal_base
        unit_in_back = removal_base + r"\s+" + cls.UNITS_REGEX
        return r"\b(" + unit_in_front + r")|(" + unit_in_back + r")\b"

    @classmethod
    def change(cls) -> str:
        unit = r"\b(" + cls.UNITS_REGEX + cls.UNIT_ID + r")"
        base = r"otrzymuje\s+brzmienie\b"
        loc = r"\s*(zdanie\s+\w+)?(\w+\s+zdanie)?(w\s+dziale\s+\w+)?\s*"
        return unit + loc + base

    @classmethod
    def acts(cls) -> str:
        return cls.ACT_REGEX

    @classmethod
    def acts_with_day(cls) -> str:
        return cls.ACT_REGEX + r"(?=" + cls.FROM_DAY + r")"

    @classmethod
    def acts_without_day(cls) -> str:
        return cls.ACT_REGEX + r"(?!" + cls.FROM_DAY + r")"

    @classmethod
    def acts_not_changed(cls) -> str:
        return r"(?<!\bo\szmianie\b)\s+" + cls.ACT_REGEX


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
    acts_not_changed: int = 0
    acts_with_day: int = 0
    acts_without_day: int = 0
    acts: int = 0
    additions: int = 0
    changes: int = 0
    removals: int = 0
    year: Optional[str] = None

    def add(self, stats: "BillStats"):
        self.acts += stats.acts
        self.acts_not_changed += stats.acts_not_changed
        self.acts_with_day += stats.acts_with_day
        self.acts_without_day += stats.acts_without_day
        self.additions += stats.additions
        self.changes += stats.changes
        self.removals += stats.removals


class BillProcessor:
    year_lookup = YearLookup()
    addition_counter = ReCounter(Patterns.addition())
    removal_counter = ReCounter(Patterns.removal())
    change_counter = ReCounter(Patterns.change())
    acts_counter = ReCounter(Patterns.acts())
    acts_not_changed_counter = ReCounter(Patterns.acts_not_changed())
    acts_with_day_counter = ReCounter(Patterns.acts_with_day())
    acts_without_day_counter = ReCounter(Patterns.acts_without_day())

    def __init__(
        self,
        text: str,
    ):
        self.text = text

    def year(self):
        return self.__class__.year_lookup(self.text)

    def stats(self) -> BillStats:
        cls = self.__class__
        return BillStats(
            year=self.year(),
            additions=self.addition_counter(self.text),
            removals=cls.removal_counter(self.text),
            changes=cls.change_counter(self.text),
            acts_not_changed=cls.acts_not_changed_counter(self.text),
            acts_with_day=cls.acts_with_day_counter(self.text),
            acts_without_day=cls.acts_without_day_counter(self.text),
            acts=cls.acts_counter(self.text),
        )


def process_file(filename: str) -> BillStats:
    with open(filename, "r") as f:
        return BillProcessor(f.read()).stats()


class CorpusProcessor:
    class CorpusStats:
        def __init__(self):
            self.stats = {}
            self.all = BillStats()

        def add(self, stats: BillStats):
            self.all.add(stats)
            year = stats.year
            if year and year not in self.stats:
                self.stats[year] = stats
                return
            self.stats[year].add(stats)

        def get_all(self) -> List[BillStats]:
            return {"by_year": list(self.stats.values()), "all": self.all}

        def plot(self):
            def plot_changes_by_type():
                df = pd.DataFrame(self.get_all()["by_year"])
                df = df.drop(
                    columns=[col for col in df.columns if col.startswith("acts")]
                )
                df = df.set_index("year")
                df = df.div(df.sum(axis=1), axis=0)
                df = df.reset_index()
                df.plot(x="year")
                plt.show(block=True)

            def plot_acts():
                all = self.get_all()["all"]
                labels = [
                    "Ustawy",
                    'Ustawy "z dnia"',
                    'Ustawy bez "z dnia"',
                    'Ustawy bez "o zmianie"',
                ]
                values = [
                    all.acts,
                    all.acts_with_day,
                    all.acts_without_day,
                    all.acts_not_changed,
                ]
                plt.bar(labels, values)
                plt.show(block=True)

            plot_changes_by_type()
            plot_acts()

    def __init__(self) -> None:
        self.corpus = Corpus()
        self.corpus_stats = self.__class__.CorpusStats()

    def process(self):
        with Pool(N_PROCESSES) as pool:
            stats = pool.map(process_file, self.corpus.list_all())
            for bill_stats in stats:
                self.corpus_stats.add(bill_stats)
            print(self.corpus_stats.get_all())
            self.corpus_stats.plot()


if __name__ == "__main__":
    CorpusProcessor().process()
