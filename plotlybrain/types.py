from typing import Literal

ScoreName = Literal[
    "rel_abundance",
    "frequency",
    "density",
]

RelAbundanceMethod = Literal[
    "within",
    "reference",
]

ReferenceMode = Literal[
    "pooled",
    "group",
]