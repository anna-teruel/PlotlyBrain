from typing import Literal

type ScoreName = Literal[
	"rel_abundance",
	"frequency",
	"density",
]

type RelAbundanceMethod = Literal[
	"within",
	"reference",
]

type ReferenceMode = Literal[
	"pooled",
	"group",
]
