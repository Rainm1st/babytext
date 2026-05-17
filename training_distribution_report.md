# BabyLM Training Source Statistics

| Source | Description | Lines | Words | Share | Size MB |
|---|---|---:|---:|---:|---:|
| bnc_spoken | BNC spoken dialogue | 803,341 | 7,620,671 | 3.95% | 37.70 |
| childes | CHILDES child-directed speech | 5,638,783 | 28,410,878 | 14.74% | 145.24 |
| gutenberg | Project Gutenberg children's books | 661,928 | 25,576,896 | 13.27% | 134.58 |
| open_subtitles | OpenSubtitles scripted dialogue | 3,333,618 | 19,205,138 | 9.96% | 97.36 |
| simple_wiki | Simple English Wikipedia | 642,735 | 15,314,317 | 7.95% | 83.87 |
| switchboard | Switchboard telephone dialogue | 30,559 | 248,491 | 0.13% | 1.16 |
| bnc_spoken | BNC spoken dialogue | 803,341 | 7,620,671 | 3.95% | 37.70 |
| childes | CHILDES child-directed speech | 5,638,783 | 28,410,878 | 14.74% | 145.24 |
| gutenberg | Project Gutenberg children's books | 661,928 | 25,576,896 | 13.27% | 134.58 |
| open_subtitles | OpenSubtitles scripted dialogue | 3,333,618 | 19,205,138 | 9.96% | 97.36 |
| simple_wiki | Simple English Wikipedia | 642,735 | 15,314,317 | 7.95% | 83.87 |
| switchboard | Switchboard telephone dialogue | 30,559 | 248,491 | 0.13% | 1.16 |
| **Total** |  | **22,221,928** | **192,752,782** | **100.00%** |  |

# Examples

## bnc_spoken
1. Well it's just that, you know, a pound, or a hundred pounds today, is not the same as a hundred pounds in a year's time, or two, two years' time.
2. Right.

## childes
1. *COL:	yes your lunch.
2. *CHI:	more.

## gutenberg
1. "My word!" said Bertha.  "I like it.  On’y you tell Bill to cut me stove-wood."  She marched into the kitchen with high enterprise in her heart.
2. Aileen was still asleep when they reached the farm.  Tom peeped into her room, and saw Garth sitting by her bedside, very upright and alert.  He shook his head frantically at his father, and Tom took the hint, and withdr

## open_subtitles
1. BUT, JUST AS EXCITING,
2. WE HAVE THE CO-HOSTS OF WAKE UP, SAN FRANCISCO.

## simple_wiki
1. = = = Bullet the Blue Sky = = =
2. "Bullet the Blue Sky" is a 1987 song by Irish rock group U2. It is the fourth track from their fifth studio album "The Joshua Tree". It is inspired by Bono trips to El Salvador and Nicaragua ravaged by United States lead

## switchboard
1. B:	Yeah,
2. B:	I'm in Texas.

## bnc_spoken
1. Well it's just that, you know, a pound, or a hundred pounds today, is not the same as a hundred pounds in a year's time, or two, two years' time.
2. Right.

## childes
1. *COL:	yes your lunch.
2. *CHI:	more.

## gutenberg
1. "My word!" said Bertha.  "I like it.  On’y you tell Bill to cut me stove-wood."  She marched into the kitchen with high enterprise in her heart.
2. Aileen was still asleep when they reached the farm.  Tom peeped into her room, and saw Garth sitting by her bedside, very upright and alert.  He shook his head frantically at his father, and Tom took the hint, and withdr

## open_subtitles
1. BUT, JUST AS EXCITING,
2. WE HAVE THE CO-HOSTS OF WAKE UP, SAN FRANCISCO.

## simple_wiki
1. = = = Bullet the Blue Sky = = =
2. "Bullet the Blue Sky" is a 1987 song by Irish rock group U2. It is the fourth track from their fifth studio album "The Joshua Tree". It is inspired by Bono trips to El Salvador and Nicaragua ravaged by United States lead

## switchboard
1. B:	Yeah,
2. B:	I'm in Texas.

# Evaluation-Suite Mapping

| Test suite | Mapped training sources | Mapped words | Share of training words | Rationale |
|---|---|---:|---:|---|
| BLiMP | childes, gutenberg, simple_wiki, bnc_spoken, open_subtitles, switchboard | 96,376,391 | 50.00% | grammar, syntax, agreement, binding, islands; broad language exposure |
| BLiMP Supplement | childes, bnc_spoken, switchboard, open_subtitles, simple_wiki, gutenberg | 96,376,391 | 50.00% | QA congruence, turn-taking, hypernymy, subject-aux inversion |
| EWoK | simple_wiki, gutenberg, childes, open_subtitles | 88,507,229 | 45.92% | world knowledge, relations, events, social/physical properties |
| COMPS | gutenberg, simple_wiki, childes | 69,302,091 | 35.95% | compositional generalization and structured sentence semantics |
| Entity Tracking | gutenberg, open_subtitles, childes, simple_wiki | 88,507,229 | 45.92% | narrative/dialogue contexts with entities and references |
| Reading Eye / SPR | childes, bnc_spoken, open_subtitles, switchboard, gutenberg | 81,062,074 | 42.05% | human-like processing difficulty, dialogue, naturalistic text |
| MNLI/RTE/WSC/MRPC/QQP/MultiRC/BoolQ | simple_wiki, gutenberg, open_subtitles, bnc_spoken, childes | 96,127,900 | 49.87% | downstream NLU: entailment, paraphrase, QA, reading comprehension |

CSV written to: results/training_source_stats.csv
