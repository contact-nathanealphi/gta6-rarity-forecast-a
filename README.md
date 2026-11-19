# gta6-rarity-forecast-a

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-a/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-a/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Hierarchical logit-curve forecast of GTA VI trophy earn rates on Exophase at 2026-11-26T00:00:00Z. Per-trophy least-squares lines of logit rate on ln(1 + hours since release), shrunk toward tier means, plus an explicit model of Exophase's refresh-dilution: at a refresh the tracked-player count jumps and each rate is pulled toward the rate among the newly tracked players, who hold fewer trophies. Spreads combine a random-walk term, the line-fit residual, refresh uncertainty, and counting noise.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input/`) and writes one `/output/<id>.out` per case: five quantiles [0.05, 0.25, 0.5, 0.75, 0.95] of each trophy's earn rate at the case target. Python 3.13, standard library only, no network, deterministic.

---

Entered on [make.fun](https://make.fun/?ref=9ncgmbd6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
