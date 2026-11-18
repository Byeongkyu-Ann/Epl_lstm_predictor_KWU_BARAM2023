# EPL Match Outcome Prediction with a Two-Stage LSTM

**English** | [한국어](README.ko.md)

A PyTorch project that predicts the result of an English Premier League match from the two teams alone.
It was built in the Robotics Club BARAM at Kwangwoon University from September to November 2023.

## Overview

Only the teams are known before kickoff. The in-match statistics that explain a result are not.
This project closes the gap with two LSTM stages.

**Stage 1** predicts the home share of five match statistics from the two teams.
**Stage 2** predicts win, draw or loss from the two teams and the Stage 1 statistics.

```
home team, away team
        |
        v
  Stage 1 (5 LSTMs)  ->  possession, shots, touches, passes, clearances (home share)
        |
        v
  Stage 2 (1 LSTM)   ->  home win / draw / loss
```

The data comes from premierleague.com and is collected with Selenium.

## Repository structure

```
eplpredict/
  config.py          paths, team index, column names
  crawl.py           Selenium crawler for one season
  preprocess.py      raw matches -> features
  model.py           LSTM model and checkpoint helpers
  data.py            chronological split and sequence windows
  training.py        full-batch training loop
  train_stage1.py    trains the five statistic models
  train_stage2.py    trains the result model
  predict.py         two-stage inference and command line tool
  app.py             PyQt5 desktop demo
experiments/
  wdl_from_match_stats.py   result from true match statistics
assets/              Qt Designer file for the demo
data/                CSV files. Not versioned.
models/              checkpoints. Not versioned.
```

## Quick start

```bash
pip install -r requirements.txt
```

**1. Crawl.** Run it once per season. The season id is the `se` value in the results page URL.
Chrome is required.

```bash
python -m eplpredict.crawl --season-id 489 --out data/raw_2022_23.csv
```

**2. Preprocess.** List the files from the oldest to the newest season.

```bash
python -m eplpredict.preprocess --inputs data/raw_2021_22.csv data/raw_2022_23.csv
```

**3. Train.**

```bash
python -m eplpredict.train_stage1
python -m eplpredict.train_stage2 --plot
```

**4. Predict.**

```bash
python -m eplpredict.predict "Man Utd" "Man City"
python -m eplpredict.app        # desktop demo, needs assets/first2.ui
```

## Data and features

The crawler stores one row per match with these fields.

| Field | Meaning |
|---|---|
| `home_team`, `away_team` | team names |
| `goal_diff` | home goals minus away goals |
| `home_` and `away_` + statistic | possession, shots, shots on target, touches, passes, tackles, clearances |

The preprocessing step adds these columns.

| Column | Definition |
|---|---|
| `home_idx`, `away_idx` | team index from 1 to 35. A higher index means a stronger team. See `config.py` |
| `possession_share`, `shot_share`, `touch_share`, `pass_share`, `clearance_share` | home / (home + away) |
| `result` | 0 = home loss, 1 = draw, 2 = home win |

Matches are never shuffled. The first 90% of the rows train the models and the last 10% test them.

## Models

| | Stage 1 | Stage 2 |
|---|---|---|
| Count | 5 | 1 |
| Input | `home_idx`, `away_idx` | team indices and the 5 Stage 1 shares |
| Output | home share in [0, 1] | 3 classes |
| Loss | MSE | cross-entropy |
| LSTM size | 4 layers, 10 hidden units | 6 layers, 30 hidden units |
| Optimizer | Adam, learning rate 0.001 | Adam, learning rate 0.001 |
| Metric | R² | accuracy |

Stage 2 is trained on the true statistics of each match. At prediction time it receives the Stage 1 outputs.

## Results

Each training script prints its metrics at the end. Stage 1 reports R² per statistic. Stage 2 reports accuracy and the accuracy of always predicting the most frequent class.

`experiments/wdl_from_match_stats.py` gives a reference accuracy when the true match statistics are known.

Example output of the original demo for Manchester United at home against Manchester City:

| Statistic | Home share | Away share |
|---|---|---|
| Possession | 39.4% | 60.6% |
| Shots | 42.2% | 57.8% |
| Touches | 38.5% | 61.5% |
| Passes | 35.3% | 64.7% |
| Clearances | 55.9% | 44.1% |

Predicted result: home loss. The real match on 29 October 2023 ended 0-3.
This is a single match and not a performance measure.

## Limitations

- The team is described by one fixed index. Recent form, lineups and injuries are not used.
- Stage 2 is trained on true statistics and tested on true statistics. The end-to-end accuracy of the two stages is not measured by the training scripts.
- The crawler relies on XPaths of the site layout in autumn 2023. It is untested on the current site layout and may need updates.

## Project information

- Author: Byeongkyu Ann, Robotics Club BARAM, Kwangwoon University
- Period: September to November 2023
- Stack: Python, PyTorch, Selenium, pandas, scikit-learn, PyQt5
