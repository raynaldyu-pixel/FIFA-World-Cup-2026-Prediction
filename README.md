# ⚽ FIFA World Cup 2026 — Match Outcome Prediction

A machine learning pipeline that predicts the outcome (Home Win / Draw / Away Win) of every FIFA World Cup 2026 group stage match. Built around a custom team rating system, macroeconomic context features, official FIFA ranking data, and a heavily regularized XGBoost classifier tuned specifically on World Cup match patterns.

---

## 📊 Results

| Metric | Score |
|---|---|
| Validation Accuracy (raw) | 58.48% |
| Validation Accuracy (draw-tuned) | **58.55%** |
| Validation Log Loss | 0.8998 |
| Validation Set | All international matches 2022 – June 2026 |

---

## 🗂️ Datasets

| Dataset | Source | Purpose |
|---|---|---|
| International Football Results (1872–2026) | `martj42` on Kaggle | Match results, scores, tournament names |
| GDP & Population for International Football | `raynaldyu` on Kaggle | Macroeconomic context per team per year |
| FIFA World Ranking (1992–2024) | `cashncarry` on Kaggle | Official FIFA ranking points |
| Former Country Names | `martj42` on Kaggle | Maps dissolved nations to their modern equivalents |

---

## 🔧 Pipeline Overview

```
Raw Data
   │
   ├─ Merge results + GDP/Population + FIFA Rankings
   ├─ Standardise former country names (e.g. West Germany → Germany)
   ├─ Interpolate missing macroeconomic values by year per team
   ├─ Inject parent-state macroeconomic fallback (e.g. Andalusia → Spain)
   │
   ├─ Train / Val / Test Split
   │     Train : year ≤ 2021        (~149k matches)
   │     Val   : 2022 – 2026-06-10  (~4.4k matches, includes WC 2022)
   │     Test  : FIFA World Cup 2026 fixtures
   │
   ├─ PRISM Rating System  ──────────────────────────────────────────────┐
   │     Build chronological ratings on train set                        │
   │     Snapshot pre-match state for each row (zero leakage)            │
   │     Inject rating features into train / val / test                  │
   │     Rebuild ratings on all pre-2026 data for test inference         │
   └──────────────────────────────────────────────────────────────────────┘
   │
   ├─ Feature Engineering
   ├─ XGBoost Training  (multi:softprob, 3 classes)
   ├─ Draw Threshold Optimisation
   └─ WC 2026 Predictions
```

---

## 🏅 PRISM Rating System

> **P**erformance **R**ating with **I**mportance **S**caling and **M**omentum

PRISM is a custom rating system built specifically for this project. Inspired by Elo and the Berrar rating framework, it tracks three independent dimensions of team strength updated via Exponential Moving Average (EMA) after every historical match.

### Three Dimensions

| Dimension | Measures | Better When |
|---|---|---|
| **Attack** | Goals scored relative to opponent's defensive strength | Higher |
| **Defense** | Goals conceded relative to opponent's attacking strength | Lower |
| **Momentum** | Rolling win/draw/loss outcome confidence | Higher |

### Opponent-Adjusted Performance

Raw goals are normalised by the opponent's rating so that context is always preserved:

```
attack_performance  = goals_scored   / max(0.2, opponent_defense_rating)
defense_performance = goals_conceded / max(0.2, opponent_attack_rating)
```

Goals are capped at 6 to prevent extreme blowouts from distorting ratings.

### The I-Factor — Match Importance Scaling

Not all matches carry equal weight. PRISM scales the EMA learning rate by an official FIFA importance multiplier:

| Tournament | I-Factor |
|---|---|
| Friendly | 1.0 |
| UEFA Nations League | 1.5 |
| World Cup / Continental Qualifiers | 2.5 |
| African Cup of Nations / AFC Asian Cup | 3.5 |
| UEFA Euro / Copa América / CONCACAF Championship | 4.0 |
| **FIFA World Cup** | **5.0** |

```
match_alpha = min(base_alpha × I-Factor, 1.0)   [base_alpha = 0.05]
```

A World Cup match updates ratings 5× faster than a friendly, and that signal persists longer through the EMA.

### Zero Leakage Design

Before each match is processed, both teams' current ratings are snapshotted as features. The EMA update only fires *after* the snapshot — meaning the model only ever sees information that was available before kick-off.

### 4-Tier Fallback for Unknown Teams

| Tier | Condition | Action |
|---|---|---|
| 1 | Team has match history | Use own rating directly |
| 2 | Sub-national / regional team | Inherit sovereign parent's rating (e.g. Andalusia → Spain) |
| 3 | No parent exists | Use average rating of all teams from the same tournament pool |
| 4 | No regional data | Fall back to global mean across all rated teams |

---

## 🛠️ Feature Engineering

| Feature | Description |
|---|---|
| `match_importance` | I-Factor of the tournament |
| `home_advantage` | 1 if home ground, 0 if neutral venue |
| `gdp_ratio` | Home team GDP per capita / Away team GDP per capita |
| `pop_diff` | Home team population − Away team population |
| `matches_ratio` | Relative historical experience (home / away) |
| `experience_advantage` | Absolute match count difference |
| `attack_vs_defense_diff` | Home attack rating − Away defense rating |
| `defense_vs_attack_diff` | Home defense rating − Away attack rating |
| `overall_momentum_diff` | Home outcome rating − Away outcome rating |
| `home_fifa_rank` | Official FIFA ranking at match date |
| `away_fifa_rank` | Official FIFA ranking at match date |
| `fifa_rank_diff` | Away rank − Home rank (positive = home better ranked) |
| `fifa_points_diff` | Home FIFA points − Away FIFA points |

All PRISM rating dimensions (attack, defense, outcome, matches played) for both home and away teams are also included as direct features.

---

## 🤖 Model

**XGBoost Classifier** with multi-class softmax probability output.

| Hyperparameter | Value | Rationale |
|---|---|---|
| `objective` | `multi:softprob` | Outputs calibrated 3-class probabilities |
| `learning_rate` | `0.01` | Slower, more precise convergence |
| `n_estimators` | `2000` | Large capacity with early stopping |
| `max_depth` | `3` | Shallow trees to prevent memorising upsets |
| `min_child_weight` | `15` | Avoids splits on small noisy subgroups |
| `reg_lambda` | `5.0` | Heavy L2 regularisation |
| `early_stopping_rounds` | `50` | Stops when validation log loss stops improving |

**Target classes:** `2 = Home Win`, `1 = Draw`, `0 = Away Win`

---

## 🎯 Draw Threshold Optimisation

Standard argmax prediction almost never selects Draw because home/away probabilities tend to slightly outcompete draw probability even in balanced matches. A post-processing threshold addresses this.

### Method

Instead of a flat draw probability floor, a **gap-based threshold** is used:

```python
gap = max(prob_home, prob_away) - prob_draw

if prob_draw >= MIN_DRAW_PROB and gap <= GAP_THRESHOLD:
    predict Draw
else:
    predict argmax(home, away)
```

This requires draw to have *both* reasonable absolute probability and a genuinely close race with the leader — preventing weak draws from firing just because all three probabilities are low.

---

## 📁 Project Structure

```
├── datasets/
│   ├── results.csv              # International match results 1872–2026
│   ├── gdpandpop_data.csv       # GDP & population per team per year
│   ├── fifa_ranking-2024.csv    # Official FIFA rankings
│   └── former_names.csv         # Country name mappings
│
├── fifa-world-cup-2026-prediction.ipynb   # Main notebook
└── README.md
```

---

## 🔍 Key Design Decisions

**Why EMA instead of a fixed window?** An EMA naturally down-weights older matches without a hard cutoff. A team that won the World Cup 3 years ago still carries that signal, but it fades proportionally as they play more matches.

**Why opponent-adjusted ratings?** Scoring 4 goals against a top-ranked defense is fundamentally different from scoring 4 against a minnow. Dividing by the opponent's rating forces the system to distinguish quality of performance rather than raw volume.

**Why train on ≤ 2021 data?** The last World Cup (2022) serves as a realistic out-of-sample validation environment — same competition format, same tier of teams, same stakes. Training up to 2021 means WC 2022 is genuinely unseen during training.
