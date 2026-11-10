# Player Action Value and Possession Threat

Who moved the ball into more threatening positions at the 2018 men's FIFA World Cup? This case study values completed moves using expected threat (xT) and compares player, team, and action profiles across **128,484 actions in 64 matches**.

[Open the notebook](notebooks/player_action_value_explorer.ipynb) · [View the player leaderboard](assets/world_cup_xt_per90.png)

## Findings

- **Neymar led total completed-move xT** at 1.799. Joshua Kimmich led the rate leaderboard at **0.492 xT per 90** among players with complete appearance coverage and at least 270 minutes.
- **Passes contributed the most total xT.** Crosses added more xT per attempt, approximately 0.0197 versus 0.0008 for passes. Attempt frequency and starting location affect both measures.
- **Germany led team xT per match at 1.906** over three matches. The small sample and differences in opponent and game state limit team comparisons.
- **The three illustrative style groups mostly separated by on-ball activity and output.** The three-cluster silhouette score was 0.338. Two and four clusters scored slightly higher, so the groups are exploratory.

## Method

Convert event data to SPADL actions and orient both teams to attack in the same direction before fitting a 16 × 12 xT grid. Score completed possession-preserving moves using the signed difference between destination and origin. Unrated actions contribute zero. The analysis covers progression on the ball and omits shooting value, turnovers, defending, and off-ball movement. The same tournament is used for fitting and description.

## Reproduce

The notebook reads the tracked action and appearance CSVs. It refits xT from their coordinates and recomputes the summaries.

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Open `notebooks/player_action_value_explorer.ipynb` and run all cells. The older parquet summaries and figures are earlier pipeline outputs with a different scoring convention.

Data provided by [StatsBomb Open Data](https://github.com/hudl/open-data). Expected threat implementation from [socceraction](https://socceraction.readthedocs.io/en/latest/documentation/valuing_actions/xT.html).

![StatsBomb logo](assets/statsbomb.svg)
