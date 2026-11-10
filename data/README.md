# Data

[StatsBomb Open Data](https://github.com/hudl/open-data) provides event and lineup records for the 2018 men's FIFA World Cup, competition 43 and season 3.

The main notebook reads the match-context SPADL action coordinates in `processed/actions_xt.csv` and player minutes from `processed/player_appearances.csv`. It rotates away-team coordinates, fits a new xT surface, and computes player and team summaries. The preexisting xT columns in the action CSV are historical. The notebook ignores those columns.

The other processed tables and raw cache belong to an earlier pipeline. Reproduce the current published figures from the notebook.
