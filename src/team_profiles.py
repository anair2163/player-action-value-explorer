"""Team-level style profile helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_team_profiles(actions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate action-value data into a team style profile table."""
    if actions is None or actions.empty:
        return pd.DataFrame(
            columns=[
                "team_name",
                "total_xt",
                "progression_xt",
                "final_third_threat",
                "xt_per_match",
                "progressive_actions",
                "matches_played",
                "action_count",
            ]
        )

    xt_column = "xt_delta" if "xt_delta" in actions.columns else "xt_added"
    match_column = "match_id" if "match_id" in actions.columns else "game_id"
    final_third_column = "final_third_threat" if "final_third_threat" in actions.columns else None
    progression_column = "progression_xt" if "progression_xt" in actions.columns else None
    progressive_flag_column = "is_progressive_action" if "is_progressive_action" in actions.columns else None

    if xt_column is None or match_column is None or progression_column is None or final_third_column is None:
        raise ValueError(
            "Actions DataFrame must include `xt_delta` or `xt_added`, `match_id` or `game_id`, "
            "`progression_xt`, and `final_third_threat`."
        )

    grouped = (
        actions.groupby("team_name", dropna=False)
        .agg(
            total_xt=(xt_column, "sum"),
            progression_xt=(progression_column, "sum"),
            final_third_threat=(final_third_column, "sum"),
            action_count=(xt_column, "size"),
            matches_played=(match_column, "nunique"),
            progressive_actions=(progressive_flag_column, "sum") if progressive_flag_column else (xt_column, lambda s: 0),
        )
        .reset_index()
    )
    grouped["xt_per_match"] = np.where(
        grouped["matches_played"].fillna(0) > 0,
        grouped["total_xt"] / grouped["matches_played"].fillna(0),
        0.0,
    )
    grouped["xt_total"] = grouped["total_xt"]
    grouped["actions"] = grouped["action_count"]
    grouped["matches"] = grouped["matches_played"]
    return grouped.sort_values("total_xt", ascending=False).reset_index(drop=True)
