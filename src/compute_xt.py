"""Compute expected threat (xT) from SPADL actions.

xT is a simple possession-value model. It assigns a value to pitch locations
and scores actions by how much they move the ball into more dangerous areas.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from socceraction.spadl import config as spadlconfig
from socceraction.xthreat import ExpectedThreat


REQUIRED_ACTION_COLUMNS = {
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "type_name",
    "result_name",
}
PROGRESSIVE_ACTION_TYPES = {"pass", "dribble"}


def _validate_actions(actions_df: pd.DataFrame, required_columns: set[str] | None = None) -> None:
    """Raise a clear error when required SPADL columns are missing."""
    if actions_df is None or actions_df.empty:
        return

    columns = required_columns or REQUIRED_ACTION_COLUMNS
    missing = sorted(columns.difference(actions_df.columns))
    if missing:
        raise ValueError(
            "Actions DataFrame is missing required columns for xT calculations: "
            + ", ".join(missing)
        )


def _empty_summary(columns: list[str]) -> pd.DataFrame:
    """Return an empty summary frame with the requested columns."""
    return pd.DataFrame(columns=columns)


def _resolve_xt_column(df: pd.DataFrame) -> str:
    """Return the xT value column used for aggregation."""
    if "xt_delta" in df.columns:
        return "xt_delta"
    if "xt_added" in df.columns:
        return "xt_added"
    raise ValueError("Actions DataFrame must contain `xt_delta` or `xt_added`.")


def _final_third_boundary() -> float:
    """Return the x-coordinate where the final third begins."""
    return spadlconfig.field_length * (2 / 3)


def _add_action_flags(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Add progression and final-third flags used in summaries."""
    actions = actions_df.copy()
    xt_column = _resolve_xt_column(actions)
    final_third_boundary = _final_third_boundary()

    progressive_mask = (
        actions["type_name"].isin(PROGRESSIVE_ACTION_TYPES)
        & (pd.to_numeric(actions["end_x"], errors="coerce") > pd.to_numeric(actions["start_x"], errors="coerce"))
        & (pd.to_numeric(actions["start_x"], errors="coerce") < final_third_boundary)
        & (
            (pd.to_numeric(actions["end_x"], errors="coerce") - pd.to_numeric(actions["start_x"], errors="coerce") >= 10)
            | (pd.to_numeric(actions["end_x"], errors="coerce") >= final_third_boundary)
        )
    )
    final_third_mask = (
        (pd.to_numeric(actions["start_x"], errors="coerce") >= final_third_boundary)
        | (pd.to_numeric(actions["end_x"], errors="coerce") >= final_third_boundary)
    )

    actions["is_progressive_action"] = progressive_mask.fillna(False)
    actions["is_final_third_action"] = final_third_mask.fillna(False)
    actions["progression_xt"] = actions[xt_column].where(actions["is_progressive_action"], 0.0)
    actions["final_third_threat"] = actions[xt_column].where(actions["is_final_third_action"], 0.0)
    return actions


def _model_is_fitted(model: ExpectedThreat) -> bool:
    """Return True when the xT model has a learned grid."""
    return hasattr(model, "xT") and np.any(model.xT)


def _cell_indices(series: pd.Series, max_value: float, bins: int) -> pd.Series:
    """Map pitch coordinates to xT grid cells."""
    numeric = pd.to_numeric(series, errors="coerce")
    clipped = numeric.clip(lower=0, upper=max_value - 1e-9)
    scaled = pd.Series(np.floor(clipped / max_value * bins), index=series.index)
    scaled = scaled.where(numeric.notna(), pd.NA).astype("Int64")
    return scaled.clip(lower=0, upper=bins - 1)


def _grid_values(actions_df: pd.DataFrame, model: ExpectedThreat, prefix: str) -> pd.Series:
    """Look up xT grid values at action start or end locations."""
    x_col = f"{prefix}_x"
    y_col = f"{prefix}_y"
    x_idx = _cell_indices(actions_df[x_col], spadlconfig.field_length, model.l)
    y_idx = _cell_indices(actions_df[y_col], spadlconfig.field_width, model.w)

    values = np.full(len(actions_df), np.nan, dtype="float64")
    valid = x_idx.notna() & y_idx.notna()
    if valid.any():
        values[valid] = model.xT[(model.w - 1 - y_idx[valid]).astype(int), x_idx[valid].astype(int)]
    return pd.Series(values, index=actions_df.index)


def _resolve_group_columns(
    df: pd.DataFrame,
    preferred_columns: list[str],
    fallback_columns: list[str] | None = None,
) -> list[str]:
    """Return the group columns that exist in the DataFrame."""
    available = [column for column in preferred_columns if column in df.columns]
    if available:
        return available
    if fallback_columns is not None:
        return [column for column in fallback_columns if column in df.columns]
    return []


def initialize_xt_model(actions_df: pd.DataFrame | None = None, l: int = 16, w: int = 12) -> ExpectedThreat:
    """Create an xT model and fit it when SPADL actions are provided."""
    model = ExpectedThreat(l=l, w=w)
    if actions_df is None or actions_df.empty:
        return model

    _validate_actions(actions_df)
    model.fit(actions_df)
    return model


def score_actions_with_xt(actions_df: pd.DataFrame, model: ExpectedThreat | None = None) -> pd.DataFrame:
    """Score SPADL actions with xT values.

    The result includes the xT value at the action start, the xT value at the
    action end, and the action score itself.
    """
    if actions_df is None or actions_df.empty:
        return pd.DataFrame()

    _validate_actions(actions_df)

    xt_model = model or initialize_xt_model(actions_df)
    if not _model_is_fitted(xt_model):
        xt_model.fit(actions_df)

    scored = actions_df.copy()
    scored["xt_start"] = _grid_values(scored, xt_model, prefix="start")
    scored["xt_end"] = _grid_values(scored, xt_model, prefix="end")
    scored["xt_value"] = np.nan_to_num(xt_model.rate(scored), nan=0.0)
    scored = add_xt_delta(scored)
    return _add_action_flags(scored)


def add_xt_delta(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Add xT change columns to an actions table.

    xT delta is the change in threat from the start of an action to the end.
    """
    if actions_df is None or actions_df.empty:
        return pd.DataFrame()

    actions = actions_df.copy()
    if {"xt_start", "xt_end"}.issubset(actions.columns):
        actions["xt_delta"] = actions["xt_end"] - actions["xt_start"]
    elif "xt_value" in actions.columns:
        actions["xt_delta"] = actions["xt_value"]
    else:
        raise ValueError("Cannot compute xT delta without `xt_start`/`xt_end` or `xt_value` columns.")

    if "xt_value" not in actions.columns:
        actions["xt_value"] = actions["xt_delta"]
    actions["xt_added"] = actions["xt_delta"]

    actions["xt_positive"] = actions["xt_delta"].clip(lower=0)
    actions["xt_negative"] = actions["xt_delta"].clip(upper=0)
    return actions


def aggregate_player_xt(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate xT outputs to the player level."""
    if actions_df is None or actions_df.empty:
        return _empty_summary(
            [
                "player_id",
                "player_name",
                "team_id",
                "team_name",
                "action_count",
                "total_xt",
                "avg_xt",
                "positive_xt",
                "negative_xt",
            ]
        )

    required = {"xt_delta"}
    _validate_actions(actions_df, required_columns=REQUIRED_ACTION_COLUMNS.union(required))
    actions_df = _add_action_flags(actions_df)
    group_columns = _resolve_group_columns(
        actions_df,
        ["player_id", "player_name", "team_id", "team_name"],
        fallback_columns=["player_name", "team_name"],
    )
    if not group_columns:
        raise ValueError("Cannot aggregate player xT without player columns.")

    summary = (
        actions_df.groupby(group_columns, dropna=False)
        .agg(
            action_count=("xt_delta", "size"),
            total_xt=("xt_delta", "sum"),
            avg_xt=("xt_delta", "mean"),
            positive_xt=("xt_positive", "sum") if "xt_positive" in actions_df.columns else ("xt_delta", lambda s: s.clip(lower=0).sum()),
            negative_xt=("xt_negative", "sum") if "xt_negative" in actions_df.columns else ("xt_delta", lambda s: s.clip(upper=0).sum()),
        )
        .reset_index()
        .sort_values("total_xt", ascending=False)
        .reset_index(drop=True)
    )
    if "match_id" in actions_df.columns:
        matches_played = (
            actions_df.groupby(group_columns, dropna=False)["match_id"]
            .nunique()
            .reset_index(name="matches_played")
        )
        summary = summary.merge(matches_played, on=group_columns, how="left")

    summary["actions"] = summary["action_count"]
    summary["xt_total"] = summary["total_xt"]
    summary["xt_mean"] = summary["avg_xt"]
    summary["xt_positive"] = summary["positive_xt"]
    summary["xt_negative"] = summary["negative_xt"]
    return summary


def aggregate_team_xt(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate xT outputs to the team level."""
    if actions_df is None or actions_df.empty:
        return _empty_summary(
            [
                "team_id",
                "team_name",
                "matches_played",
                "action_count",
                "total_xt",
                "progression_xt",
                "final_third_threat",
                "xt_per_match",
                "progressive_actions",
            ]
        )

    required = {"xt_delta"}
    _validate_actions(actions_df, required_columns=REQUIRED_ACTION_COLUMNS.union(required))
    actions_df = _add_action_flags(actions_df)
    group_columns = _resolve_group_columns(
        actions_df,
        ["team_id", "team_name"],
        fallback_columns=["team_name"],
    )
    if not group_columns:
        raise ValueError("Cannot aggregate team xT without team columns.")

    match_column = "match_id" if "match_id" in actions_df.columns else "game_id" if "game_id" in actions_df.columns else None

    aggregations: dict[str, Any] = {
        "action_count": ("xt_delta", "size"),
        "total_xt": ("xt_delta", "sum"),
        "avg_xt": ("xt_delta", "mean"),
        "positive_xt": ("xt_positive", "sum") if "xt_positive" in actions_df.columns else ("xt_delta", lambda s: s.clip(lower=0).sum()),
        "negative_xt": ("xt_negative", "sum") if "xt_negative" in actions_df.columns else ("xt_delta", lambda s: s.clip(upper=0).sum()),
        "progression_xt": ("progression_xt", "sum"),
        "final_third_threat": ("final_third_threat", "sum"),
        "progressive_actions": ("is_progressive_action", "sum"),
    }
    if match_column is not None:
        aggregations["matches_played"] = (match_column, "nunique")

    summary = (
        actions_df.groupby(group_columns, dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("total_xt", ascending=False)
        .reset_index(drop=True)
    )
    if "matches_played" not in summary.columns:
        summary["matches_played"] = np.nan
    summary["xt_per_match"] = np.where(
        summary["matches_played"].fillna(0) > 0,
        summary["total_xt"] / summary["matches_played"].fillna(0),
        0.0,
    )
    summary["actions"] = summary["action_count"]
    summary["matches"] = summary["matches_played"]
    summary["xt_total"] = summary["total_xt"]
    summary["xt_mean"] = summary["avg_xt"]
    summary["xt_positive"] = summary["positive_xt"]
    summary["xt_negative"] = summary["negative_xt"]
    return summary


def add_per90_metrics(summary_df: pd.DataFrame, minutes_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add per-90 metrics to a player or team summary.

    If a minutes table is provided, it is merged first. Otherwise the function
    uses an existing `minutes_played` column when present.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    summary = summary_df.copy()

    if minutes_df is not None and not minutes_df.empty:
        key_options = [
            ["player_id", "team_name"],
            ["player_id", "team_id"],
            ["player_id"],
            ["team_id"],
            ["player_name", "team_name"],
            ["player_name"],
            ["team_name"],
        ]
        merge_keys = next(
            (keys for keys in key_options if set(keys).issubset(summary.columns) and set(keys).issubset(minutes_df.columns)),
            None,
        )
        if merge_keys is None:
            raise ValueError("Could not find shared keys between summary_df and minutes_df.")

        minute_columns = [column for column in ["minutes_played", "minutes"] if column in minutes_df.columns]
        if not minute_columns:
            raise ValueError("minutes_df must contain `minutes_played` or `minutes`.")

        minute_column = minute_columns[0]
        minutes_lookup = minutes_df.loc[:, merge_keys + [minute_column]].copy()
        minutes_lookup = minutes_lookup.rename(columns={minute_column: "minutes_played"})
        minutes_lookup = (
            minutes_lookup.groupby(merge_keys, dropna=False)["minutes_played"]
            .sum()
            .reset_index()
        )
        summary = summary.merge(minutes_lookup, on=merge_keys, how="left")

    if "minutes_played" not in summary.columns:
        raise ValueError("Per-90 metrics require a `minutes_played` column or a valid minutes_df.")

    minutes = summary["minutes_played"].fillna(0)
    xt_total_column = "total_xt" if "total_xt" in summary.columns else "xt_total"
    action_count_column = "action_count" if "action_count" in summary.columns else "actions" if "actions" in summary.columns else None
    summary["xt_per90"] = np.where(minutes > 0, summary[xt_total_column] / minutes * 90, 0.0)
    summary["xt_per_90"] = summary["xt_per90"]
    if action_count_column is not None:
        summary["action_count_per90"] = np.where(minutes > 0, summary[action_count_column] / minutes * 90, 0.0)
        summary["actions_per_90"] = summary["action_count_per90"]
    return summary


def get_action_type_xt_summary(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize xT by action type for reporting and charting."""
    if actions_df is None or actions_df.empty:
        return _empty_summary(
            ["type_name", "actions", "xt_total", "xt_mean", "xt_positive", "xt_negative"]
        )

    required = {"xt_delta"}
    _validate_actions(actions_df, required_columns=REQUIRED_ACTION_COLUMNS.union(required))

    summary = (
        actions_df.groupby("type_name", dropna=False)
        .agg(
            actions=("xt_delta", "size"),
            xt_total=("xt_delta", "sum"),
            xt_mean=("xt_delta", "mean"),
            xt_positive=("xt_positive", "sum") if "xt_positive" in actions_df.columns else ("xt_delta", lambda s: s.clip(lower=0).sum()),
            xt_negative=("xt_negative", "sum") if "xt_negative" in actions_df.columns else ("xt_delta", lambda s: s.clip(upper=0).sum()),
        )
        .reset_index()
        .sort_values("xt_total", ascending=False)
        .reset_index(drop=True)
    )
    return summary


if __name__ == "__main__":
    from src.build_actions import attach_action_metadata, convert_match_events_to_actions
    from src.load_data import get_competitions, get_match_events, get_matches

    competitions = get_competitions()
    sample_competition = competitions.iloc[0]
    matches = get_matches(
        competition_id=int(sample_competition["competition_id"]),
        season_id=int(sample_competition["season_id"]),
    )
    sample_match_id = int(matches.iloc[0]["match_id"])

    events = get_match_events(sample_match_id)
    actions = convert_match_events_to_actions(events)
    actions = attach_action_metadata(actions, events, match_id=sample_match_id)
    scored_actions = score_actions_with_xt(actions)

    player_summary = aggregate_player_xt(scored_actions)
    team_summary = aggregate_team_xt(scored_actions)
    action_summary = get_action_type_xt_summary(scored_actions)

    print(f"Sample match: {sample_match_id}")
    print(scored_actions[["type_name", "result_name", "xt_start", "xt_end", "xt_delta"]].head(10).to_string(index=False))
    print("\nTop players by xT:")
    print(player_summary.head(5).to_string(index=False))
    print("\nTeam xT summary:")
    print(team_summary.head(5).to_string(index=False))
    print("\nAction type xT summary:")
    print(action_summary.head(5).to_string(index=False))
