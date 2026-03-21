"""Export processed football analytics tables for Quarto pages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_ACTIONS_PATH = DEFAULT_OUTPUT_DIR / "actions.parquet"
DEFAULT_PLAYER_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "player_summary.parquet"
DEFAULT_TEAM_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "team_summary.parquet"
DEFAULT_ACTION_TYPE_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "action_type_summary.parquet"


def ensure_output_dir(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Create the processed output directory if it does not already exist."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """Raise a clear error for missing or invalid export inputs."""
    if df is None:
        raise ValueError(f"`{name}` cannot be None.")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`{name}` must be a pandas DataFrame.")


def _normalize_actions_table(actions_df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent action-level naming before export."""
    actions = actions_df.copy()
    if "xt_delta" not in actions.columns and "xt_added" in actions.columns:
        actions["xt_delta"] = actions["xt_added"]
    if "xt_added" not in actions.columns and "xt_delta" in actions.columns:
        actions["xt_added"] = actions["xt_delta"]
    if "action_type" not in actions.columns and "type_name" in actions.columns:
        actions["action_type"] = actions["type_name"]
    if "xT_start" not in actions.columns and "xt_start" in actions.columns:
        actions["xT_start"] = actions["xt_start"]
    if "xT_end" not in actions.columns and "xt_end" in actions.columns:
        actions["xT_end"] = actions["xt_end"]
    return actions


def _normalize_player_summary(player_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent player-summary naming before export."""
    summary = player_summary_df.copy()
    if "total_xt" not in summary.columns and "xt_total" in summary.columns:
        summary["total_xt"] = summary["xt_total"]
    if "action_count" not in summary.columns and "actions" in summary.columns:
        summary["action_count"] = summary["actions"]
    if "xt_per90" not in summary.columns and "xt_per_90" in summary.columns:
        summary["xt_per90"] = summary["xt_per_90"]
    if "xt_mean" not in summary.columns and {"total_xt", "action_count"}.issubset(summary.columns):
        summary["xt_mean"] = summary["total_xt"] / summary["action_count"].clip(lower=1)
    summary["xt_total"] = summary["total_xt"]
    summary["actions"] = summary["action_count"]
    if "xt_per90" in summary.columns:
        summary["xt_per_90"] = summary["xt_per90"]
    return summary


def _normalize_team_summary(team_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent team-summary naming before export."""
    summary = team_summary_df.copy()
    if "total_xt" not in summary.columns and "xt_total" in summary.columns:
        summary["total_xt"] = summary["xt_total"]
    if "matches_played" not in summary.columns and "matches" in summary.columns:
        summary["matches_played"] = summary["matches"]
    if "action_count" not in summary.columns and "actions" in summary.columns:
        summary["action_count"] = summary["actions"]
    if "xt_per_match" not in summary.columns and {"total_xt", "matches_played"}.issubset(summary.columns):
        summary["xt_per_match"] = (
            summary["total_xt"] / summary["matches_played"].replace(0, pd.NA)
        ).fillna(0.0)
    if "xt_mean" not in summary.columns and {"total_xt", "action_count"}.issubset(summary.columns):
        summary["xt_mean"] = summary["total_xt"] / summary["action_count"].clip(lower=1)
    summary["xt_total"] = summary["total_xt"]
    summary["matches"] = summary["matches_played"]
    summary["actions"] = summary["action_count"]
    return summary


def _normalize_action_type_summary(action_type_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent action-type naming before export."""
    summary = action_type_summary_df.copy()
    if "xt_mean" not in summary.columns and "avg_xt" in summary.columns:
        summary = summary.rename(columns={"avg_xt": "xt_mean"})
    if "avg_xt" not in summary.columns and "xt_mean" in summary.columns:
        summary["avg_xt"] = summary["xt_mean"]
    return summary


def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame to parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def export_action_xt_table(
    actions_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Export the action-level xT table to `data/processed/actions.parquet`."""
    _validate_dataframe(actions_df, "actions_df")
    output_path = ensure_output_dir(output_dir) / DEFAULT_ACTIONS_PATH.name
    return _write_parquet(_normalize_actions_table(actions_df), output_path)


def export_player_summary_table(
    player_summary_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Export the player summary table to `data/processed/player_summary.parquet`."""
    _validate_dataframe(player_summary_df, "player_summary_df")
    output_path = ensure_output_dir(output_dir) / DEFAULT_PLAYER_SUMMARY_PATH.name
    return _write_parquet(_normalize_player_summary(player_summary_df), output_path)


def export_team_summary_table(
    team_summary_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Export the team summary table to `data/processed/team_summary.parquet`."""
    _validate_dataframe(team_summary_df, "team_summary_df")
    output_path = ensure_output_dir(output_dir) / DEFAULT_TEAM_SUMMARY_PATH.name
    return _write_parquet(_normalize_team_summary(team_summary_df), output_path)


def export_action_type_summary_table(
    action_type_summary_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    """Export the action-type summary table to `data/processed/action_type_summary.parquet`."""
    _validate_dataframe(action_type_summary_df, "action_type_summary_df")
    output_path = ensure_output_dir(output_dir) / DEFAULT_ACTION_TYPE_SUMMARY_PATH.name
    return _write_parquet(_normalize_action_type_summary(action_type_summary_df), output_path)


def export_all_processed_outputs(
    actions_df: pd.DataFrame,
    player_summary_df: pd.DataFrame,
    team_summary_df: pd.DataFrame,
    action_type_summary_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Export the main processed tables needed by the Quarto website."""
    ensure_output_dir(output_dir)
    return {
        "actions": export_action_xt_table(actions_df, output_dir=output_dir),
        "player_summary": export_player_summary_table(player_summary_df, output_dir=output_dir),
        "team_summary": export_team_summary_table(team_summary_df, output_dir=output_dir),
        "action_type_summary": export_action_type_summary_table(
            action_type_summary_df,
            output_dir=output_dir,
        ),
    }


if __name__ == "__main__":
    sample_actions = pd.DataFrame(
        {
            "match_id": [1, 1, 1],
            "action_id": [0, 1, 2],
            "player_name": ["Player A", "Player B", "Player A"],
            "team_name": ["Team X", "Team Y", "Team X"],
            "type_name": ["pass", "cross", "dribble"],
            "xt_start": [0.01, 0.02, 0.03],
            "xt_end": [0.04, 0.05, 0.07],
            "xt_delta": [0.03, 0.03, 0.04],
        }
    )

    sample_player_summary = pd.DataFrame(
        {
            "player_name": ["Player A", "Player B"],
            "team_name": ["Team X", "Team Y"],
            "minutes_played": [540, 450],
            "actions": [120, 95],
            "xt_total": [0.72, 0.51],
            "xt_per_90": [0.12, 0.10],
        }
    )

    sample_team_summary = pd.DataFrame(
        {
            "team_name": ["Team X", "Team Y"],
            "matches": [6, 5],
            "actions": [620, 575],
            "xt_total": [1.84, 1.22],
            "xt_mean": [0.0030, 0.0021],
        }
    )

    sample_action_type_summary = pd.DataFrame(
        {
            "type_name": ["pass", "cross", "dribble"],
            "actions": [420, 70, 55],
            "xt_total": [1.25, 0.44, 0.30],
            "xt_mean": [0.0030, 0.0063, 0.0055],
        }
    )

    exported = export_all_processed_outputs(
        actions_df=sample_actions,
        player_summary_df=sample_player_summary,
        team_summary_df=sample_team_summary,
        action_type_summary_df=sample_action_type_summary,
    )
    print("Exported files:")
    for group_name, output_path in exported.items():
        print(group_name, output_path)
