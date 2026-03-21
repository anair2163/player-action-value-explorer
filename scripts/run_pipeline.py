"""Run the first-pass football analytics pipeline."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings
from collections.abc import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.build_actions import build_player_minutes, convert_multiple_matches_to_actions
from src.compute_xt import (
    add_per90_metrics,
    aggregate_player_xt,
    aggregate_team_xt,
    get_action_type_xt_summary,
    score_actions_with_xt,
)
from src.export_outputs import DEFAULT_OUTPUT_DIR, export_all_processed_outputs
from src.load_data import get_match_lineups, get_matches, get_multiple_match_events


warnings.filterwarnings(
    "ignore",
    message="credentials were not supplied. open data access only",
)


# Configuration
COMPETITION_ID = 43
SEASON_ID = 3
OUTPUT_DIR = DEFAULT_OUTPUT_DIR


def _parse_match_clock(value: object) -> float:
    """Convert a StatsBomb lineup clock string like '80:01' into minutes."""
    if value is None:
        return 0.0
    text = str(value)
    if not text or text.lower() == "nan":
        return 0.0
    parts = text.split(":")
    if len(parts) != 2:
        return 0.0
    minutes, seconds = parts
    return float(minutes) + float(seconds) / 60.0


def _build_minutes_from_lineups(match_ids: Iterable[int]) -> object:
    """Build a player-minutes table from StatsBomb lineup position segments."""
    frames = []
    for match_id in match_ids:
        lineups_df = get_match_lineups(int(match_id))
        if lineups_df.empty or "positions" not in lineups_df.columns:
            continue

        lineup_rows = []
        for _, row in lineups_df.iterrows():
            positions = row.get("positions", [])
            if not isinstance(positions, list) or not positions:
                continue

            minutes_played = sum(
                max(_parse_match_clock(position.get("to")) - _parse_match_clock(position.get("from")), 0.0)
                for position in positions
                if isinstance(position, dict)
            )
            starts = int(
                any(
                    isinstance(position, dict) and position.get("start_reason") == "Starting XI"
                    for position in positions
                )
            )
            lineup_rows.append(
                {
                    "match_id": row.get("match_id"),
                    "player_id": row.get("player_id"),
                    "player_name": row.get("player_name"),
                    "team_name": row.get("team_name"),
                    "minutes_played": minutes_played,
                    "starts": starts,
                }
            )

        if lineup_rows:
            frames.append(pd.DataFrame(lineup_rows))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _print_step(message: str) -> None:
    """Print a concise pipeline progress update."""
    print(f"[pipeline] {message}")


def run_pipeline(
    competition_id: int = COMPETITION_ID,
    season_id: int = SEASON_ID,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, object]:
    """Run the full first-pass analytics pipeline and export processed outputs."""
    output_path = Path(output_dir)

    _print_step(f"Loading matches for competition_id={competition_id}, season_id={season_id}")
    matches_df = get_matches(competition_id=competition_id, season_id=season_id)
    if matches_df.empty:
        raise ValueError("No matches were returned for the selected competition and season.")

    match_ids = matches_df["match_id"].dropna().astype(int).tolist()
    if not match_ids:
        raise ValueError("No valid match IDs were found in the selected match set.")
    _print_step(f"Loaded {len(matches_df):,} matches")

    _print_step("Loading match events")
    events_df = get_multiple_match_events(match_ids)
    if events_df.empty:
        raise ValueError("No events were returned for the selected matches.")
    _print_step(f"Loaded {len(events_df):,} events")

    _print_step("Converting events to SPADL actions")
    events_by_match = {
        int(match_id): match_events.reset_index(drop=True)
        for match_id, match_events in events_df.groupby("match_id", sort=True)
    }
    actions_df = convert_multiple_matches_to_actions(events_by_match)
    if actions_df.empty:
        raise ValueError("SPADL conversion returned no actions.")
    match_context = matches_df.loc[
        :,
        [column for column in ["match_id", "match_date", "home_team", "away_team", "competition_name", "season_name"] if column in matches_df.columns],
    ].copy()
    actions_df = actions_df.merge(match_context, on="match_id", how="left")
    actions_df = actions_df.rename(
        columns={
            "competition_name": "competition",
            "season_name": "season",
        }
    )
    _print_step(f"Built {len(actions_df):,} actions")

    _print_step("Computing expected threat")
    scored_actions_df = score_actions_with_xt(actions_df)
    _print_step("Expected threat scoring complete")

    _print_step("Building player and team summaries")
    minutes_df = _build_minutes_from_lineups(match_ids)
    if minutes_df.empty:
        minutes_df = build_player_minutes(events_df)
    player_summary_df = aggregate_player_xt(scored_actions_df)
    if not minutes_df.empty:
        player_summary_df = add_per90_metrics(player_summary_df, minutes_df=minutes_df)
    team_summary_df = aggregate_team_xt(scored_actions_df)
    action_type_summary_df = get_action_type_xt_summary(scored_actions_df)
    _print_step(
        "Summaries ready: "
        f"players={len(player_summary_df):,}, teams={len(team_summary_df):,}, "
        f"action_types={len(action_type_summary_df):,}"
    )

    _print_step(f"Exporting processed outputs to {output_path}")
    export_results = export_all_processed_outputs(
        actions_df=scored_actions_df,
        player_summary_df=player_summary_df,
        team_summary_df=team_summary_df,
        action_type_summary_df=action_type_summary_df,
        output_dir=output_path,
    )
    _print_step(
        f"Actions exported: {export_results['actions']} "
        f"(rows={len(scored_actions_df):,})"
    )
    _print_step(
        f"Player summary exported: {export_results['player_summary']} "
        f"(rows={len(player_summary_df):,})"
    )
    _print_step(
        f"Team summary exported: {export_results['team_summary']} "
        f"(rows={len(team_summary_df):,})"
    )
    _print_step(
        f"Action type summary exported: {export_results['action_type_summary']} "
        f"(rows={len(action_type_summary_df):,})"
    )
    _print_step("Export complete")

    return {
        "matches": matches_df,
        "events": events_df,
        "actions": scored_actions_df,
        "player_summary": player_summary_df,
        "team_summary": team_summary_df,
        "action_type_summary": action_type_summary_df,
        "exports": export_results,
    }


def main() -> int:
    """Run the configured pipeline with basic error handling."""
    try:
        results = run_pipeline()
    except Exception as exc:
        print(f"[pipeline] Failed: {exc}")
        return 1

    print(
        "[pipeline] Done: "
        f"matches={len(results['matches']):,}, "
        f"events={len(results['events']):,}, "
        f"actions={len(results['actions']):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
