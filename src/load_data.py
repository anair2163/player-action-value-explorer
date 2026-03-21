"""Reusable loaders for StatsBomb Open Data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
from statsbombpy import sb


def _safe_dataframe(records: Any) -> pd.DataFrame:
    """Return a DataFrame from a statsbombpy payload."""
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, dict):
        return pd.DataFrame(records)
    return pd.DataFrame(records or [])


def _rename_if_present(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    """Rename only columns that exist in the input DataFrame."""
    applicable = {old: new for old, new in rename_map.items() if old in df.columns}
    return df.rename(columns=applicable)


def _standardize_competitions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent competition column names."""
    competitions = _rename_if_present(
        df,
        {
            "competition_name": "competition_name",
            "season_name": "season_name",
            "match_updated": "match_updated",
            "match_available": "match_available",
        },
    )
    preferred_order = [
        "competition_id",
        "competition_name",
        "country_name",
        "season_id",
        "season_name",
        "competition_gender",
        "competition_youth",
        "competition_international",
        "match_updated",
        "match_available",
    ]
    ordered = [column for column in preferred_order if column in competitions.columns]
    remaining = [column for column in competitions.columns if column not in ordered]
    return competitions.loc[:, ordered + remaining]


def _standardize_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent match column names."""
    matches = _rename_if_present(
        df,
        {
            "match_id": "match_id",
            "home_team": "home_team",
            "away_team": "away_team",
            "competition": "competition_name",
            "season": "season_name",
            "competition_stage": "competition_stage",
            "stadium": "stadium_name",
            "referee": "referee_name",
        },
    )
    if "match_date" in matches.columns:
        matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")

    preferred_order = [
        "match_id",
        "match_date",
        "kick_off",
        "competition_id",
        "competition_name",
        "season_id",
        "season_name",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "competition_stage",
        "match_week",
        "stadium_name",
        "referee_name",
    ]
    ordered = [column for column in preferred_order if column in matches.columns]
    remaining = [column for column in matches.columns if column not in ordered]
    return matches.loc[:, ordered + remaining]


def _standardize_events(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent event column names and types."""
    events = _rename_if_present(
        df,
        {
            "id": "event_id",
            "type": "type_name",
            "team": "team_name",
            "player": "player_name",
            "play_pattern": "play_pattern_name",
            "position": "position_name",
        },
    )

    if "timestamp" in events.columns:
        events["timestamp"] = pd.to_timedelta(events["timestamp"], errors="coerce")
    if "related_events" in events.columns:
        events["related_events"] = events["related_events"].apply(
            lambda value: value if isinstance(value, list) else []
        )
    for column in ["counterpress", "under_pressure", "off_camera", "out"]:
        if column in events.columns:
            events[column] = events[column].astype("boolean").fillna(False).astype(bool)

    preferred_order = [
        "match_id",
        "event_id",
        "index",
        "period",
        "timestamp",
        "minute",
        "second",
        "possession",
        "possession_team_id",
        "possession_team",
        "play_pattern_name",
        "team_id",
        "team_name",
        "player_id",
        "player_name",
        "position_name",
        "type_name",
        "location",
        "duration",
        "related_events",
    ]
    ordered = [column for column in preferred_order if column in events.columns]
    remaining = [column for column in events.columns if column not in ordered]
    events = events.loc[:, ordered + remaining]

    sort_columns = [column for column in ["match_id", "period", "timestamp", "index"] if column in events.columns]
    if sort_columns:
        events = events.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return events


def _standardize_lineups(lineups: dict[str, pd.DataFrame] | pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Convert the statsbombpy lineups payload into one tidy DataFrame."""
    if isinstance(lineups, pd.DataFrame):
        frame = lineups.copy()
        if "team_name" not in frame.columns:
            frame["team_name"] = pd.NA
        frame["match_id"] = match_id
        return frame

    lineup_frames: list[pd.DataFrame] = []
    for team_name, team_df in lineups.items():
        frame = team_df.copy()
        frame["team_name"] = team_name
        frame["match_id"] = match_id
        lineup_frames.append(frame)

    if not lineup_frames:
        return pd.DataFrame()

    lineup_df = pd.concat(lineup_frames, ignore_index=True)
    lineup_df = _rename_if_present(
        lineup_df,
        {
            "player_name": "player_name",
            "player_nickname": "player_nickname",
            "jersey_number": "jersey_number",
            "country": "country_name",
        },
    )

    preferred_order = [
        "match_id",
        "team_name",
        "player_id",
        "player_name",
        "player_nickname",
        "jersey_number",
        "country_name",
        "positions",
        "cards",
    ]
    ordered = [column for column in preferred_order if column in lineup_df.columns]
    remaining = [column for column in lineup_df.columns if column not in ordered]
    return lineup_df.loc[:, ordered + remaining]


def get_competitions() -> pd.DataFrame:
    """Return available StatsBomb Open Data competitions."""
    try:
        competitions = sb.competitions(fmt="dataframe")
    except Exception as exc:
        raise RuntimeError("Failed to load StatsBomb competitions.") from exc
    return _standardize_competitions(_safe_dataframe(competitions))


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return match metadata for a competition and season."""
    try:
        matches = sb.matches(competition_id=int(competition_id), season_id=int(season_id), fmt="dataframe")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load matches for competition_id={competition_id}, season_id={season_id}."
        ) from exc

    if matches.empty:
        return pd.DataFrame()
    return _standardize_matches(_safe_dataframe(matches))


def get_match_events(match_id: int) -> pd.DataFrame:
    """Return event-level data for a single match."""
    try:
        events = sb.events(match_id=int(match_id), fmt="dataframe", flatten_attrs=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to load events for match_id={match_id}.") from exc

    if events.empty:
        return pd.DataFrame()
    return _standardize_events(_safe_dataframe(events))


def get_match_lineups(match_id: int) -> pd.DataFrame:
    """Return lineups for a single match as one tidy DataFrame."""
    try:
        lineups = sb.lineups(match_id=int(match_id), fmt="dataframe")
    except Exception as exc:
        raise RuntimeError(f"Failed to load lineups for match_id={match_id}.") from exc

    lineup_df = _standardize_lineups(lineups, match_id=int(match_id))
    if lineup_df.empty:
        return pd.DataFrame()
    return lineup_df.reset_index(drop=True)


def get_multiple_match_events(match_ids: Iterable[int]) -> pd.DataFrame:
    """Return concatenated event data for multiple matches."""
    match_ids = [int(match_id) for match_id in match_ids]
    if not match_ids:
        return pd.DataFrame()

    event_frames: list[pd.DataFrame] = []
    for match_id in match_ids:
        events = get_match_events(match_id)
        if not events.empty:
            event_frames.append(events)

    if not event_frames:
        return pd.DataFrame()
    return pd.concat(event_frames, ignore_index=True)


if __name__ == "__main__":
    competitions_df = get_competitions()
    print("Available competitions:")
    print(
        competitions_df[
            ["competition_id", "competition_name", "season_id", "season_name"]
        ].head(10).to_string(index=False)
    )

    if competitions_df.empty:
        raise SystemExit("No competitions returned from StatsBomb Open Data.")

    sample_competition = competitions_df.iloc[0]
    competition_id = int(sample_competition["competition_id"])
    season_id = int(sample_competition["season_id"])

    print(f"\nLoading matches for competition_id={competition_id}, season_id={season_id}")
    matches_df = get_matches(competition_id=competition_id, season_id=season_id)
    print(matches_df[["match_id", "home_team", "away_team", "match_date"]].head(5).to_string(index=False))

    if matches_df.empty:
        raise SystemExit("No matches returned for the sample competition and season.")

    sample_match_ids = matches_df["match_id"].head(3).tolist()
    print(f"\nLoading events for matches: {sample_match_ids}")
    multi_events_df = get_multiple_match_events(sample_match_ids)
    print(f"Loaded {len(multi_events_df):,} events across {len(sample_match_ids)} matches.")

    sample_match_id = int(sample_match_ids[0])
    print(f"\nLoading one sample match: {sample_match_id}")
    sample_events_df = get_match_events(sample_match_id)
    sample_columns = [
        column
        for column in ["event_id", "period", "minute", "second", "type_name", "team_name", "player_name"]
        if column in sample_events_df.columns
    ]
    print(sample_events_df[sample_columns].head(10).to_string(index=False))
