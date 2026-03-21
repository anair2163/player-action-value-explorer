"""Convert StatsBomb events into SPADL actions.

SPADL is a simple, consistent action table for football. It turns raw events
like passes, carries, shots, and duels into a cleaner action-based dataset
that is easier to value with models such as expected threat.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import pandas as pd
import socceraction.spadl as spadl
from socceraction.spadl.statsbomb import convert_to_actions


REQUIRED_EVENT_COLUMNS = {
    "match_id",
    "event_id",
    "index",
    "period",
    "timestamp",
    "minute",
    "second",
    "possession",
    "team_id",
    "team_name",
    "type_name",
}


def _empty_spadl_events() -> pd.DataFrame:
    """Return an empty frame with the columns socceraction expects."""
    return pd.DataFrame(
        columns=[
            "game_id",
            "event_id",
            "period_id",
            "team_id",
            "player_id",
            "type_name",
            "index",
            "timestamp",
            "minute",
            "second",
            "possession",
            "possession_team_id",
            "possession_team_name",
            "play_pattern_id",
            "play_pattern_name",
            "team_name",
            "duration",
            "extra",
            "related_events",
            "player_name",
            "position_id",
            "position_name",
            "location",
            "under_pressure",
            "counterpress",
        ]
    )


def _validate_events(events_df: pd.DataFrame) -> None:
    """Raise a clear error when required event columns are missing."""
    if events_df is None or events_df.empty:
        return

    missing = sorted(REQUIRED_EVENT_COLUMNS.difference(events_df.columns))
    if missing:
        raise ValueError(
            "Events DataFrame is missing required columns for SPADL conversion: "
            + ", ".join(missing)
        )


def _name_node(value: Any) -> dict[str, Any] | None:
    """Return a StatsBomb-style name node when a value exists."""
    if pd.isna(value):
        return None
    return {"name": value}


def _id_name_node(name: Any, identifier: Any) -> dict[str, Any] | None:
    """Return a StatsBomb-style id/name node when either value exists."""
    if pd.isna(name) and pd.isna(identifier):
        return None
    node: dict[str, Any] = {}
    if not pd.isna(identifier):
        node["id"] = identifier
    if not pd.isna(name):
        node["name"] = name
    return node or None


def _build_extra(row: pd.Series) -> dict[str, Any]:
    """Rebuild the nested StatsBomb `extra` field from flattened columns."""
    extra: dict[str, Any] = {}

    pass_data: dict[str, Any] = {}
    if "pass_end_location" in row.index and isinstance(row["pass_end_location"], list):
        pass_data["end_location"] = row["pass_end_location"]
    if (node := _name_node(row.get("pass_type"))) is not None:
        pass_data["type"] = node
    if (node := _name_node(row.get("pass_height"))) is not None:
        pass_data["height"] = node
    if (node := _name_node(row.get("pass_outcome"))) is not None:
        pass_data["outcome"] = node
    if (node := _name_node(row.get("pass_body_part"))) is not None:
        pass_data["body_part"] = node
    if (node := _id_name_node(row.get("pass_recipient"), row.get("pass_recipient_id"))) is not None:
        pass_data["recipient"] = node
    for column, key in {
        "pass_cross": "cross",
        "pass_cut_back": "cut_back",
        "pass_switch": "switch",
        "pass_shot_assist": "shot_assist",
        "pass_goal_assist": "goal_assist",
        "pass_through_ball": "through_ball",
        "pass_deflected": "deflected",
    }.items():
        value = row.get(column)
        if not pd.isna(value):
            pass_data[key] = bool(value)
    if pass_data:
        extra["pass"] = pass_data

    carry_data: dict[str, Any] = {}
    if "carry_end_location" in row.index and isinstance(row["carry_end_location"], list):
        carry_data["end_location"] = row["carry_end_location"]
    if carry_data:
        extra["carry"] = carry_data

    shot_data: dict[str, Any] = {}
    if "shot_end_location" in row.index and isinstance(row["shot_end_location"], list):
        shot_data["end_location"] = row["shot_end_location"]
    if (node := _name_node(row.get("shot_type"))) is not None:
        shot_data["type"] = node
    if (node := _name_node(row.get("shot_outcome"))) is not None:
        shot_data["outcome"] = node
    if (node := _name_node(row.get("shot_body_part"))) is not None:
        shot_data["body_part"] = node
    if shot_data:
        extra["shot"] = shot_data

    dribble_data: dict[str, Any] = {}
    if (node := _name_node(row.get("dribble_outcome"))) is not None:
        dribble_data["outcome"] = node
    if dribble_data:
        extra["dribble"] = dribble_data

    duel_data: dict[str, Any] = {}
    if (node := _name_node(row.get("duel_type"))) is not None:
        duel_data["type"] = node
    if (node := _name_node(row.get("duel_outcome"))) is not None:
        duel_data["outcome"] = node
    if duel_data:
        extra["duel"] = duel_data

    interception_data: dict[str, Any] = {}
    if (node := _name_node(row.get("interception_outcome"))) is not None:
        interception_data["outcome"] = node
    if interception_data:
        extra["interception"] = interception_data

    foul_data: dict[str, Any] = {}
    if (node := _name_node(row.get("foul_committed_card"))) is not None:
        foul_data["card"] = node
    if foul_data:
        extra["foul_committed"] = foul_data

    bad_behaviour_data: dict[str, Any] = {}
    if (node := _name_node(row.get("bad_behaviour_card"))) is not None:
        bad_behaviour_data["card"] = node
    if bad_behaviour_data:
        extra["bad_behaviour"] = bad_behaviour_data

    goalkeeper_data: dict[str, Any] = {}
    if "goalkeeper_end_location" in row.index and isinstance(row["goalkeeper_end_location"], list):
        goalkeeper_data["end_location"] = row["goalkeeper_end_location"]
    if (node := _name_node(row.get("goalkeeper_type"))) is not None:
        goalkeeper_data["type"] = node
    if (node := _name_node(row.get("goalkeeper_outcome"))) is not None:
        goalkeeper_data["outcome"] = node
    if (node := _name_node(row.get("goalkeeper_body_part"))) is not None:
        goalkeeper_data["body_part"] = node
    if goalkeeper_data:
        extra["goalkeeper"] = goalkeeper_data

    clearance_data: dict[str, Any] = {}
    if (node := _name_node(row.get("clearance_body_part"))) is not None:
        clearance_data["body_part"] = node
    if clearance_data:
        extra["clearance"] = clearance_data

    substitution_data: dict[str, Any] = {}
    if (node := _id_name_node(row.get("substitution_replacement"), row.get("substitution_replacement_id"))) is not None:
        substitution_data["replacement"] = node
    if (node := _name_node(row.get("substitution_outcome"))) is not None:
        substitution_data["outcome"] = node
    if substitution_data:
        extra["substitution"] = substitution_data

    return extra


def _extract_starting_players(match_events: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract starting lineups from Starting XI events."""
    starters: list[dict[str, Any]] = []
    if "tactics_lineup" not in match_events.columns:
        return starters

    starting_rows = match_events.loc[match_events["type_name"] == "Starting XI"].copy()
    for _, row in starting_rows.iterrows():
        lineup = row.get("tactics_lineup")
        if not isinstance(lineup, list):
            continue

        team_id = row.get("team_id")
        team_name = row.get("team_name")
        match_id = row.get("match_id")
        for player_entry in lineup:
            if not isinstance(player_entry, dict):
                continue

            player_info = player_entry.get("player", {}) if isinstance(player_entry.get("player"), dict) else {}
            player_id = player_entry.get("player_id", player_info.get("id"))
            player_name = player_entry.get("player_name", player_info.get("name"))
            if pd.isna(player_id) and pd.isna(player_name):
                continue

            starters.append(
                {
                    "match_id": match_id,
                    "team_id": team_id,
                    "team_name": team_name,
                    "player_id": player_id,
                    "player_name": player_name,
                    "start_minute": 0.0,
                    "end_minute": pd.NA,
                    "starts": 1,
                }
            )
    return starters


def _build_substitution_events(match_events: pd.DataFrame, match_end_minute: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return outgoing and incoming substitution minute tables."""
    substitutions = match_events.loc[match_events["type_name"] == "Substitution"].copy()
    if substitutions.empty:
        empty = pd.DataFrame(columns=["player_id", "team_id", "minute"])
        return empty, empty

    substitutions["minute_float"] = (
        pd.to_numeric(substitutions.get("minute"), errors="coerce").fillna(0)
        + pd.to_numeric(substitutions.get("second"), errors="coerce").fillna(0) / 60.0
    ).clip(lower=0, upper=match_end_minute)

    outgoing = substitutions.loc[
        :,
        [column for column in ["player_id", "team_id", "minute_float"] if column in substitutions.columns],
    ].rename(columns={"minute_float": "minute"})

    incoming_columns = [
        column
        for column in [
            "substitution_replacement_id",
            "substitution_replacement",
            "team_id",
            "team_name",
            "match_id",
            "minute_float",
        ]
        if column in substitutions.columns
    ]
    incoming = substitutions.loc[:, incoming_columns].rename(
        columns={
            "substitution_replacement_id": "player_id",
            "substitution_replacement": "player_name",
            "minute_float": "minute",
        }
    )
    return outgoing, incoming


def _build_match_player_minutes(match_events: pd.DataFrame) -> pd.DataFrame:
    """Build one-match player appearances from Starting XI and Substitution events."""
    if match_events.empty:
        return pd.DataFrame()

    minute_series = pd.to_numeric(match_events.get("minute"), errors="coerce").fillna(0)
    second_series = pd.to_numeric(match_events.get("second"), errors="coerce").fillna(0)
    match_end_minute = max(90.0, float((minute_series + second_series / 60.0).max()))

    starters = pd.DataFrame(_extract_starting_players(match_events))
    outgoing_subs, incoming_subs = _build_substitution_events(match_events, match_end_minute)

    if not starters.empty and not outgoing_subs.empty:
        outgoing_first = (
            outgoing_subs.dropna(subset=["player_id"])
            .sort_values("minute")
            .drop_duplicates(["player_id", "team_id"], keep="first")
            .rename(columns={"minute": "sub_out_minute"})
        )
        starters = starters.merge(outgoing_first, on=["player_id", "team_id"], how="left")
        starters["end_minute"] = starters["sub_out_minute"].fillna(match_end_minute)
        starters = starters.drop(columns=["sub_out_minute"])
    elif not starters.empty:
        starters["end_minute"] = match_end_minute

    appearances = starters.copy()

    if not incoming_subs.empty:
        if not outgoing_subs.empty:
            outgoing_first = (
                outgoing_subs.dropna(subset=["player_id"])
                .sort_values("minute")
                .drop_duplicates(["player_id", "team_id"], keep="first")
                .rename(columns={"minute": "sub_out_minute"})
            )
            incoming_subs = incoming_subs.merge(outgoing_first, on=["player_id", "team_id"], how="left")
            incoming_subs["end_minute"] = incoming_subs["sub_out_minute"].fillna(match_end_minute)
            incoming_subs = incoming_subs.drop(columns=["sub_out_minute"])
        else:
            incoming_subs["end_minute"] = match_end_minute

        incoming_subs = incoming_subs.rename(columns={"minute": "start_minute"})
        incoming_subs["starts"] = 0
        appearances = pd.concat([appearances, incoming_subs], ignore_index=True, sort=False)

    if appearances.empty:
        return pd.DataFrame()

    appearances["minutes_played"] = (
        pd.to_numeric(appearances["end_minute"], errors="coerce").fillna(match_end_minute)
        - pd.to_numeric(appearances["start_minute"], errors="coerce").fillna(0)
    ).clip(lower=0)

    output_columns = [
        column
        for column in [
            "match_id",
            "team_id",
            "team_name",
            "player_id",
            "player_name",
            "start_minute",
            "end_minute",
            "minutes_played",
            "starts",
        ]
        if column in appearances.columns
    ]
    appearances = appearances.loc[:, output_columns]
    return appearances.drop_duplicates(["match_id", "team_id", "player_id"], keep="first").reset_index(drop=True)


def _infer_home_team_id(events_df: pd.DataFrame) -> int:
    """Infer the home team id from the StatsBomb event stream when not supplied."""
    if "team_id" not in events_df.columns:
        raise ValueError("Cannot infer home_team_id because `team_id` is missing.")

    starting_xi = events_df.loc[events_df["type_name"] == "Starting XI", "team_id"].dropna()
    if not starting_xi.empty:
        return int(starting_xi.iloc[0])

    first_team = events_df["team_id"].dropna()
    if first_team.empty:
        raise ValueError("Cannot infer home_team_id from an event stream without team identifiers.")
    return int(first_team.iloc[0])


def prepare_events_for_spadl(events_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare StatsBomb events for socceraction's SPADL converter.

    This converts a flat event table into the column structure expected by
    `socceraction.spadl.statsbomb.convert_to_actions`.
    """
    if events_df is None or events_df.empty:
        return _empty_spadl_events()

    _validate_events(events_df)

    prepared = events_df.copy()
    rename_map = {
        "match_id": "game_id",
        "period": "period_id",
        "possession_team": "possession_team_name",
        "play_pattern": "play_pattern_name",
        "position": "position_name",
    }
    prepared = prepared.rename(columns={old: new for old, new in rename_map.items() if old in prepared.columns})

    defaults: dict[str, Any] = {
        "player_id": pd.NA,
        "player_name": pd.NA,
        "position_id": pd.NA,
        "position_name": pd.NA,
        "possession_team_id": pd.NA,
        "possession_team_name": pd.NA,
        "play_pattern_id": pd.NA,
        "play_pattern_name": pd.NA,
        "duration": 0.0,
        "location": pd.NA,
        "related_events": None,
        "under_pressure": False,
        "counterpress": False,
    }
    for column, default_value in defaults.items():
        if column not in prepared.columns:
            prepared[column] = default_value

    if "timestamp" in prepared.columns:
        prepared["timestamp"] = pd.to_timedelta(prepared["timestamp"], errors="coerce")
    if "related_events" in prepared.columns:
        prepared["related_events"] = prepared["related_events"].apply(
            lambda value: value if isinstance(value, list) else []
        )
    for column in ["under_pressure", "counterpress"]:
        prepared[column] = prepared[column].fillna(False).astype(bool)

    prepared["extra"] = prepared.apply(_build_extra, axis=1)

    spadl_columns = [
        "game_id",
        "event_id",
        "period_id",
        "team_id",
        "player_id",
        "type_name",
        "index",
        "timestamp",
        "minute",
        "second",
        "possession",
        "possession_team_id",
        "possession_team_name",
        "play_pattern_id",
        "play_pattern_name",
        "team_name",
        "duration",
        "extra",
        "related_events",
        "player_name",
        "position_id",
        "position_name",
        "location",
        "under_pressure",
        "counterpress",
    ]
    prepared = prepared.loc[:, spadl_columns]
    prepared = prepared.sort_values(
        ["game_id", "period_id", "timestamp", "index"],
        kind="mergesort",
    ).reset_index(drop=True)
    return prepared


def convert_match_events_to_actions(
    events_df: pd.DataFrame,
    home_team_id: int | None = None,
) -> pd.DataFrame:
    """Convert one match of StatsBomb events into SPADL actions."""
    if events_df is None or events_df.empty:
        return pd.DataFrame()

    prepared_events = prepare_events_for_spadl(events_df)
    resolved_home_team_id = home_team_id if home_team_id is not None else _infer_home_team_id(events_df)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        actions = convert_to_actions(
            prepared_events,
            home_team_id=int(resolved_home_team_id),
        )

    actions = spadl.add_names(actions)
    return attach_action_metadata(actions.reset_index(drop=True), events_df)


def convert_multiple_matches_to_actions(
    events_by_match: Mapping[Any, pd.DataFrame] | list[pd.DataFrame],
) -> pd.DataFrame:
    """Convert several matches into one clean SPADL action table."""
    if events_by_match is None:
        return pd.DataFrame()

    action_frames: list[pd.DataFrame] = []

    if isinstance(events_by_match, Mapping):
        iterable = events_by_match.values()
    else:
        iterable = events_by_match

    for events_df in iterable:
        if events_df is None or events_df.empty:
            continue
        actions = convert_match_events_to_actions(events_df)
        if not actions.empty:
            action_frames.append(actions)

    if not action_frames:
        return pd.DataFrame()
    return pd.concat(action_frames, ignore_index=True)


def attach_action_metadata(
    actions_df: pd.DataFrame,
    events_df: pd.DataFrame,
    match_id: int | None = None,
    competition: str | None = None,
    season: str | None = None,
) -> pd.DataFrame:
    """Attach player, team, and optional match context to SPADL actions."""
    if actions_df is None or actions_df.empty:
        return pd.DataFrame()

    actions = actions_df.copy()
    if events_df is not None and not events_df.empty:
        if {"player_id", "player_name"}.issubset(events_df.columns):
            player_lookup = (
                events_df.loc[events_df["player_id"].notna(), ["player_id", "player_name"]]
                .drop_duplicates("player_id")
            )
            if not player_lookup.empty:
                actions = actions.merge(player_lookup, on="player_id", how="left")

        if {"team_id", "team_name"}.issubset(events_df.columns):
            team_lookup = (
                events_df.loc[events_df["team_id"].notna(), ["team_id", "team_name"]]
                .drop_duplicates("team_id")
            )
            if not team_lookup.empty:
                actions = actions.merge(team_lookup, on="team_id", how="left")

        if match_id is None and "match_id" in events_df.columns and events_df["match_id"].nunique() == 1:
            match_id = int(events_df["match_id"].iloc[0])

    if match_id is not None:
        actions["match_id"] = match_id
    elif "game_id" in actions.columns and "match_id" not in actions.columns:
        actions["match_id"] = actions["game_id"]

    if competition is not None:
        actions["competition"] = competition
    if season is not None:
        actions["season"] = season

    return actions.reset_index(drop=True)


def build_spadl_actions(events: pd.DataFrame, matches: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compatibility wrapper for existing project code."""
    if events is None or events.empty:
        return pd.DataFrame()

    action_frames: list[pd.DataFrame] = []
    match_meta = pd.DataFrame() if matches is None else matches.copy()

    for match_id, match_events in events.groupby("match_id", sort=True):
        home_team_id = None
        competition = None
        season = None

        if not match_meta.empty and "game_id" in match_meta.columns:
            match_row = match_meta.loc[match_meta["game_id"] == match_id]
            if not match_row.empty:
                if "home_team_id" in match_row.columns:
                    home_team_id = int(match_row.iloc[0]["home_team_id"])
                competition = match_row.iloc[0].get("competition_label")
                season = match_row.iloc[0].get("season_label")

        match_actions = convert_match_events_to_actions(match_events, home_team_id=home_team_id)
        match_actions = attach_action_metadata(
            match_actions,
            match_events,
            match_id=int(match_id),
            competition=competition,
            season=season,
        )
        action_frames.append(match_actions)

    if not action_frames:
        return pd.DataFrame()
    return pd.concat(action_frames, ignore_index=True)


def build_player_minutes(events: pd.DataFrame) -> pd.DataFrame:
    """Return player appearance and minutes-played data from prepared events."""
    if events is None or events.empty:
        return pd.DataFrame()

    appearance_frames: list[pd.DataFrame] = []
    for _, match_events in events.groupby("match_id", sort=True):
        match_appearances = _build_match_player_minutes(match_events)
        if not match_appearances.empty:
            appearance_frames.append(match_appearances)

    if not appearance_frames:
        return pd.DataFrame()
    return pd.concat(appearance_frames, ignore_index=True)


if __name__ == "__main__":
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

    print(f"Sample match: {sample_match_id}")
    print(f"Events: {len(events):,}")
    print(f"Actions: {len(actions):,}")
    columns = [
        column
        for column in [
            "match_id",
            "action_id",
            "type_name",
            "result_name",
            "team_name",
            "player_name",
            "start_x",
            "start_y",
            "end_x",
            "end_y",
        ]
        if column in actions.columns
    ]
    print(actions[columns].head(10).to_string(index=False))
