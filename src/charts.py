"""Plotly chart helpers for the Quarto reporting pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOT_TEMPLATE = "plotly_white"
TEXT_COLOR = "#1a2530"
GRID_COLOR = "#d8dde3"
PRIMARY_COLOR = "#0f5c4d"
BAR_COLORS = px.colors.sequential.Tealgrn


def _validate_dataframe(df: pd.DataFrame | None, required_columns: list[str], name: str) -> pd.DataFrame:
    """Validate a chart input table and return a copy."""
    if df is None:
        raise ValueError(f"`{name}` is not available.")
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"`{name}` must be a pandas DataFrame.")
    if df.empty:
        raise ValueError(f"`{name}` is empty.")

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"`{name}` is missing columns: {', '.join(missing)}.")

    return df.copy()


def _validate_positive_int(value: int, name: str) -> None:
    """Validate a positive integer argument."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"`{name}` must be a positive integer.")


def _apply_layout(fig: go.Figure, title: str, height: int = 520) -> go.Figure:
    """Apply a consistent reporting layout to a Plotly figure."""
    fig.update_layout(
        title=title,
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=24, r=24, t=72, b=24),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color=TEXT_COLOR),
        title_font=dict(size=22, color=TEXT_COLOR),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    return fig


def plot_top_players_total_xt(player_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Plot the leading players by total expected threat."""
    _validate_positive_int(top_n, "top_n")
    chart_df = _validate_dataframe(
        player_df,
        ["player_name", "team_name", "total_xt"],
        "player_df",
    )

    chart_df = (
        chart_df.sort_values("total_xt", ascending=False)
        .head(top_n)
        .sort_values("total_xt", ascending=True)
    )
    if chart_df.empty:
        raise ValueError("No player rows are available for the total xT chart.")

    fig = px.bar(
        chart_df,
        x="total_xt",
        y="player_name",
        orientation="h",
        color_discrete_sequence=[PRIMARY_COLOR],
        text="total_xt",
        hover_name="player_name",
        hover_data={"team_name": True, "total_xt": ":.2f", "minutes_played": True},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_xaxes(title="Total xT")
    fig.update_yaxes(title=None)
    return _apply_layout(fig, "Top Players By Total xT", height=540)


def plot_top_players_xt_per90(
    player_df: pd.DataFrame,
    top_n: int = 15,
    min_minutes: int = 90,
) -> go.Figure:
    """Plot the leading players by xT per 90."""
    _validate_positive_int(top_n, "top_n")
    _validate_positive_int(min_minutes, "min_minutes")
    chart_df = _validate_dataframe(
        player_df,
        ["player_name", "team_name", "minutes_played", "xt_per90"],
        "player_df",
    )

    chart_df = (
        chart_df.loc[
            (chart_df["minutes_played"].fillna(0) >= min_minutes)
            & chart_df["xt_per90"].notna()
        ]
        .sort_values("xt_per90", ascending=False)
        .head(top_n)
        .sort_values("xt_per90", ascending=True)
    )
    if chart_df.empty:
        raise ValueError(f"No player rows are available above {min_minutes} minutes.")

    fig = px.bar(
        chart_df,
        x="xt_per90",
        y="player_name",
        orientation="h",
        color_discrete_sequence=[PRIMARY_COLOR],
        text="xt_per90",
        hover_name="player_name",
        hover_data={"team_name": True, "xt_per90": ":.2f", "minutes_played": True},
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_xaxes(title="xT per 90")
    fig.update_yaxes(title=None)
    return _apply_layout(fig, f"Top Players By xT Per 90 ({min_minutes}+ Minutes)", height=540)


def plot_xt_by_action_type(action_type_df: pd.DataFrame) -> go.Figure:
    """Plot total expected threat by action type."""
    chart_df = _validate_dataframe(
        action_type_df,
        ["type_name", "xt_total", "actions"],
        "action_type_df",
    )
    chart_df = chart_df.sort_values("xt_total", ascending=False)
    if chart_df.empty:
        raise ValueError("No action-type rows are available for charting.")

    fig = px.bar(
        chart_df,
        x="type_name",
        y="xt_total",
        color="xt_total",
        color_continuous_scale="Tealgrn",
        hover_name="type_name",
        hover_data={"actions": True, "xt_total": ":.2f", "xt_mean": ":.4f"},
    )
    fig.update_xaxes(title="Action Type", categoryorder="total descending")
    fig.update_yaxes(title="Total xT")
    return _apply_layout(fig, "xT By Action Type", height=500)


def plot_team_progression_vs_final_third(team_df: pd.DataFrame) -> go.Figure:
    """Plot team progression threat against final-third threat."""
    chart_df = _validate_dataframe(
        team_df,
        ["team_name", "progression_xt", "final_third_threat", "total_xt", "matches_played"],
        "team_df",
    )

    fig = px.scatter(
        chart_df,
        x="progression_xt",
        y="final_third_threat",
        text="team_name",
        size="matches_played",
        color="total_xt",
        color_continuous_scale="Tealgrn",
        hover_name="team_name",
        hover_data={
            "matches_played": True,
            "total_xt": ":.2f",
            "xt_per_match": ":.2f",
            "progression_xt": ":.2f",
            "final_third_threat": ":.2f",
        },
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(line=dict(width=1, color="white"), opacity=0.85),
    )
    fig.update_xaxes(title="Progression xT")
    fig.update_yaxes(title="Final-Third xT")
    return _apply_layout(fig, "Team Progression xT Vs Final-Third Threat", height=550)


def plot_teams_xt_per_match(team_df: pd.DataFrame) -> go.Figure:
    """Plot team xT per match."""
    chart_df = _validate_dataframe(
        team_df,
        ["team_name", "xt_per_match"],
        "team_df",
    )
    chart_df = chart_df.sort_values("xt_per_match", ascending=False)
    if chart_df.empty:
        raise ValueError("No team rows are available for the xT per match chart.")

    fig = px.bar(
        chart_df,
        x="team_name",
        y="xt_per_match",
        color="xt_per_match",
        color_continuous_scale="Tealgrn",
        hover_name="team_name",
        hover_data={"xt_per_match": ":.2f", "total_xt": ":.2f", "matches_played": True},
    )
    fig.update_xaxes(title="Team", categoryorder="total descending")
    fig.update_yaxes(title="xT per Match")
    return _apply_layout(fig, "Teams By xT Per Match", height=520)


def plot_player_action_mix(action_mix_df: pd.DataFrame) -> go.Figure:
    """Plot the action-type mix behind player xT."""
    chart_df = _validate_dataframe(
        action_mix_df,
        ["player_name", "action_type", "total_xt"],
        "action_mix_df",
    )
    chart_df = chart_df.sort_values(["player_name", "total_xt"], ascending=[True, False])
    if chart_df.empty:
        raise ValueError("No action-mix rows are available for charting.")

    fig = px.bar(
        chart_df,
        x="player_name",
        y="total_xt",
        color="action_type",
        barmode="stack",
        color_discrete_sequence=BAR_COLORS,
        hover_name="player_name",
        hover_data={"total_xt": ":.2f"},
    )
    fig.update_xaxes(title="Player")
    fig.update_yaxes(title="Total xT")
    return _apply_layout(fig, "Action Mix Behind Player xT", height=550)
