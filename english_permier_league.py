import os
import requests
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(
    page_title="EPL Final Table Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ EPL Final Table Predictor")
st.caption("Team strength + fixtures + rescheduled matches + European fatigue + live market odds")

BASE_DIR = os.path.dirname(__file__)

HOME_ADVANTAGE = 0.15
EUROPE_FATIGUE = 1.2
MODEL_WEIGHT = 0.3
MARKET_WEIGHT = 0.7

TEAM_NAME_MAP = {
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Tottenham Hotspur FC": "Tottenham",
    "Nottingham Forest FC": "Nottingham Forest",
    "Newcastle United FC": "Newcastle",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Brighton & Hove Albion FC": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Crystal Palace FC": "Crystal Palace",
    "Aston Villa FC": "Aston Villa",
    "Burnley FC": "Burnley",
    "Leeds United FC": "Leeds",
    "Sunderland AFC": "Sunderland"
}

st.sidebar.header("Live Odds Settings")

api_key = st.sidebar.text_input(
    "The Odds API Key",
    type="password",
    key="odds_api_key"
)

use_live_odds = st.sidebar.checkbox(
    "Use live odds",
    value=False,
    key="use_live_odds_checkbox"
)

football_data_api_key = st.sidebar.text_input(
    "Football-Data API Key",
    type="password",
    key="football_data_api_key"
)

use_live_results = st.sidebar.checkbox(
    "Use live results",
    value=True,
    key="use_live_results_checkbox"
)

use_live_score = st.sidebar.checkbox(
    "Use live score",
    value=True,
    key="use_live_score_checkbox"
)

deterministic = st.sidebar.checkbox(
    "Deterministic Mode",
    value=True,
    key="deterministic_mode_checkbox"
)

if deterministic:
    np.random.seed(42)

upset_gap = st.sidebar.slider(
    "Upset Level",
    0.05,
    0.20,
    0.12,
    key="upset_level_slider"
)

num_simulations = st.sidebar.slider(
    "Number of Simulations",
    100,
    5000,
    1000,
    step=100,
    key="num_simulations_slider"
)


@st.cache_data
def load_data():
    standings = pd.read_csv(os.path.join(BASE_DIR, "english_premier_league_standings.csv"))
    epl_fixtures = pd.read_csv(os.path.join(BASE_DIR, "epl_fixtures.csv"))
    rescheduled = pd.read_csv(os.path.join(BASE_DIR, "epl_rescheduled.csv"))
    europe = pd.read_csv(os.path.join(BASE_DIR, "europe_fixtures.csv"))
    return standings, epl_fixtures, rescheduled, europe


@st.cache_data(ttl=3600)
def fetch_live_odds(api_key):
    url = "https://api.the-odds-api.com/v4/sports/soccer_epl/odds"

    params = {
        "apiKey": api_key,
        "regions": "uk,eu,au",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "dateFormat": "iso"
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()
        rows = []

        for match in data:
            if not match.get("bookmakers"):
                continue

            home_team = match["home_team"]
            away_team = match["away_team"]

            bookmaker = match["bookmakers"][0]

            if not bookmaker.get("markets"):
                continue

            outcomes = bookmaker["markets"][0]["outcomes"]
            odds_dict = {item["name"]: item["price"] for item in outcomes}

            rows.append({
                "home_team": home_team,
                "away_team": away_team,
                "home_odds": odds_dict.get(home_team),
                "draw_odds": odds_dict.get("Draw"),
                "away_odds": odds_dict.get(away_team)
            })

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def fetch_finished_epl_results(api_key):
    url = "https://api.football-data.org/v4/competitions/PL/matches"

    headers = {
        "X-Auth-Token": api_key
    }

    params = {
        "status": "FINISHED"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()
        rows = []

        for match in data.get("matches", []):
            home_team = match["homeTeam"]["name"]
            away_team = match["awayTeam"]["name"]

            home_score = match["score"]["fullTime"]["home"]
            away_score = match["score"]["fullTime"]["away"]

            if home_score is None or away_score is None:
                continue

            rows.append({
                "date": match.get("utcDate"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "competition": "EPL"
            })

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_live_epl_scores(api_key):
    url = "https://api.football-data.org/v4/competitions/PL/matches"

    headers = {
        "X-Auth-Token": api_key
    }

    params = {
        "status": "LIVE"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()
        rows = []

        for match in data.get("matches", []):
            home_team = match["homeTeam"]["name"]
            away_team = match["awayTeam"]["name"]

            home_score = match["score"]["fullTime"]["home"]
            away_score = match["score"]["fullTime"]["away"]

            if home_score is None:
                home_score = 0

            if away_score is None:
                away_score = 0

            rows.append({
                "date": match.get("utcDate"),
                "status": match.get("status"),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "competition": "EPL"
            })

        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame()


def clean_standings(df):
    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "GP": "played",
        "P": "points",
        "GD": "gd",
        "F": "goals_for",
        "A": "goals_against"
    })

    numeric_columns = ["played", "points", "gd", "goals_for", "goals_against"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ppg"] = df["points"] / df["played"]
    df["attack"] = df["goals_for"] / df["played"]
    df["defense"] = df["goals_against"] / df["played"]

    df["strength"] = (
        df["ppg"] * 45
        + df["gd"] * 0.6
        + df["attack"] * 10
        - df["defense"] * 8
    )

    return df


def odds_to_probs(home_odds, draw_odds, away_odds):
    if pd.isna(home_odds) or pd.isna(draw_odds) or pd.isna(away_odds):
        return None

    home = 1 / float(home_odds)
    draw = 1 / float(draw_odds)
    away = 1 / float(away_odds)

    total = home + draw + away
    return home / total, draw / total, away / total


def model_probs(diff):
    if diff > 8:
        return 0.65, 0.22, 0.13
    elif diff > 3:
        return 0.52, 0.27, 0.21
    elif diff >= -3:
        return 0.34, 0.32, 0.34
    elif diff >= -8:
        return 0.21, 0.27, 0.52
    else:
        return 0.13, 0.22, 0.65


def get_european_epl_teams(standings, europe):
    epl_teams = set(standings["team"])

    europe_teams = set(europe["home_team"]).union(
        set(europe["away_team"])
    )

    europe_epl_teams = europe_teams.intersection(epl_teams)

    guaranteed_finalist_pressure = ["Aston Villa", "Nottingham Forest"]

    for team in guaranteed_finalist_pressure:
        if team in epl_teams:
            europe_epl_teams.add(team)

    return europe_epl_teams


def predict_match_probabilities(table, row, europe_epl_teams):
    home_team = row["home_team"]
    away_team = row["away_team"]

    home = table[table["team"] == home_team].iloc[0]
    away = table[table["team"] == away_team].iloc[0]

    home_strength = home["strength"] + HOME_ADVANTAGE
    away_strength = away["strength"]

    if home_team in europe_epl_teams:
        home_strength -= EUROPE_FATIGUE

    if away_team in europe_epl_teams:
        away_strength -= EUROPE_FATIGUE

    diff = home_strength - away_strength

    model_home, model_draw, model_away = model_probs(diff)

    odds_result = odds_to_probs(
        row.get("home_odds"),
        row.get("draw_odds"),
        row.get("away_odds")
    )

    if odds_result:
        market_home, market_draw, market_away = odds_result

        final_home = model_home * MODEL_WEIGHT + market_home * MARKET_WEIGHT
        final_draw = model_draw * MODEL_WEIGHT + market_draw * MARKET_WEIGHT
        final_away = model_away * MODEL_WEIGHT + market_away * MARKET_WEIGHT

        total = final_home + final_draw + final_away
        final_home /= total
        final_draw /= total
        final_away /= total

    else:
        market_home, market_draw, market_away = None, None, None
        final_home, final_draw, final_away = model_home, model_draw, model_away

    return (
        model_home,
        model_draw,
        model_away,
        market_home,
        market_draw,
        market_away,
        final_home,
        final_draw,
        final_away
    )


def sample_match_result(final_home, final_draw, final_away):
    gap = abs(final_home - final_away)

    if gap <= upset_gap or final_draw >= 0.27:
        return "draw"

    return np.random.choice(
        ["home_win", "draw", "away_win"],
        p=[final_home, final_draw, final_away]
    )


def estimate_score(home_attack, away_attack, home_defense, away_defense, result):
    home_lambda = max(0.2, home_attack * 0.7 + away_defense * 0.3)
    away_lambda = max(0.2, away_attack * 0.7 + home_defense * 0.3)

    home_goals = np.random.poisson(home_lambda)
    away_goals = np.random.poisson(away_lambda)

    if result == "home_win" and home_goals <= away_goals:
        home_goals = away_goals + 1

    elif result == "away_win" and away_goals <= home_goals:
        away_goals = home_goals + 1

    elif result == "draw":
        avg_goals = round((home_goals + away_goals) / 2)
        home_goals = avg_goals
        away_goals = avg_goals

    home_goals = min(home_goals, 5)
    away_goals = min(away_goals, 5)

    return home_goals, away_goals


def apply_result(table, home_team, away_team, home_score, away_score):
    if home_score > away_score:
        home_points, away_points = 3, 0
    elif home_score < away_score:
        home_points, away_points = 0, 3
    else:
        home_points, away_points = 1, 1

    table.loc[table["team"] == home_team, "points"] += home_points
    table.loc[table["team"] == away_team, "points"] += away_points

    table.loc[table["team"] == home_team, "gd"] += home_score - away_score
    table.loc[table["team"] == away_team, "gd"] += away_score - home_score

    table.loc[table["team"] == home_team, "goals_for"] += home_score
    table.loc[table["team"] == away_team, "goals_for"] += away_score

    table.loc[table["team"] == home_team, "goals_against"] += away_score
    table.loc[table["team"] == away_team, "goals_against"] += home_score

    table.loc[table["team"] == home_team, "played"] += 1
    table.loc[table["team"] == away_team, "played"] += 1

    return table


def apply_finished_results(table, fixtures, finished_results):
    if finished_results.empty:
        return table, fixtures

    finished_results = finished_results.copy()

    finished_results["home_team"] = finished_results["home_team"].replace(TEAM_NAME_MAP)
    finished_results["away_team"] = finished_results["away_team"].replace(TEAM_NAME_MAP)

    for _, row in finished_results.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        if home not in table["team"].values or away not in table["team"].values:
            continue

        table = apply_result(
            table,
            home,
            away,
            int(row["home_score"]),
            int(row["away_score"])
        )

        fixtures = fixtures[
            ~(
                (fixtures["home_team"] == home)
                & (fixtures["away_team"] == away)
            )
        ]

    fixtures = fixtures.reset_index(drop=True)

    return table, fixtures


def run_monte_carlo_simulation(standings, fixtures, europe_epl_teams, num_simulations):
    champion_counts = {}
    top5_counts = {}
    sixth_counts = {}
    relegation_counts = {}
    total_points = {}

    for i in range(num_simulations):
        table = standings.copy()

        for _, row in fixtures.iterrows():
            home = row["home_team"]
            away = row["away_team"]

            (
                model_home,
                model_draw,
                model_away,
                market_home,
                market_draw,
                market_away,
                final_home,
                final_draw,
                final_away
            ) = predict_match_probabilities(
                table,
                row,
                europe_epl_teams
            )

            result = sample_match_result(
                final_home,
                final_draw,
                final_away
            )

            home_row = table[table["team"] == home].iloc[0]
            away_row = table[table["team"] == away].iloc[0]

            home_score, away_score = estimate_score(
                home_row["attack"],
                away_row["attack"],
                home_row["defense"],
                away_row["defense"],
                result
            )

            table = apply_result(
                table,
                home,
                away,
                home_score,
                away_score
            )

        final_table = table.sort_values(
            by=["points", "gd", "goals_for"],
            ascending=[False, False, False]
        ).reset_index(drop=True)

        for _, team_row in final_table.iterrows():
            team = team_row["team"]
            points = team_row["points"]
            total_points[team] = total_points.get(team, 0) + points

        champion = final_table.iloc[0]["team"]
        champion_counts[champion] = champion_counts.get(champion, 0) + 1

        for team in final_table.iloc[:5]["team"]:
            top5_counts[team] = top5_counts.get(team, 0) + 1

        sixth = final_table.iloc[5]["team"]
        sixth_counts[sixth] = sixth_counts.get(sixth, 0) + 1

        for team in final_table.iloc[-3:]["team"]:
            relegation_counts[team] = relegation_counts.get(team, 0) + 1

    results = []

    for team in standings["team"].tolist():
        results.append({
            "team": team,
            "avg_points": round(total_points.get(team, 0) / num_simulations, 2),
            "champion_probability": champion_counts.get(team, 0) / num_simulations,
            "top5_probability": top5_counts.get(team, 0) / num_simulations,
            "sixth_probability": sixth_counts.get(team, 0) / num_simulations,
            "relegation_probability": relegation_counts.get(team, 0) / num_simulations
        })

    return pd.DataFrame(results)

try:
    standings, epl_fixtures, rescheduled, europe = load_data()
    standings = clean_standings(standings)

    fixtures = pd.concat(
        [epl_fixtures, rescheduled],
        ignore_index=True
    )

    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = fixtures.sort_values(by="date").reset_index(drop=True)

    finished_results = pd.DataFrame()
    live_scores = pd.DataFrame()

    if use_live_results and football_data_api_key:
        try:
            finished_results = fetch_finished_epl_results(football_data_api_key)

            if not finished_results.empty:
                st.sidebar.success("Live results loaded.")
            else:
                st.sidebar.warning("Live results unavailable. Falling back to model only.")

        except Exception:
            st.sidebar.warning("Live results API failed. Falling back to model only.")
            finished_results = pd.DataFrame()

    if use_live_score and football_data_api_key:
        try:
            live_scores = fetch_live_epl_scores(football_data_api_key)

            if not live_scores.empty:
                st.sidebar.success("Live scores loaded.")
            else:
                st.sidebar.info("No live EPL matches right now. Model still running.")

        except Exception:
            st.sidebar.warning("Live scores API failed. Model still running.")
            live_scores = pd.DataFrame()

    europe_epl_teams = get_european_epl_teams(standings, europe)

    simulation_table = standings.copy()
    predicted_matches = []

    simulation_table, fixtures = apply_finished_results(
        simulation_table,
        fixtures,
        finished_results
    )

    for _, row in fixtures.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        (
            model_home,
            model_draw,
            model_away,
            market_home,
            market_draw,
            market_away,
            final_home,
            final_draw,
            final_away
        ) = predict_match_probabilities(
            simulation_table,
            row,
            europe_epl_teams
        )

        result = sample_match_result(
            final_home,
            final_draw,
            final_away
        )

        home_row = simulation_table[simulation_table["team"] == home].iloc[0]
        away_row = simulation_table[simulation_table["team"] == away].iloc[0]

        home_score, away_score = estimate_score(
            home_row["attack"],
            away_row["attack"],
            home_row["defense"],
            away_row["defense"],
            result
        )

        predicted_matches.append({
            "date": row["date"].date() if pd.notna(row["date"]) else "",
            "competition": row["competition"],
            "home_team": home,
            "away_team": away,
            "predicted_score": f"{home_score}-{away_score}",

            "home_win_prob": round(final_home, 3),
            "draw_prob": round(final_draw, 3),
            "away_win_prob": round(final_away, 3),

            "model_home_prob": round(model_home, 3),
            "model_draw_prob": round(model_draw, 3),
            "model_away_prob": round(model_away, 3),

            "market_home_prob": round(market_home, 3) if market_home is not None else None,
            "market_draw_prob": round(market_draw, 3) if market_draw is not None else None,
            "market_away_prob": round(market_away, 3) if market_away is not None else None,

            "home_odds": row.get("home_odds"),
            "draw_odds": row.get("draw_odds"),
            "away_odds": row.get("away_odds"),

            "odds_used": market_home is not None
        })

        simulation_table = apply_result(
            simulation_table,
            home,
            away,
            home_score,
            away_score
        )

    final_table = simulation_table.sort_values(
        by=["points", "gd", "goals_for"],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    final_table.index = final_table.index + 1

    champion = final_table.iloc[0]["team"]
    top_5 = final_table.iloc[:5]["team"].tolist()
    sixth = final_table.iloc[5]["team"]
    relegation = final_table.iloc[-3:]["team"].tolist()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Champion", champion)
    col2.metric("Top 5", ", ".join(top_5))
    col3.metric("6th Place", sixth)
    col4.metric("Relegation", ", ".join(relegation))

    st.divider()

    if not live_scores.empty:
        live_scores_display = live_scores.copy()
        live_scores_display["home_team"] = live_scores_display["home_team"].replace(TEAM_NAME_MAP)
        live_scores_display["away_team"] = live_scores_display["away_team"].replace(TEAM_NAME_MAP)

        st.subheader("🔴 Live EPL Scores")
        st.dataframe(
            live_scores_display[[
                "status",
                "home_team",
                "home_score",
                "away_score",
                "away_team"
            ]],
            use_container_width=True
        )

    st.subheader("🌍 European Fatigue Teams")
    st.write(sorted(list(europe_epl_teams)))

    st.subheader("📊 Predicted Final Table")
    st.dataframe(
        final_table[[
            "team",
            "played",
            "points",
            "gd",
            "goals_for",
            "goals_against",
            "strength"
        ]],
        use_container_width=True
    )

    st.subheader("🗓️ Predicted Matches")
    predicted_df = pd.DataFrame(predicted_matches)
    st.dataframe(predicted_df, use_container_width=True)

    st.subheader("🏆 Monte Carlo Simulation Results")

    with st.spinner("Running Monte Carlo simulation..."):
        simulation_results = run_monte_carlo_simulation(
            simulation_table.copy(),
            fixtures,
            europe_epl_teams,
            num_simulations
        )

    simulation_results = simulation_results.sort_values(
        by="avg_points",
        ascending=False
    )

    st.dataframe(
        simulation_results,
        use_container_width=True
    )

    chart_data = simulation_results[
        simulation_results["champion_probability"] > 0
    ]

    st.bar_chart(
        chart_data.set_index("team")["champion_probability"]
    )

    st.caption(
        "Disclaimer: This project is for educational and analytical purposes only. "
        "It does not provide betting advice."
    )

except FileNotFoundError as e:
    st.error("CSV file not found.")
    st.write(e)

except IndexError:
    st.error("Team name mismatch. Please check that team names are exactly the same in all CSV files.")

except Exception as e:
    st.error("Something went wrong.")
    st.write(e)
