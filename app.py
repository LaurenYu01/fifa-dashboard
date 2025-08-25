import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

# ---------------------------
# BigQuery client (from secrets or fallback)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_bq_client_and_project():
    """Return (client, project_id) using Streamlit secrets when present."""
    if "gcp_service_account" in st.secrets:
        info = st.secrets["gcp_service_account"]
        creds = service_account.Credentials.from_service_account_info(info)
        proj = info.get("project_id")
        client = bigquery.Client(credentials=creds, project=proj, location="US")
        return client, proj
    # local fallback
    proj = "big-query-fifa"
    client = bigquery.Client(project=proj, location="US")
    return client, proj


client, PROJECT = get_bq_client_and_project()
DATASET = "fifa_ds"

@st.cache_data(show_spinner=False, ttl=600)
def resolve_table_ids(project: str, dataset: str):
    """
    Detect which table exists: 'players' or 'player'.
    Returns:
      (table_backticked, table_plain_name)
      e.g. ("`proj.ds.players`", "proj.ds.players")
    """
    dataset_ref = f"{project}.{dataset}"
    found = set(t.table_id.lower() for t in client.list_tables(dataset_ref))
    chosen = None
    for cand in ("players", "player"):
        if cand in found:
            chosen = cand
            break
    if not chosen:
        raise RuntimeError(
            f"No table named 'players' or 'player' in {dataset_ref}. "
            f"Found: {sorted(found)}"
        )
    plain = f"{project}.{dataset}.{chosen}"
    ticked = f"`{plain}`"
    return ticked, plain, chosen  # backticked, plain, short name

TABLE, TABLE_PLAIN, TABLE_SHORT = resolve_table_ids(PROJECT, DATASET)

# model stays the same dataset
MODEL = f"`{PROJECT}.{DATASET}.player_value_model`"

# ---------------------------
# Helpers
# ---------------------------
def run_df(sql: str, params: list | None = None) -> pd.DataFrame:
    job_config = bigquery.QueryJobConfig(query_parameters=params or [])
    return client.query(sql, job_config=job_config, location="US").to_dataframe()

@st.cache_data(show_spinner=False, ttl=300)
def list_columns() -> set[str]:
    tbl = client.get_table(TABLE_PLAIN)  # use detected table
    return {c.name.lower() for c in tbl.schema}

def first_existing(candidates: list[str], available: set[str]) -> str | None:
    for c in candidates:
        if c.lower() in available:
            return c
    return None

def predict_value(
    age:int, overall:int, potential:int, skill_moves:int, wage_eur:int,
    international_reputation:int, weak_foot:int, pace:int, shooting:int,
    passing:int, dribbling:int, defending:int, physic:int
) -> float:
    sql_predict = f"""
    SELECT *
    FROM ML.PREDICT(
      MODEL {MODEL},
      (
        SELECT
          @age AS age,
          @overall AS overall,
          @potential AS potential,
          @skill_moves AS skill_moves,
          @wage_eur AS wage_eur,
          @international_reputation AS international_reputation,
          @weak_foot AS weak_foot,
          @pace AS pace,
          @shooting AS shooting,
          @passing AS passing,
          @dribbling AS dribbling,
          @defending AS defending,
          @physic AS physic
      )
    )
    """
    params = [
        bigquery.ScalarQueryParameter("age", "INT64", age),
        bigquery.ScalarQueryParameter("overall", "INT64", overall),
        bigquery.ScalarQueryParameter("potential", "INT64", potential),
        bigquery.ScalarQueryParameter("skill_moves", "INT64", skill_moves),
        bigquery.ScalarQueryParameter("wage_eur", "INT64", wage_eur),
        bigquery.ScalarQueryParameter("international_reputation", "INT64", international_reputation),
        bigquery.ScalarQueryParameter("weak_foot", "INT64", weak_foot),
        bigquery.ScalarQueryParameter("pace", "INT64", pace),
        bigquery.ScalarQueryParameter("shooting", "INT64", shooting),
        bigquery.ScalarQueryParameter("passing", "INT64", passing),
        bigquery.ScalarQueryParameter("dribbling", "INT64", dribbling),
        bigquery.ScalarQueryParameter("defending", "INT64", defending),
        bigquery.ScalarQueryParameter("physic", "INT64", physic),
    ]
    df_pred = run_df(sql_predict, params)
    return float(df_pred.iloc[0]["predicted_value_eur"])

# ---------------------------
# Page layout & header
# ---------------------------
st.set_page_config(page_title="FIFA Market Insights", layout="wide")
st.title("⚽ FIFA Player Market Insights Dashboard")
st.caption("Prediction powered by BigQuery ML • Data: FIFA 21 (Kaggle)")

# ---------------------------
# Sidebar — prediction inputs
# ---------------------------
st.sidebar.header("Predict Player Market Value (€)")
age = st.sidebar.number_input("Age", 15, 45, 24)
overall = st.sidebar.number_input("Overall", 40, 95, 86)
potential = st.sidebar.number_input("Potential", 40, 95, 92)
skill_moves = st.sidebar.number_input("Skill Moves", 1, 5, 4)
wage_eur = st.sidebar.number_input("Wage (EUR / week)", 0, 1_000_000, 120_000, step=5_000)
international_reputation = st.sidebar.number_input("International Reputation", 1, 5, 3)
weak_foot = st.sidebar.number_input("Weak Foot", 1, 5, 4)
pace = st.sidebar.number_input("Pace", 1, 99, 85)
shooting = st.sidebar.number_input("Shooting", 1, 99, 80)
passing = st.sidebar.number_input("Passing", 1, 99, 82)
dribbling = st.sidebar.number_input("Dribbling", 1, 99, 86)
defending = st.sidebar.number_input("Defending", 1, 99, 60)
physic = st.sidebar.number_input("Physic", 1, 99, 78)

# First render: compute once so the metric shows immediately
if "pred_val" not in st.session_state:
    try:
        st.session_state.pred_val = predict_value(
            age, overall, potential, skill_moves, wage_eur,
            international_reputation, weak_foot, pace, shooting,
            passing, dribbling, defending, physic
        )
    except Exception as e:
        st.session_state.pred_val = None
        st.warning(f"Prediction not available: {e}")

if st.sidebar.button("Predict value"):
    try:
        st.session_state.pred_val = predict_value(
            age, overall, potential, skill_moves, wage_eur,
            international_reputation, weak_foot, pace, shooting,
            passing, dribbling, defending, physic
        )
    except Exception as e:
        st.session_state.pred_val = None
        st.warning(f"Prediction failed: {e}")

# Top metric
if st.session_state.pred_val is not None:
    st.metric("Predicted Market Value (€)", f"{st.session_state.pred_val:,.0f}")

st.markdown("---")
st.subheader("Exploratory Analysis (live from BigQuery)")

available_cols = list_columns()

# 1) Value buckets
sql_buckets = f"""
WITH buckets AS (
  SELECT
    CASE
      WHEN value_eur < 1000000 THEN '<1M'
      WHEN value_eur < 5000000 THEN '1M–5M'
      WHEN value_eur < 10000000 THEN '5M–10M'
      WHEN value_eur < 20000000 THEN '10M–20M'
      WHEN value_eur < 50000000 THEN '20M–50M'
      ELSE '≥50M'
    END AS value_bucket
  FROM {TABLE}
  WHERE value_eur IS NOT NULL
)
SELECT value_bucket, COUNT(*) AS player_count
FROM buckets
GROUP BY value_bucket
ORDER BY
  CASE value_bucket
    WHEN '<1M' THEN 1
    WHEN '1M–5M' THEN 2
    WHEN '5M–10M' THEN 3
    WHEN '10M–20M' THEN 4
    WHEN '20M–50M' THEN 5
    ELSE 6
  END
"""
df_buckets = run_df(sql_buckets)

# 2) Top-10 group (nationality -> club/league -> age group)
dim = first_existing(["nationality_name", "nationality"], available_cols)
label = None
sql_top = None

if dim:
    label = "Top 10 nationalities by average value (€)"
    sql_top = f"""
    SELECT
      {dim} AS grp,
      AVG(value_eur) AS avg_value,
      COUNT(*) AS num_players
    FROM {TABLE}
    WHERE value_eur IS NOT NULL AND {dim} IS NOT NULL
    GROUP BY grp
    HAVING COUNT(*) >= 30
    ORDER BY avg_value DESC
    LIMIT 10
    """
else:
    dim = first_existing(["club_name", "league_name"], available_cols)
    if dim:
        label = f"Top 10 by average value (€) — grouped by {dim}"
        sql_top = f"""
        SELECT
          {dim} AS grp,
          AVG(value_eur) AS avg_value,
          COUNT(*) AS num_players
        FROM {TABLE}
        WHERE value_eur IS NOT NULL AND {dim} IS NOT NULL
        GROUP BY grp
        HAVING COUNT(*) >= 20
        ORDER BY avg_value DESC
        LIMIT 10
        """

if not sql_top:
    label = "Top age groups by average value (€)"
    sql_top = f"""
    WITH base AS (
      SELECT
        CASE
          WHEN age IS NULL THEN 'Unknown'
          WHEN age < 20 THEN '<20'
          WHEN age BETWEEN 20 AND 22 THEN '20–22'
          WHEN age BETWEEN 23 AND 25 THEN '23–25'
          WHEN age BETWEEN 26 AND 28 THEN '26–28'
          WHEN age BETWEEN 29 AND 31 THEN '29–31'
          ELSE '32+'
        END AS age_group,
        value_eur
      FROM {TABLE}
      WHERE value_eur IS NOT NULL
    )
    SELECT age_group AS grp,
           AVG(value_eur) AS avg_value,
           COUNT(*) AS num_players
    FROM base
    GROUP BY age_group
    ORDER BY
      CASE age_group
        WHEN '<20' THEN 1
        WHEN '20–22' THEN 2
        WHEN '23–25' THEN 3
        WHEN '26–28' THEN 4
        WHEN '29–31' THEN 5
        ELSE 6
      END
    LIMIT 10
    """

df_top = run_df(sql_top)

# 3) Scatter (overall preferred, else potential)
x_col = "overall" if "overall" in available_cols else ("potential" if "potential" in available_cols else None)
df_scatter = pd.DataFrame()
if x_col:
    sql_scatter = f"""
    SELECT {x_col} AS x, value_eur
    FROM {TABLE}
    WHERE value_eur IS NOT NULL AND {x_col} IS NOT NULL
    ORDER BY RAND()
    LIMIT 2000
    """
    df_scatter = run_df(sql_scatter)

# --- Layout ---
col1, col2 = st.columns(2, gap="large")
with col1:
    st.caption("Player count by value bucket")
    st.bar_chart(df_buckets.set_index("value_bucket"))

with col2:
    if df_top.empty:
        st.info(
            "This table has neither 'nationality' nor 'nationality_name' and also lacks "
            "fallback columns ('club_name' / 'league_name'). — chart skipped."
        )
    else:
        st.caption(label)
        st.bar_chart(df_top.set_index("grp")["avg_value"])

st.caption("Overall vs Value (sampled)" if x_col == "overall" else "Potential vs Value (sampled)")
if df_scatter.empty:
    st.info("Missing both 'overall' and 'potential' — scatter skipped.")
else:
    st.scatter_chart(df_scatter.rename(columns={"x": x_col}), x=x_col, y="value_eur")

st.markdown("---")
st.caption(f"Model: BigQuery ML linear_reg • Table: {TABLE_PLAIN}")
