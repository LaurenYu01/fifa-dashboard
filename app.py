import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import os


@st.cache_resource(show_spinner=False)
def get_bq_client():
    # 1) Local dev: prefer ADC via env var (we already set GOOGLE_APPLICATION_CREDENTIALS)
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        return bigquery.Client(project="fifa-ml-20250822", location="US")

    # 2) Streamlit Cloud: try secrets (might not exist locally)
    try:
        svc = st.secrets["gcp_service_account"]  # will raise if secrets.toml is missing
        creds = service_account.Credentials.from_service_account_info(svc)
        return bigquery.Client(
            credentials=creds,
            project=svc["project_id"],
            location="US",
        )
    except Exception:
        pass

    # 3) Fallback: default ADC (e.g., gcloud auth login)
    return bigquery.Client(project="fifa-ml-20250822", location="US")

client = get_bq_client()

PROJECT = "fifa-ml-20250822"
DATASET = "fifa_ds"
TABLE = f"`{PROJECT}.{DATASET}.players`"
MODEL = f"`{PROJECT}.{DATASET}.player_value_model`"

st.set_page_config(page_title="FIFA Market Insights", layout="wide")
st.title("⚽ FIFA Player Market Insights Dashboard")
st.caption("Prediction powered by BigQuery ML • Data: FIFA 21 (Kaggle)")

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

if st.sidebar.button("Predict value"):
    sql = f"""
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
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
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
    )
    with st.spinner("Running ML.PREDICT on BigQuery..."):
        df_pred = client.query(sql, job_config=job_config).to_dataframe()
    pred_val = float(df_pred.iloc[0]["predicted_value_eur"])
    st.metric("Predicted Market Value (€)", f"{pred_val:,.0f}")

st.markdown("---")
st.subheader("Exploratory Analysis (live from BigQuery)")

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
ORDER BY player_count DESC
"""
df_buckets = client.query(sql_buckets).to_dataframe()
col1, col2 = st.columns(2)
with col1:
    st.caption("Player count by value bucket")
    st.bar_chart(df_buckets.set_index("value_bucket"))

sql_nat = f"""
SELECT
  nationality,
  AVG(value_eur) AS avg_value,
  COUNT(*) AS num_players
FROM {TABLE}
WHERE value_eur IS NOT NULL
GROUP BY nationality
HAVING COUNT(*) >= 30
ORDER BY avg_value DESC
LIMIT 10
"""
df_nat = client.query(sql_nat).to_dataframe()
with col2:
    st.caption("Top 10 nationalities by average value (€)")
    st.bar_chart(df_nat.set_index("nationality")["avg_value"])

sql_scatter = f"""
SELECT overall, value_eur
FROM {TABLE}
WHERE value_eur IS NOT NULL
ORDER BY RAND()
LIMIT 2000
"""
df_scatter = client.query(sql_scatter).to_dataframe()
st.caption("Overall vs Value (sampled)")
st.scatter_chart(df_scatter, x="overall", y="value_eur")

st.markdown("---")
st.caption("Model: BigQuery ML linear_reg • Table: fifa-ml-20250822.fifa_ds.players")
