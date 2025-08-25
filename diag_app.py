# diag_app.py
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

st.set_page_config(
    page_title="Diagnostics: Streamlit ↔ BigQuery",
    page_icon="✅",
    layout="centered",
)

st.title("Diagnostics: Streamlit ↔ BigQuery")
st.caption("This page checks: secrets, BQ ping, table, model.")

# ---------- Settings ----------
# 你在 BigQuery 里创建的数据集/表/模型名称（保持与第5步一致）
DATASET = "fifa_ds"
TABLE = "players"
MODEL = "player_value_model"
BQ_LOCATION = "US"  # 你的数据集/模型都在 US

# ---------- Load secrets & create client ----------
ok_secrets = True
project_id = "N/A"
cred_email = "N/A"
client: bigquery.Client | None = None

try:
    creds_info = st.secrets["gcp_service_account"]
    project_id = creds_info.get("project_id", "N/A")
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    cred_email = credentials.service_account_email

    client = bigquery.Client(
        project=project_id,
        credentials=credentials,
    )
except Exception as e:
    ok_secrets = False
    st.error(f"❌ Secrets load failed: {e}")

st.write(f"**Client mode:** secrets.toml")
st.write(f"**Using project:** {project_id}")
st.write(f"**Credential email:** {cred_email if ok_secrets else 'N/A'}")

def run_query(client_: bigquery.Client, sql: str):
    """
    执行查询。注意：location 需要作为 client.query 的参数传入，
    不能放进 QueryJobConfig（否则会报 unknown property）。
    """
    job = client_.query(sql, location=BQ_LOCATION)
    return job.result().to_dataframe(create_bqstorage_client=False)

def fq(project: str, dataset: str, name: str) -> str:
    return f"`{project}.{dataset}.{name}`"

FQ_TABLE = fq(project_id, DATASET, TABLE)
FQ_MODEL = fq(project_id, DATASET, MODEL)

# ---------- 1) BigQuery ping ----------
st.subheader("1) BigQuery ping")
if ok_secrets and client:
    try:
        df = run_query(client, "SELECT 1 AS ok")
        st.success("✅ Query job succeeded.")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error("❌ Access or resource not found (check roles/IDs/region).")
        st.code(str(e))
else:
    st.info("Skipped because secrets not loaded.")

# ---------- 2) Read a row from table ----------
st.subheader("2) Read a row from table")
st.caption(f"{project_id}.{DATASET}.{TABLE}")
if ok_secrets and client:
    try:
        df = run_query(client, f"SELECT * FROM {FQ_TABLE} LIMIT 1")
        st.success("✅ Table read succeeded.")
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error("❌ Table read failed. Ensure dataset/table exists and Data Viewer is granted.")
        st.code(str(e))
else:
    st.info("Skipped because secrets not loaded.")

# ---------- 3) Probe model (ML.WEIGHTS) ----------
st.subheader("3) Probe model (ML.WEIGHTS)")
st.caption(f"{project_id}.{DATASET}.{MODEL}")
if ok_secrets and client:
    try:
        dfw = run_query(client, f"SELECT * FROM ML.WEIGHTS(MODEL {FQ_MODEL}) LIMIT 10")
        st.success("✅ Model probe succeeded.")
        st.dataframe(dfw, use_container_width=True)
    except Exception as e:
        st.error("❌ Model probe failed. Check model name and location (US). Also grant Data Viewer/Model Viewer.")
        st.code(str(e))
else:
    st.info("Skipped because secrets not loaded.")
