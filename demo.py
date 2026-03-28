import streamlit as st
import pandas as pd
import os
import datetime
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import transformers
transformers.utils.logging.set_verbosity_error()

# ==============================
# 1. ENV SETUP
# ==============================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATASET_PATH = os.getenv("DATASET_PATH", "diabetes_prediction_dataset.csv")
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "faiss_diabetes_index")
RESEARCH_LOG_FILE = "persistent_research_audit.csv"
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")

st.set_page_config(page_title="Clinical AI Research System", layout="wide")

# ==============================
# 2. EMBEDDINGS
# ==============================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==============================
# 3. DATA LOADING
# ==============================
@st.cache_data
def load_and_split_data():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at {DATASET_PATH}")
        st.stop()
    
    df_full = pd.read_csv(DATASET_PATH)
    df_full = df_full.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(0.8 * len(df_full))
    return df_full.iloc[:split_idx], df_full.iloc[split_idx:]

train_df, test_df = load_and_split_data()

# ==============================
# 4. PERSISTENT VECTOR DB
# ==============================
@st.cache_resource
def load_or_create_vector_db():
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    st.warning("Indexing training set for the first time...")
    sample_df = train_df.sample(min(10000, len(train_df)), random_state=42)
    docs = [
        Document(
            page_content=f"Patient: {r['gender']}, Age {r['age']}, BMI {r['bmi']}, HbA1c {r['HbA1c_level']}, Glucose {r['blood_glucose_level']}, Diabetes: {'Positive' if r['diabetes'] == 1 else 'Negative'}",
            metadata=r.to_dict()
        ) for _, r in sample_df.iterrows()
    ]
    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)
    return vector_db

vector_db = load_or_create_vector_db()

# ==============================
# 5. LLM + AGENT
# ==============================
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in .env file")
    st.stop()

llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME)
agent = create_pandas_dataframe_agent(
    llm, train_df, verbose=True, agent_type="zero-shot-react-description", 
    allow_dangerous_code=True, return_intermediate_steps=True, max_iterations=10
)

# ==============================
# 6. LOGGING (BATCH ONLY)
# ==============================
def log_research_event(query, response_data, latency, context):
    executed_code = "No code executed"
    if "intermediate_steps" in response_data and response_data["intermediate_steps"]:
        try:
            executed_code = response_data["intermediate_steps"][0][0].tool_input
        except: pass

    log_entry = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Query": query,
        "Response": response_data.get("output", ""),
        "Executed_Code": executed_code,
        "Latency_sec": round(latency, 3),
        "Model": MODEL_NAME
    }
    pd.DataFrame([log_entry]).to_csv(
        RESEARCH_LOG_FILE, mode="a", index=False, header=not os.path.exists(RESEARCH_LOG_FILE)
    )

# ==============================
# 7. UI
# ==============================
st.title("🔬 Clinical Agentic RAG System")
tab1, tab2 = st.tabs(["Interactive Query", "Batch Systematic Review"])

# --- TAB 1: INTERACTIVE (NO LOGGING) ---
with tab1:
    st.info("💡 Real-time analysis. Data here is NOT saved to the CSV.")
    query = st.chat_input("Enter clinical research question...")
    if query:
        with st.spinner("Analyzing..."):
            docs = vector_db.similarity_search(query, k=5)
            context = "\n".join([d.page_content for d in docs])
            response = agent.invoke({"input": f"Context:\n{context}\n\nTask: {query}"})
            st.subheader("📊 AI Analysis")
            st.write(response["output"])
            with st.expander("🔍 Logic & Context"):
                st.code(context[:500])

# --- TAB 2: BATCH (SAVES TO CSV) ---
with tab2:
    st.subheader("Batch Evaluation Control")
    st.write("Enter multiple queries below (one per line). Each will be logged to the CSV for analysis.")
    
    # 📝 TEXT AREA FOR BATCH INPUT
    default_queries = "Mean glucose levels for hypertension patients\nCorrelation between age and HbA1c\nPercentage of heart disease cases in smokers"
    batch_input = st.text_area("List of Research Queries:", value=default_queries, height=200)
    
    if st.button("🚀 Run Batch Analysis"):
        # Split text area by lines and remove empty lines
        queries_to_run = [q.strip() for q in batch_input.split('\n') if q.strip()]
        
        if not queries_to_run:
            st.error("Please enter at least one query.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, q in enumerate(queries_to_run):
                status_text.text(f"Processing ({i+1}/{len(queries_to_run)}): {q}")
                start_t = time.time()
                
                # Retrieve & Process
                docs = vector_db.similarity_search(q, k=3)
                ctx = "\n".join([d.page_content for d in docs])
                res = agent.invoke({"input": f"Context: {ctx}\nTask: {q}"})
                
                # LOG TO CSV
                log_research_event(q, res, time.time() - start_t, ctx)
                
                progress_bar.progress((i + 1) / len(queries_to_run))
            
            status_text.text("✅ Batch Analysis Complete!")
            st.success(f"Processed {len(queries_to_run)} queries. Results saved to {RESEARCH_LOG_FILE}.")

# ==============================
# 8. SIDEBAR
# ==============================
with st.sidebar:
    st.header("📈 Manuscript Metrics")
    if os.path.exists(RESEARCH_LOG_FILE):
        logs = pd.read_csv(RESEARCH_LOG_FILE)
        st.metric("Batch Runs Logged", len(logs))
        st.download_button("Download CSV for Manuscript", logs.to_csv(index=False), "clinical_research_data.csv")