import sys
import os
import streamlit as st

# Add the src folder to the path so we can import our agents
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from agent1_retriever import get_retriever_chain, load_and_split_pdfs, build_vectorstore
from agent2_critic import evaluate_answer
from agent3_risk_sentinel import analyze_risk, generate_pdf_report

# Paths
FAISS_INDEX_PATH = "data/processed/faiss_index"
PDF_FOLDER = "data/raw"

# --- Page configuration ---
st.set_page_config(
    page_title="FinSight",
    page_icon="📊",
    layout="wide"
)

# --- Title and description ---
st.title("📊 FinSight — Multi-Agent RAG for Financial Intelligence")
st.markdown("Ask any question about ECB Annual Reports (2021–2025). The system retrieves answers, checks their quality, and analyzes financial risks.")

st.divider()

# --- Build FAISS index if needed (runs once) ---
@st.cache_resource  # caches the chain so it doesn't reload on every interaction
def load_chain():
    if not os.path.exists(FAISS_INDEX_PATH):
        with st.spinner("Building vector store from PDFs... (first time only)"):
            chunks = load_and_split_pdfs(PDF_FOLDER)
            build_vectorstore(chunks)
    return get_retriever_chain()

chain = load_chain()

# --- Question input ---
question = st.text_input("Enter your question:", placeholder="e.g. What were the main inflation risks in 2024?")

if st.button("Run Pipeline") and question:

    # --- Agent 1 ---
    with st.spinner("Agent 1 — Retrieving answer from ECB reports..."):
        result = chain.invoke({"query": question})
        answer = result["result"]
        source_documents = result["source_documents"]

    st.subheader("📄 Agent 1 — Retriever")
    st.write(answer)

    # --- Agent 2 ---
    with st.spinner("Agent 2 — Evaluating answer quality..."):
        critic_verdict = evaluate_answer(question, answer, source_documents)

    st.subheader("🔍 Agent 2 — Critic")
    if "PASS" in critic_verdict:
        st.success(critic_verdict)
    else:
        st.warning(critic_verdict)

    # --- Agent 3 ---
    with st.spinner("Agent 3 — Analyzing financial risks..."):
        risk_analysis = analyze_risk(question, answer)

    st.subheader("⚠️ Agent 3 — Risk Sentinel")
    if "High" in risk_analysis:
        st.error(risk_analysis)
    elif "Medium" in risk_analysis:
        st.warning(risk_analysis)
    else:
        st.success(risk_analysis)

    # --- PDF Report ---
    with st.spinner("Generating PDF report..."):
        report_path = generate_pdf_report(question, answer, critic_verdict, risk_analysis)

    st.subheader("📥 Download Report")
    with open(report_path, "rb") as f:
        st.download_button(
            label="Download PDF Report",
            data=f,
            file_name="finsight_report.pdf",
            mime="application/pdf"
        )