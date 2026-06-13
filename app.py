import sys
import os

# Add the src folder to the path so we can import our agents
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from agent1_retriever import get_retriever_chain, load_and_split_pdfs, build_vectorstore
from agent2_critic import evaluate_answer
from agent3_risk_sentinel import analyze_risk, generate_pdf_report

# Paths
FAISS_INDEX_PATH = "data/processed/faiss_index"
PDF_FOLDER = "data/raw"

def run_pipeline(question):
    """
    Runs the full 3-agent pipeline for a given question:
    1. Agent 1 retrieves an answer from the ECB reports
    2. Agent 2 evaluates the answer quality
    3. Agent 3 analyzes the risk and generates a PDF report
    """

    print("=" * 60)
    print("FINSIGHT MULTI-AGENT RAG PIPELINE")
    print("=" * 60)

    # --- Agent 1: Retriever ---
    print("\n[Agent 1 - Retriever] Searching ECB reports...")

    # Build FAISS index if it doesn't exist yet
    if not os.path.exists(FAISS_INDEX_PATH):
        print("Building vector store from PDFs...")
        chunks = load_and_split_pdfs(PDF_FOLDER)
        build_vectorstore(chunks)

    chain = get_retriever_chain()
    result = chain.invoke({"query": question})
    answer = result["result"]
    source_documents = result["source_documents"]

    print(f"Question: {question}")
    print(f"Answer: {answer}")

    # --- Agent 2: Critic ---
    print("\n[Agent 2 - Critic] Evaluating answer quality...")
    critic_verdict = evaluate_answer(question, answer, source_documents)
    print(f"Verdict:\n{critic_verdict}")

    # --- Agent 3: Risk Sentinel ---
    print("\n[Agent 3 - Risk Sentinel] Analyzing financial risks...")
    risk_analysis = analyze_risk(question, answer)
    print(f"Risk Analysis:\n{risk_analysis}")

    # --- Generate PDF Report ---
    print("\nGenerating PDF report...")
    report_path = generate_pdf_report(question, answer, critic_verdict, risk_analysis)

    print("\n" + "=" * 60)
    print(f"Pipeline complete! Report saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    question = "What are the main financial stability risks identified by the ECB in 2024?"
    run_pipeline(question)