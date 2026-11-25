import os
import sys

# Add the src folder to the path so we can import our agents
sys.path.append(os.path.dirname(__file__))

from typing import TypedDict, List              # to define the structure of our shared state
from langgraph.graph import StateGraph, END     # core LangGraph tools

from agent1_retriever import get_retriever_chain, load_and_split_pdfs, build_vectorstore
from agent2_critic import evaluate_answer
from agent3_risk_sentinel import analyze_risk, generate_pdf_report

# Paths
FAISS_INDEX_PATH = "data/processed/faiss_index"
PDF_FOLDER = "data/raw"

# --- State definition ---
# This is the shared "memory" that gets passed between all nodes in the graph
# Each agent reads from it and writes its results back into it
class FinSightState(TypedDict):
    question: str           # the user's question (set at the start)
    answer: str             # filled by Agent 1
    source_documents: List  # filled by Agent 1 (the chunks used to answer)
    critic_verdict: str     # filled by Agent 2
    risk_analysis: str      # filled by Agent 3
    report_path: str        # filled by Agent 3 (path to the generated PDF)


# --- Node 1: Retriever ---
def retriever_node(state: FinSightState) -> FinSightState:
    """
    Agent 1 — reads the question from state, retrieves an answer from the ECB reports,
    and writes the answer and source documents back into state.
    """
    print("[Agent 1 - Retriever] Searching ECB reports...")

    # Build FAISS index if it doesn't exist yet
    if not os.path.exists(FAISS_INDEX_PATH):
        print("Building vector store from PDFs...")
        chunks = load_and_split_pdfs(PDF_FOLDER)
        build_vectorstore(chunks)

    chain = get_retriever_chain()
    result = chain.invoke({"query": state["question"]})

    # Write results into the shared state
    state["answer"] = result["result"]
    state["source_documents"] = result["source_documents"]
    return state


# --- Node 2: Critic ---
def critic_node(state: FinSightState) -> FinSightState:
    """
    Agent 2 — reads the question, answer, and source documents from state,
    evaluates the answer quality, and writes the verdict back into state.
    """
    print("[Agent 2 - Critic] Evaluating answer quality...")

    verdict = evaluate_answer(
        state["question"],
        state["answer"],
        state["source_documents"]
    )

    # Write verdict into the shared state
    state["critic_verdict"] = verdict
    return state


# --- Node 3: Risk Sentinel ---
def risk_node(state: FinSightState) -> FinSightState:
    """
    Agent 3 — reads the question and answer from state,
    analyzes financial risks, generates a PDF report,
    and writes the risk analysis and report path back into state.
    """
    print("[Agent 3 - Risk Sentinel] Analyzing financial risks...")

    risk_analysis = analyze_risk(state["question"], state["answer"])
    report_path = generate_pdf_report(
        state["question"],
        state["answer"],
        state["critic_verdict"],
        risk_analysis
    )

    # Write results into the shared state
    state["risk_analysis"] = risk_analysis
    state["report_path"] = report_path
    return state


# --- Build the LangGraph pipeline ---
def build_pipeline():
    """
    Connects the 3 agent nodes into a LangGraph pipeline.
    The graph flows: START → retriever → critic → risk → END
    """
    # Create the graph using our shared state structure
    graph = StateGraph(FinSightState)

    # Add each agent as a node
    graph.add_node("retriever", retriever_node)
    graph.add_node("critic", critic_node)
    graph.add_node("risk", risk_node)

    # Define the edges (the flow between nodes)
    graph.set_entry_point("retriever")       # start here
    graph.add_edge("retriever", "critic")    # retriever → critic
    graph.add_edge("critic", "risk")         # critic → risk
    graph.add_edge("risk", END)              # risk → end

    # Compile and return the runnable pipeline
    return graph.compile()


# --- Test the pipeline ---
if __name__ == "__main__":
    pipeline = build_pipeline()

    # Define the initial state — only the question is set at the start
    initial_state = {
        "question": "What were the main financial risks identified by the ECB in 2023?",
        "answer": "",
        "source_documents": [],
        "critic_verdict": "",
        "risk_analysis": "",
        "report_path": ""
    }

    print("=" * 60)
    print("FINSIGHT LANGGRAPH PIPELINE")
    print("=" * 60)

    # Run the pipeline
    final_state = pipeline.invoke(initial_state)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nAnswer:\n{final_state['answer']}")
    print(f"\nCritic Verdict:\n{final_state['critic_verdict']}")
    print(f"\nRisk Analysis:\n{final_state['risk_analysis']}")
    print(f"\nReport saved to: {final_state['report_path']}")