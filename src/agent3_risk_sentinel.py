import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq                  # to connect to Groq's LLM API
from langchain.prompts import PromptTemplate         # to create a structured prompt template
from reportlab.lib.pagesizes import A4               # standard A4 page size for the PDF
from reportlab.lib import colors                     # color constants for styling
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer  # PDF building blocks
from reportlab.lib.styles import getSampleStyleSheet  # default text styles for the PDF

# Load the GROQ_API_KEY from our .env file
load_dotenv()

# Folder where the PDF report will be saved
REPORTS_FOLDER = "data/processed/reports"

def analyze_risk(question, answer):
    """
    Takes the question and Agent 1's answer.
    Asks the LLM to identify financial risk signals and assign a severity level.
    Returns structured risk analysis as text.
    """

    prompt_template = PromptTemplate(
        input_variables=["question", "answer"],
        template="""
You are a financial risk analyst working for a central bank. 
Analyze the following answer for financial risk signals.

Question: {question}
Answer: {answer}

Identify any risk signals present (e.g. inflation risk, credit risk, liquidity risk, interest rate risk, geopolitical risk).
Assign an overall severity level: Low, Medium, or High.

Respond in this exact format:
SEVERITY: Low or Medium or High
RISKS IDENTIFIED: (list each risk on a new line starting with -)
SUMMARY: (two sentences summarizing the overall risk picture)
"""
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
    )

    prompt = prompt_template.format(question=question, answer=answer)
    response = llm.invoke(prompt)
    return response.content


def generate_pdf_report(question, answer, critic_verdict, risk_analysis):
    """
    Takes all outputs from the 3 agents and generates a structured PDF report.
    Saves it to data/processed/reports/
    """
    os.makedirs(REPORTS_FOLDER, exist_ok=True)
    report_path = os.path.join(REPORTS_FOLDER, "finsight_report.pdf")

    # Set up the PDF document
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []   # list of elements to add to the PDF, in order

    # Title
    story.append(Paragraph("FinSight Risk Intelligence Report", styles["Title"]))
    story.append(Spacer(1, 12))  # adds vertical space

    # Question section
    story.append(Paragraph("Question", styles["Heading2"]))
    story.append(Paragraph(question, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Answer section
    story.append(Paragraph("Answer (Agent 1 — Retriever)", styles["Heading2"]))
    story.append(Paragraph(answer, styles["Normal"]))
    story.append(Spacer(1, 12))

    # Critic verdict section
    story.append(Paragraph("Quality Check (Agent 2 — Critic)", styles["Heading2"]))
    story.append(Paragraph(critic_verdict.replace("\n", "<br/>"), styles["Normal"]))
    story.append(Spacer(1, 12))

    # Risk analysis section
    story.append(Paragraph("Risk Analysis (Agent 3 — Risk Sentinel)", styles["Heading2"]))
    story.append(Paragraph(risk_analysis.replace("\n", "<br/>"), styles["Normal"]))
    story.append(Spacer(1, 12))

    # Build and save the PDF
    doc.build(story)
    print(f"PDF report saved to {report_path}")
    return report_path


if __name__ == "__main__":
    # Test Agent 3 with hardcoded inputs
    test_question = "What was the ECB's main monetary policy decision in 2023?"
    test_answer = "The ECB raised interest rates by 200 basis points in 2023 to bring inflation back to 2%, with the deposit facility rate reaching 4% by September 2023."
    test_critic_verdict = "VERDICT: PASS\nREASON: The answer is factually supported by the source documents."

    print("Analyzing risk...")
    risk_analysis = analyze_risk(test_question, test_answer)
    print(f"\nRisk Analysis:\n{risk_analysis}")

    print("\nGenerating PDF report...")
    generate_pdf_report(test_question, test_answer, test_critic_verdict, risk_analysis)