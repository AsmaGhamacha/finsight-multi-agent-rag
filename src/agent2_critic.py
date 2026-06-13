import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq                  # to connect to Groq's LLM API
from langchain.prompts import PromptTemplate         # to create a structured prompt template

# Load the GROQ_API_KEY from our .env file
load_dotenv()

def evaluate_answer(question, answer, source_documents):
    """
    Takes the question, Agent 1's answer, and the source chunks used to generate it.
    Asks the LLM to evaluate whether the answer is grounded in the sources.
    Returns a verdict: PASS or FAIL with a reason.
    """

    # Combine all source chunks into one block of text for the LLM to review
    sources_text = "\n\n".join([doc.page_content for doc in source_documents])

    # This is the prompt we send to the LLM — it acts as an instruction sheet
    prompt_template = PromptTemplate(
        input_variables=["question", "answer", "sources"],
        template="""
You are a financial analyst critic. Your job is to evaluate whether an answer is grounded in the provided source documents.

Question: {question}

Answer given: {answer}

Source documents used:
{sources}

Evaluate the answer based on these criteria:
1. Is the answer factually supported by the source documents?
2. Does the answer contain any claims not found in the sources?
3. Is the answer relevant to the question?

Respond in this exact format:
VERDICT: PASS or FAIL
REASON: (one sentence explaining your verdict)
"""
    )

    # Connect to Groq LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,   # 0 = deterministic, consistent verdicts
    )

    # Fill in the prompt with the actual question, answer and sources
    prompt = prompt_template.format(
        question=question,
        answer=answer,
        sources=sources_text
    )

    # Send the prompt to the LLM and get the verdict
    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    # Test Agent 2 using a hardcoded example to verify it works
    test_question = "What was the ECB's main monetary policy decision in 2023?"
    test_answer = "The ECB raised interest rates by 200 basis points in 2023 to bring inflation back to 2%."

    # Simulate a source document (normally this comes from Agent 1)
    from langchain.schema import Document
    test_sources = [
        Document(page_content="In 2023 the ECB raised its key interest rates by a further 200 basis points, bringing the deposit facility rate to 4%.")
    ]

    print(f"Question: {test_question}")
    print(f"Answer: {test_answer}")
    print("\nEvaluating...")
    verdict = evaluate_answer(test_question, test_answer, test_sources)
    print(f"\nCritic verdict:\n{verdict}")