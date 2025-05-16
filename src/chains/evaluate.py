from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_google_genai import ChatGoogleGenerativeAI

class EvaluateDocs(BaseModel):
    """Model for document relevance evaluation."""
    score: str = Field(
        description="Document is relevant to the question, 'yes' or 'no'"
    )

def create_evaluate_chain() -> RunnableSequence:
    """Creates a chain to evaluate document relevance."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    structured_output = llm.with_structured_output(EvaluateDocs)

    system_prompt = """You are an expert evaluator assessing whether a document is relevant to a user's question.
    Instructions:
    1. Review the document content and the user's question.
    2. Determine if the document provides sufficient information to answer the question.
    3. Consider:
       - Relevance of main topics to the question.
       - Depth and specificity of information.
       - Alignment with the question's intent.
    4. Provide a binary score: 'yes' (relevant) or 'no' (not relevant).

    Example:
    Question: "What is AI?"
    Document: "Artificial Intelligence (AI) is the simulation of human intelligence in machines."
    Score: yes

    Question: "What is AI?"
    Document: "The history of the Roman Empire spans several centuries."
    Score: no
    """

    evaluate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}\n\nDocument: {document}"),
        ]
    )

    return evaluate_prompt | structured_output