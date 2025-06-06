from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence


def create_generate_chain() -> RunnableSequence:
    """Creates a chain to generate answers based on context and question."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
)

    prompt = ChatPromptTemplate.from_template(
        """Context:
        {context}

        Question: {question}

        You are an assistant tasked with answering questions accurately, using only the information provided internally. If the necessary information is not available, simply respond that you do not know the answer, without referring to any source or context.
    
        Answer:
        """
    )

    return prompt | llm | StrOutputParser()