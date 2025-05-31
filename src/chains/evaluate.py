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

    system_prompt = """Anda adalah evaluator ahli untuk menilai apakah suatu dokumen relevan terhadap pertanyaan pengguna.
    Instruksi:
    Tinjau isi dokumen dan pertanyaan pengguna dengan cermat.
    Tentukan hanya jika dokumen tersebut memberikan informasi yang relevan untuk menjawab pertanyaan.
    Jangan membuat asumsi atau mengisi kekosongan informasi.
    Berikan hanya satu dari dua jawaban berikut:
        1. 'ya' → Dokumen relevan dan sepenuhnya mendukung jawaban atas pertanyaan.
        2. 'tidak' → Dokumen tidak relevan secara jelas, atau informasinya tidak cukup.
    Bersikap tegas. Jika ragu, pilih 'tidak'.
    """

    evaluate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "Question: {question}\n\nDocument: {document}"),
        ]
    )

    return evaluate_prompt | structured_output