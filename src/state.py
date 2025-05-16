from typing import List, TypedDict, Optional

class GraphState(TypedDict):
    """State for the LangGraph workflow."""
    question: str
    documents: List[str]
    web_results: Optional[str]
    solution: Optional[str]
    conversation_id: int