from typing import TypedDict, Optional, List, Dict

class AgentState(TypedDict):

    user_id: str
    session_id: str
    user_input: str
    source_image_id: str
    auth_token: str

    status: str
    chat_history: List[Dict[str, str]]
    retry_count: int
    intent: Optional[str]
    style_description: Optional[str]
    selected_index: Optional[int]
    rag_response: Optional[str]

    generated_search_query: Optional[str]
    candidate_images: Optional[Dict[str, str]]
    selected_reference_url: Optional[str]
    uploaded_reference_url: Optional[str]

    reference_image_id: Optional[str]
    translation_task_id: Optional[str]
    translated_image_id: Optional[str]
    final_output_url: Optional[str]

    quality_score: Optional[str]
    errors: List[str]

    # LangGraph routing (optional, persisted in session)
    route: Optional[str]
    selected_candidate_id: Optional[str]
