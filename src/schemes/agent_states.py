from typing import TypedDict, Optional, List, Dict


class QueryAgentState(TypedDict):
    """State after QueryAgent processes user input."""
    user_input: str
    user_id: str
    auth_token: str

    # Output from QueryAgent
    generated_search_query: str

    # Error tracking
    errors: List[str]


class APIExecutionAgentState(TypedDict):
    """State after APIExecutionAgent searches for images."""
    user_input: str
    user_id: str
    auth_token: str

    generated_search_query: str

    # Output from APIExecutionAgent
    candidate_images: Dict[str, str]  # image_id -> image_url

    # Error tracking
    errors: List[str]


class QualityControlAgentState(TypedDict):
    """State after QualityControlAgent filters images."""
    user_input: str
    user_id: str
    auth_token: str

    generated_search_query: str
    candidate_images: Dict[str, str]

    # Output from QualityControlAgent
    quality_score: Optional[str]

    # Error tracking
    errors: List[str]


class ReferenceSelectorAgentState(TypedDict):
    """State after ReferenceSelectorAgent selects and uploads reference."""
    user_input: str
    user_id: str
    auth_token: str
    
    generated_search_query: str
    candidate_images: Dict[str, str]
    quality_score: Optional[str]

    # Output from ReferenceSelectorAgent
    selected_reference_url: str
    reference_image_id: str

    # Error tracking
    errors: List[str]


class SupervisorAgentState(TypedDict):
    """State after SupervisorAgent completes the workflow."""
    user_input: str
    user_id: str
    auth_token: str

    # All agent outputs (accumulated)
    generated_search_query: Optional[str]
    candidate_images: Optional[Dict[str, str]]
    quality_score: Optional[str]
    selected_reference_url: Optional[str]
    reference_image_id: Optional[str]

    # User workflow (manual upload)
    source_image_id: Optional[str]

    # Final translation
    translation_task_id: Optional[str]
    final_output_url: Optional[str]

    # Error tracking
    errors: List[str]
