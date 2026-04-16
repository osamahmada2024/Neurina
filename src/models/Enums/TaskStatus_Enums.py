from enum import Enum


class TaskStatus(Enum):
    """Status of image translation tasks"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
