from pydantic import BaseModel


class AudioSearchRequest(BaseModel):  # Updated class name
    audio: str  # Can be URL or base64 string
    is_base64: bool = False
    k: int = 3
