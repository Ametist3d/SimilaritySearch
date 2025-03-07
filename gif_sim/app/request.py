from pydantic import BaseModel


class ImageSearchRequest(BaseModel):
    gif: str  # Can be URL or base64 string
    is_base64: bool = False
    k: int = 3
