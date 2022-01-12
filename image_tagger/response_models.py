from pydantic import BaseModel, AnyHttpUrl


class PredictionItem(BaseModel):
    """Model for each label contained in a prediction."""
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    """Model to be returned when a successful prediction is performed."""
    url: AnyHttpUrl
    predictions: list[PredictionItem]


class InfoResponse(BaseModel):
    """Model to be returned when general info about the API are requested."""
    version: str
    revision: int
    description: str
    author: str
    github_url: AnyHttpUrl


class ValidationErrorDetail(BaseModel):
    loc: list[str]
    msg: str
    type: str


class ValidationErrorInfo(BaseModel):
    type: str
    detail: list[ValidationErrorDetail]


class ValidationErrorResponse(BaseModel):
    """Model for all validation error responses."""
    error: ValidationErrorInfo


class HttpErrorInfo(BaseModel):
    type: str
    detail: str


class HttpErrorResponse(BaseModel):
    """Model for all HTTP errors."""
    error: HttpErrorInfo
