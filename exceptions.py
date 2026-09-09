class DocMindError(Exception):
    """Base exception for 1-DocMind application."""
    def __init__(self, message, error_code="INTERNAL_ERROR", status_code=500):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class TesseractUnavailableError(DocMindError):
    def __init__(self, message="Tesseract OCR engine is unavailable or not found in PATH."):
        super().__init__(message, error_code="TESSERACT_UNAVAILABLE", status_code=503)


class OllamaUnavailableError(DocMindError):
    def __init__(self, message="Ollama LLM service is unavailable or unreachable."):
        super().__init__(message, error_code="OLLAMA_UNAVAILABLE", status_code=503)


class ModelNotFoundError(DocMindError):
    def __init__(self, model_name="llama3.1:8b"):
        message = f"Configured model '{model_name}' was not found in Ollama instance."
        super().__init__(message, error_code="MODEL_NOT_FOUND", status_code=503)


class LLMParseError(DocMindError):
    def __init__(self, message="Failed to parse LLM response into valid JSON structure."):
        super().__init__(message, error_code="LLM_PARSE_FAILED", status_code=422)


class InvalidImageError(DocMindError):
    def __init__(self, message="Uploaded file is not a valid or readable image."):
        super().__init__(message, error_code="INVALID_IMAGE", status_code=400)


class InvalidFileTypeError(DocMindError):
    def __init__(self, message="Unsupported file extension."):
        super().__init__(message, error_code="INVALID_FILE_TYPE", status_code=415)


class FileTooLargeError(DocMindError):
    def __init__(self, message="Uploaded file exceeds maximum allowed size."):
        super().__init__(message, error_code="FILE_TOO_LARGE", status_code=400)
