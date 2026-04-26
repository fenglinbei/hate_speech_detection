from typing import (
    Optional,
    Union
)
from fastapi import Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from config import SETTINGS
from utils.compat import model_json, model_dump
from utils.constants import ErrorCode
from utils.protocol import (
    ErrorResponse,
    ReviewRequest,
    ImageContent,
    FaceLibRequest,
    UploadRequest,
    ImageReviewRequest
)
from modules.utils import get_img

async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    ) -> Optional[str]:
    """
    Verify the provided API key.

    If SETTINGS.api_keys is not set, all requests are allowed. If the API key is not provided, a 401
    Unauthorized error is raised with a message indicating that the API key is missing. If an API key is
    provided but invalid, a 403 Forbidden error is raised with a message indicating that the API key is invalid.

    Args:
        auth (HTTPAuthorizationCredentials, optional): The authorization credentials received from the request.

    Returns:
        Optional[str]: The valid API key if provided and verified, otherwise None.

    Raises:
        HTTPException: If the API key is missing or invalid.
    """
    if not SETTINGS.api_keys:
        # api_keys not set; allow all
        return None
    if auth is None:
        logger.warning("API key is missing from the request.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error": {
                    "message": "API key is missing",
                    "type": "authentication_error",
                    "param": None,
                    "code": "missing_api_key",
                }
            },
        )
    
    if (token := auth.credentials) not in SETTINGS.api_keys:
        logger.warning(f"Invalid API key provided: {token}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": {
                    "message": f"API key: {(token := auth.credentials)} is invalid",
                    "type": "authentication_error",
                    "param": None,
                    "code": "invalid_api_key",
                }
            },
        )
    return token


def create_error_response(code: int, message: str, status_code: int=500) -> JSONResponse:
    return JSONResponse(model_dump(ErrorResponse(message=message, code=code)), status_code=status_code)
 

def handle_text_requests(request: ReviewRequest) -> ReviewRequest:

    if request.model not in ["classification", "llm", "politic_classification", "politic_llm", "emotion_classification"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={
                "error":{
                    "message": f"Please provide an available model type.",
                    "status_code": status.HTTP_400_BAD_REQUEST
                    }
                }
        )
    
    if request.model not in ["llm", "politic_llm"] and request.llm_response_tokens is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={
                "error":{
                    "message": f"Param llm_response_tokens is not available while model is {request.model}.",
                    "status_code": status.HTTP_400_BAD_REQUEST
                    }
                }
        )
    else:
        if request.llm_response_tokens is not None and (request.llm_response_tokens < 0 or request.llm_response_tokens > SETTINGS.max_tokens):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                "error":{
                    "message": f"Param 'llm_response_tokens' must be within the range of (0, {SETTINGS.max_tokens})",
                    "status_code": status.HTTP_400_BAD_REQUEST
                    }
                }
            )


    if len(request.text) > SETTINGS.max_text_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={
                "error":{
                    "message": f"Content length {len(request.text)} is greater than the maximum of {SETTINGS.max_text_length}",
                    "status_code": status.HTTP_400_BAD_REQUEST
                    }
                }
        )
    
async def handle_image_requests(request: ImageReviewRequest):

    if request.model not in ["classification", "llm", "politic_classification", "politic_llm"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={
                "error":{
                    "message": f"Please provide an available model type.",
                    "status_code": status.HTTP_400_BAD_REQUEST
                    }
                }
        )
    
    if request.model not in ["llm", "politic_llm"] and request.llm_response_tokens is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={
                "error":{
                    "message": f"Param llm_response_tokens is not available while model is {request.model}.",
                    "status_code": status.HTTP_400_BAD_REQUEST
                    }
                }
        )
    else:
        if request.llm_response_tokens is not None and (request.llm_response_tokens < 0 or request.llm_response_tokens > SETTINGS.max_tokens):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                "error":{
                    "message": f"Param 'llm_response_tokens' must be within the range of (0, {SETTINGS.max_tokens})",
                    "status_code": status.HTTP_400_BAD_REQUEST
                    }
                }
            )
        
    try:
        image = await get_img(content=request.image)
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail={
                "error":{
                    "message": str(e),
                    "status_code": status.HTTP_400_BAD_REQUEST
                    }
                }
        )

def check_facelib_operation_requests(request: FaceLibRequest):
    
    if request.operation.lower() not in ["add", "delete", "search", "match"]:
        return create_error_response(
            code=ErrorCode.UNSUPPORTED_OPERATION,
            message=f"operation {request.operation} unsupported, must in [add, delete, search, match] - 'operation'",
        )
    
    operation = request.operation.lower()
    
    if operation == "add":
        if (not request.face_img and not request.face_img_url) and not request.name and not request.categories:
            return create_error_response(
            code=ErrorCode.PARAM_MISSING,
            message=f"operation {request.operation} necessary parameters are missing, please check apidoc in serverurl/docs or /redoc",
        )
    elif operation == "delete":
        if (not request.face_id and not request.person_id):
            return create_error_response(
            code=ErrorCode.PARAM_MISSING,
            message=f"operation {request.operation} necessary parameters are missing, please check apidoc in serverurl/docs or /redoc",
        )
    elif operation == "search":
        if request.name is None:
            return create_error_response(
            code=ErrorCode.PARAM_MISSING,
            message=f"operation {request.operation} necessary parameters are missing, please check apidoc in serverurl/docs or /redoc",
        )
    elif operation == "match":
        if (not request.face_img and not request.face_img_url):
             return create_error_response(
            code=ErrorCode.PARAM_MISSING,
            message=f"operation {request.operation} necessary parameters are missing, please check apidoc in serverurl/docs or /redoc",
        )
             
             
def check_upload_requests(request: UploadRequest):
    
    if request.content_type.lower() not in ["text", "image", "mix"]:
        return create_error_response(
            code=ErrorCode.INVALID_PARAM,
            message=f"upload content type {request.content_type} unsupported, must in [text, image, mix] - 'content_type'",
        )
    
    if request.content.suggestion.lower() not in ["pass", "block", "review"]:
        return create_error_response(
            code=ErrorCode.INVALID_PARAM,
            message=f"the content suggestion {request.content.suggestion} unsupported, must in [pass, block, review] - 'suggestion'",
        )
        
    if len(request.content.contents) == 0:
        return create_error_response(
            code=ErrorCode.INVALID_PARAM,
            message=f"the content len should be biger than 0 - 'contents'",
        )