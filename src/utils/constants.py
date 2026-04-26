from enum import IntEnum, Enum

CONTROLLER_HEART_BEAT_EXPIRATION = 90
WORKER_HEART_BEAT_INTERVAL = 30
WORKER_API_TIMEOUT = 20

class Suggestion(str, Enum):
    
    BLOCK = "block"
    PASS = "pass"
    REVIEW = "review"

class CategoryEnum(str, Enum):
    text_review = "text_review"
    ocr_text_review = "ocr_text_review"
    nsfw_detection = "nsfw_detection"
    ter_detection = "ter_detection"
    face_detection = "face_detection"
    qr_detection = "qr_detection"

class ErrorCode(IntEnum):

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102

    PARAM_MISSING = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303
    UNSUPPORTED_OPERATION = 40304
    INVALID_PARAM = 40305

    INTERNAL_ERROR = 50000
    TEXT_FAILED = 50001
    IMAGE_FAILED = 50002
    OCR_FAILED = 50003
    NSFW_FAILED = 50004
    FACE_DETECTION_FAILED = 50005
    LOAD_TIME_OUT = 50007


class ModelType(Enum):
    Unknown = -1
    OpenAI = 0
    ChatGLM = 1
    LLaMA = 2
    XMChat = 3
    StableLM = 4
    MOSS = 5
    YuanAI = 6
    Minimax = 7
    ChuanhuAgent = 8
    GooglePaLM = 9
    LangchainChat = 10
    Midjourney = 11
    Spark = 12
    OpenAIInstruct = 13
    Claude = 14
    Qwen = 15
    OpenAIVision = 16

    @classmethod
    def get_type(cls, model_name: str):
        model_type = None
        model_name_lower = model_name.lower()
        if "gpt" in model_name_lower:
            if "instruct" in model_name_lower:
                model_type = ModelType.OpenAIInstruct
            elif "vision" in model_name_lower:
                model_type = ModelType.OpenAIVision
            else:
                model_type = ModelType.OpenAI
        elif "chatglm3" in model_name_lower:
            model_type = ModelType.OpenAI
        elif "chatglm" in model_name_lower:
            model_type = ModelType.ChatGLM
        elif "llama" in model_name_lower or "alpaca" in model_name_lower:
            model_type = ModelType.LLaMA
        elif "xmchat" in model_name_lower:
            model_type = ModelType.XMChat
        elif "stablelm" in model_name_lower:
            model_type = ModelType.StableLM
        elif "moss" in model_name_lower:
            model_type = ModelType.MOSS
        elif "yuanai" in model_name_lower:
            model_type = ModelType.YuanAI
        elif "minimax" in model_name_lower:
            model_type = ModelType.Minimax
        elif "川虎助理" in model_name_lower:
            model_type = ModelType.ChuanhuAgent
        elif "palm" in model_name_lower:
            model_type = ModelType.GooglePaLM
        elif "midjourney" in model_name_lower:
            model_type = ModelType.Midjourney
        elif "azure" in model_name_lower or "api" in model_name_lower:
            model_type = ModelType.LangchainChat
        elif "星火大模型" in model_name_lower:
            model_type = ModelType.Spark
        elif "claude" in model_name_lower:
            model_type = ModelType.Claude
        elif "qwen" in model_name_lower:
            model_type = ModelType.Qwen
        else:
            model_type = ModelType.LLaMA
        return model_type