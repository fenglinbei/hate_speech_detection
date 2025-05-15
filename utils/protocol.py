import secrets
import uuid

import numpy as np
from datetime import datetime
from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field, validator, HttpUrl, Base64Str


class ProcessError(Exception):
    def __init__(self, msg: str, code: int, *args: object) -> None:
        super().__init__(*args)
        self.msg = msg
        self.code = code

class ErrorResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"error-{secrets.token_hex(16)}", description="请求的唯一标识")
    object: str = Field(default="error", description="请求的返回类型")
    created: str = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    message: str = Field(..., description="错误信息")
    code: int = Field(..., description="错误代码")
    

class Person(BaseModel):
    
    person_id: str = Field(default_factory=lambda: f"person-{secrets.token_hex(16)}", description="人物的唯一id")
    name: str = Field(..., description="人物名称")
    categories: List[int] = Field(default=[], description="人物所属类别")

class Face(BaseModel):

    class Config:
        arbitrary_types_allowed = [np.ndarray]
    
    face_id: str = Field(default_factory=lambda: f"face-{secrets.token_hex(16)}", description="人脸的唯一id")
    
    person: Optional[Person] = Field(default=None, description="人物信息")
    
    bbox: np.ndarray = Field(default=None, description="人脸在原图的位置，[左上x, 左上y, 右下x, 右下y]")
    kps: np.ndarray = Field(default=None, description="人脸关键点坐标，[左眼坐标, 右眼坐标, 鼻子坐标, 左嘴角坐标, 右嘴角坐标]")
    pose: np.ndarray = Field(default=None, description="人脸的3d旋转角，[x轴角度, y轴角度, z轴角度]")
    
    embedding: Optional[np.ndarray] = Field(default=None, description="人脸的特征向量，为512维的向量")
    
    face_img: Optional[np.ndarray] = Field(default=None, description="人脸对齐后的图像")
    src_img: Optional[np.ndarray] = Field(default=None, description="原始图像")


class UsageInfo(BaseModel):
    """LLM的token使用信息"""
    completion_tokens: int = Field(default=0, description="llm生成回复消耗的token数")
    prompt_tokens: int = Field(default=0, description="prompt消耗的token数")
    total_tokens: int = Field(default=0, description="完成该次审核总共消耗的token数")
        
    def add(self, usage):
        self.completion_tokens += usage.completion_tokens
        self.prompt_tokens += usage.prompt_tokens
        self.total_tokens += usage.total_tokens
    
class ImageContent(BaseModel):
    type: Literal["base64", "url"] = Field(..., example="url", description="信息类型")
    data: str = Field(..., example="http://example.com/image.png", description="图像数据，支持url以及base64")
    # data: Union[HttpUrl, Base64Str] = Field(None, example="http://example.com/image.png", description="图像数据，支持url以及base64")
    
class Segment(BaseModel):
    """风险信息片段"""
    segment: str = Field(..., description="风险信息片段")
    glossary_name: str = Field(..., description="命中的敏感词")
    glossary_class: str = Field(..., description="命中的敏感词所属类别")
    confidence: Optional[float] = Field(default=None, description="风险信息的置信度")
    

class QRDetectionResult(BaseModel):
    top_left_x: int = Field(..., description="检测出二维码的左上角横坐标")
    top_left_y: int = Field(..., description="检测出二维码的左上角纵坐标")
    botton_right_x: int = Field(..., description="检测出二维码的右下角横坐标")
    botton_right_y: int = Field(..., description="检测出二维码的右下角纵坐标")
    content: str = Field(..., description="二维码包含的信息")
    

class FaceDetectionBox(BaseModel):
    top_left_x: int = Field(..., description="检测出人脸的左上角横坐标")
    top_left_y: int = Field(..., description="检测出人脸的左上角纵坐标")
    botton_right_x: int = Field(..., description="检测出人脸的右下角横坐标")
    botton_right_y: int = Field(..., description="检测出人脸的右下角纵坐标")


class FaceDetectionResult(BaseModel):
    suggestion: str = Field(..., description="""检测结果  
                            block：包含敏感人物  
                            pass：不包含敏感人物  
                            review：可能与敏感人物相似""")
    bbox: FaceDetectionBox = Field(..., description="检测出的人脸")
    personage: Optional[str] = Field(default=None, description="检测出与该人脸最相似的人物")
    confidence: Optional[float] = Field(default=None, description="检测出与该人脸最相似的人物的置信度")
    categories: Optional[List[str]] = Field(default=None, description="检测出与该人脸最相似的人物所属类别")
    

class VLReviewResult(BaseModel):
    suggestion: str = Field(..., description="""审核结果  
                            block：包含敏感信息，不通过  
                            pass：不包含敏感信息，通过  
                            review：需要人工复检""")
    response: str = Field(default="", description="模型生成的原始回答")
    has_option: bool = Field(..., description="""生成的回答中，是否包含选项  
                             若包含，则根据选项返回检测结果  
                             若不包含，模型将根据服务启动参数SECOND_REVIEW确定是否重新以新的参数输入给llm审核，不包含选项时，检测结果将返回review""")
    usage: UsageInfo = Field(..., description="llm的token消耗信息")
    
class NSFWDetectionResult(BaseModel):
    suggestion: str = Field(..., description="""审核结果  
                            block：包含敏感信息，不通过  
                            pass：不包含敏感信息，通过  
                            review：需要人工复检""")
    category: str = Field(..., description="识别结果中置信度最高的类别")
    confidence: float = Field(..., description="识别结果中置信度最高的类别的置信度")
    categories: List[str] = Field(default=[], description="识别结果的类别")
    confidences: List[float] = Field(default=[], description="识别结果各类别的置信度")
    
class TerDetectionResult(BaseModel):
    suggestion: str = Field(..., description="""审核结果  
                            block：包含敏感信息，不通过  
                            pass：不包含敏感信息，通过  
                            review：需要人工复检""")
    category: str = Field(..., description="识别结果中置信度最高的类别")
    confidence: float = Field(..., description="识别结果中置信度最高的类别的置信度")
    categories: List[str] = Field(default=[], description="识别结果的类别")
    confidences: List[float] = Field(default=[], description="识别结果各类别的置信度")


    

class TextReviewResult(BaseModel):
    """文本审核结果"""
    suggestion: str = Field(..., description="""审核结果  
                            block：包含敏感信息，不通过  
                            pass：不包含敏感信息，通过  
                            review：需要人工复检""")
    categories: Optional[List[str]] = Field(default=[], description="风险类别")
    confidence: Optional[float] = Field(default=None, description="检测的风险值")
    segments: List[Segment] = Field(..., description="命中的敏感词或语义模型识别到的风险片段")
    review_text: str = Field(..., description="经过预处理后的文本")
    llm_response: Optional[str] = Field(default=None, description="llm模型生成的原始回答，当single_word为true时为空")
    llm_usage: Optional[UsageInfo] = Field(default=None, description="llm的token消耗信息")
    

class OCRResult(TextReviewResult):
    ocr_text: str = Field(..., description="ocr检测出的文本")

class ImageDetectionResult(BaseModel):
    suggestion: str = Field(..., description="""审核结果  
                            block：包含敏感信息，不通过  
                            pass：不包含敏感信息，通过  
                            review：需要人工复检""")
    ocr_result: Optional[OCRResult] = Field(default=None, description="ocr图文检测的结果")
    nsfw_detection: Optional[NSFWDetectionResult] = Field(default=None, description="色情识别的结果")
    terrorism_detection: Optional[TerDetectionResult] = Field(default=None, description="暴恐识别的结果，当前未启用")
    face_detection: Optional[List[FaceDetectionResult]] = Field(default=None, description="涉政人物识别的结果")
    qr_detection: Optional[List[QRDetectionResult]] = Field(default=None, description="二维码检测的结果，当前未启用")

class ImageReviewResult(BaseModel):
    """图像审核结果"""
    suggestion: str = Field(..., description="""审核结果  
                            block：包含敏感信息，不通过  
                            pass：不包含敏感信息，通过  
                            review：需要人工复检""")
    details: ImageDetectionResult = Field(..., description="检测详情")

class TextReviewResponse(BaseModel):
    """文本审核接口响应参数"""
    id: str = Field(default_factory=lambda: f"ctwvtx-{secrets.token_hex(16)}", description="请求的唯一标识")
    object: str = Field(default="review.text.result", description="请求的返回类型")
    created: str = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    model: Optional[str] = Field(default=None, description="使用的LLM模型名称")
    result: TextReviewResult = Field(..., description="审核结果")
    usage: UsageInfo = Field(..., description="llm的token消耗信息")

class ImageReviewRequest(BaseModel):
    model: Literal["classification", "llm", "politic_classification", "politic_llm"] = Field(default="classification", description="""
                                                                                             设定用于文字审查的模型
                                                                                             llm: 使用大模型进行违规审查
                                                                                             politic_llm: 使用大模型进行政治敏感审查
                                                                                             classification: 使用常规模型进行违规审查
                                                                                             politic_classification: 使用常规模型进行政治敏感审查""")
    image: ImageContent= Field(..., description=f"待审图像数据")
    use_sensitive_predict: bool = Field(default=True, description="是否启用敏感词检查")
    llm_response_tokens: Optional[int] = Field(default=None, description="大模型审核回复的token数量，设置为一定值可以获取大模型审核的审核建议，仅当model属于[llm, politic_llm]时有效")
    
class ImageReviewResponse(BaseModel):
    """图像审核接口响应参数"""
    id: str = Field(default_factory=lambda: f"ctwvim-{secrets.token_hex(16)}", description="请求的唯一标识")
    object: str = Field(default="review.image.result", description="请求的返回类型")
    created: str = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    llm_model: Optional[str] = Field(default=None, description="使用的LLM模型名称")
    result: ImageReviewResult = Field(..., description="审核结果")
    usage: UsageInfo = Field(..., description="llm的token消耗信息")
    

class FaceLibRequest(BaseModel):
    """人脸库操作请求Body"""
    operation: Literal["add", "delete", "search", "match"] = Field(..., description="""需要对人脸库执行的操作，可用的参数如下:  
                           add: 把人脸添加进人脸库，需要提供人脸图像以及人名与所属类别信息  
                           delete: 删除指定人脸id或者人物id的人脸，需要提供人脸id或人物id  
                           search: 根据人名查找人物id，需要提供人名信息  
                           match: 匹配最多topK张与传入的图像最匹配的人脸，需要提供人脸图像""")
    face_img: ImageContent = Field(default=None, description="""人脸图像，需要确保图像中有且只有一张完整且清晰的人脸  
                                      若存在多个人脸，则只会把置信度最大的一张作为目标人脸""")
    name: Optional[str] = Field(default="", description="人物的完整人名")
    categories: Optional[List[int]] = Field(default=[], description="人物所属类别")
    face_id: Optional[str] = Field(default="", description="人脸的唯一id")
    person_id: Optional[str] = Field(default="", description="人物的唯一id")
    limit: Optional[int] = Field(default=10, description="最大匹配数目")

class UploadFaceParams(BaseModel):
    image: ImageContent = Field(..., description="""人脸图像，需要确保图像中有且只有一张完整且清晰的人脸  
                                      若存在多个人脸，则只会把置信度最大的一张作为目标人脸""")
    name: Optional[str] = Field(default="", description="人物的完整人名")
    categories: Optional[List[int]] = Field(default=[], description="人物所属类别")
    person_id: Optional[str] = Field(default="", description="人物的唯一id")

class FaceMatchParams(BaseModel):
    image: ImageContent = Field(..., description="""人脸图像，需要确保图像中有且只有一张完整且清晰的人脸  
                                      若存在多个人脸，则只会把置信度最大的一张作为目标人脸""")
    limit: Optional[int] = Field(default=10, description="最大匹配数目")


class FaceMatchResult(BaseModel):
    dis: float = Field(..., description="匹配的余弦距离，越低表示越相似")
    face_id: str = Field(..., description="人脸的唯一id")
    person_id: str = Field(..., description="人物的唯一id")
    name: str = Field(..., description="人物的完整人名")


class FaceLibResponse(BaseModel):
    operation: str = Field(..., description="执行的操作")
    flag: bool = Field(..., description="执行是否成功的标志")
    match_person: Optional[List[Person]] = Field(default=None, description="search操作返回的人物id信息")
    match_res: Optional[List[FaceMatchResult]] = Field(default=None, description="match操作返回的人脸匹配结果")


class UploadDataIdentification(BaseModel):
    data_id: str = Field(default_factory=lambda: uuid.uuid1().hex, description="基于时间戳的唯一id")
    created: datetime = Field(default_factory=lambda: datetime.today(), description="请求创建时间")
    creator: str = Field(default="anonymity", description="数据贡献者")
    content_type: Literal["text", "image", "mix"] = Field(default="text", description="待审数据类型，可选项有 text: 文字 image: 图像 mix: 混合")


class UploadContent(BaseModel):
    text: Optional[str] = Field(default="", description="文字内容")
    url: Optional[str] = Field(default="", description="图像的url")
    base64_image: Optional[str] = Field(default="", description="图像的base64编码，与url二选一")
    

class UploadContents(BaseModel):
    contents: List[UploadContent] = Field(default=[], description="内容列表，当content_type为mix时列表有多项")
    suggestion: Literal["pass", "block", "review"] = Field(default="pass", description="标签信息，一般为”pass”或”block”")
    reason: Optional[str] = Field(default="", description="给出判断的理由")


class UploadData(BaseModel):
    identification: UploadDataIdentification = Field(default=UploadDataIdentification(), description="数据标识部分")
    content: UploadContents = Field(default=UploadContents(), description="数据内容部分")
    

class UploadRequest(BaseModel):
    creator: Optional[str] = Field(default="anonymity", description="数据贡献者")
    content_type: Literal["text", "image", "mix"] = Field(..., description="待审数据类型，可选项有 text: 文字 image: 图像 mix: 混合")
    content: UploadContents
    

class UploadResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"upload-{secrets.token_hex(16)}", description="请求的唯一标识")
    object: str = Field(default="collector.upload", description="请求的返回类型")
    created: datetime = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    data_id: str = Field(default_factory=lambda: uuid.uuid1().hex, description="数据的id")
    result: bool = Field(..., description="状态标识")

class PromptRequset(BaseModel):
    new_prompt: Optional[str] = Field(default=None, description="新的prompt")
    reset: Optional[bool] = Field(default=False, description="是否重置为原来的prompt")
    task_type: Optional[Literal["default", "politic"]] = Field(default="default", description="重置prompt后设置的prompt类型，当reset为true时生效")
    shuffle: Optional[bool] = Field(default=False, description="是否打乱案例")

class PromptResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"prompt-{secrets.token_hex(16)}", description="请求的唯一标识")
    object: str = Field(default="prompt", description="请求的返回类型")
    created: datetime = Field(default_factory=lambda: datetime.today().isoformat(), description="请求创建时间")
    prompt: str = Field(..., description="prompt")