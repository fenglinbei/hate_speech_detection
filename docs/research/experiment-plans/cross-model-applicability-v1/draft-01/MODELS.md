# 三模型选择与 CPU 适配

建议采用下表三个本地快照。选择依据是同代同族规模对照、另一家族近似规模、中文及通用指令训练适用性、完整本地权重和可读取logits的代码接口；未查看本次材料上的模型输出。仓库中项目微调的`models/exps`权重未选用。

| 角色 | 标称官方模型 | 本地路径 | CPU答案token：有／无 |
|---|---|---|---|
| M0 | Qwen/Qwen3-8B | models/base/Qwen3-8B | 18830／42192 |
| M1 | Qwen/Qwen3-14B | /data/models/Qwen3-14B | 18830／42192 |
| M2 | zai-org/glm-4-9b-chat（旧命名THUDM/glm-4-9b-chat） | models/base/GLM-4-9B-Chat | 98318／98479 |

Qwen官方将8B／14B列为经过预训练与后训练的模型，并提供关闭显式思考的模板开关；不因本地目录名base而称其为纯预训练模型。[Qwen3-8B官方卡](https://huggingface.co/Qwen/Qwen3-8B)、[Qwen3-14B官方卡](https://huggingface.co/Qwen/Qwen3-14B)。GLM官方将该版本描述为经过人类偏好对齐的Chat模型，旧THUDM页面目前重定向到zai-org。[GLM-4-9B-Chat官方卡](https://huggingface.co/zai-org/glm-4-9b-chat)。这些卡片于2026-09-18查阅，不作为本地权重与上游逐字节一致的证明。

GLM被选作可审核的中文、近似规模、另一家族对照，不代表当下最大或最强模型。其训练年代、后训练、架构、模板均与Qwen不同，差异不能全部归因于家族名。

## 已实际完成的 CPU 核查

- 三个本地模型的全部23个safetensors分片已逐字节SHA256；读取安全JSON头核对张量名称、形状、分片索引和数据字节总数，没有反序列化模型张量。
- 配置、tokenizer、模板、索引和GLM本地实现文件另有哈希。身份详情在[cpu-model-inventory.json](cpu-model-inventory.json)。随后通过公开Hub文件元数据核对，三个模型全部23个权重分片与下表官方commit的SHA256相同；该后续证据见[upstream-verification.json](upstream-verification.json)，初始本地清单中尚未知上游版本的记录原样保留。
- 使用本地tokenizer，无网络下载和远程模型代码获取。GLM的tokenizer源码已阅读后按本地文件导入，没有导入其模型实现执行权重加载。
- 三模型分别检查192个新输入和156个旧输入，共1044次prompt重建、2088个候选续接边界。要求每个“有／无”单token、`encode(prompt+label)=encode(prompt)+label_id`、原生模板tokenize结果一致、长度小于8192。
- 8B的156个历史输入还必须逐字节匹配原prompt哈希、逐项匹配原token序列。实际核查结果和逐模型长度范围以[cpu-tokenizer-audit.json](cpu-tokenizer-audit.json)为准。

Qwen原生模板使用`enable_thinking=false`并保留其原生空think块；GLM原生模板直接结束在assistant标记，没有伪造一个同名思考开关。二者任务消息相同但模板不同。GLM本地`ChatGLMForConditionalGeneration.forward`静态接口返回logits；实际前向、数值内核和自由生成遵从性均尚未验证。

| 模型 | 权重匹配的官方commit | 其他文件核对 |
|---|---|---|
| Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 | 所核查配置／tokenizer等文件也全部匹配 |
| Qwen3-14B | 40c069824f4251a91eefaf281ebe4c544efd3e18 | 所核查配置／tokenizer等文件也全部匹配 |
| GLM-4-9B-Chat | bd8234fe5e0c09c48637a92abb0c797cb5fa0e73 | 权重、config、词表及索引匹配；四个本地配套文件与此上游commit不同 |

GLM差异为`generation_config.json`、`tokenizer_config.json`、`tokenization_chatglm.py`、`modeling_chatglm.py`。本次核查并绑定的是本地版本，没有用新版文件覆盖。故不能把GLM整套运行目录称为上述commit的完全一致checkout；权重身份和本地实现身份分开记录。CPU模板核查采用这些本地文件，GPU前需要另行落实其兼容性。

## 精度、用卡与未完成事项

拟保留FP32模型计算及FP64读数数学，具体执行适配另版冻结。仅按存储张量元素估算FP32权重：8B约32.76GB、14B约59.07GB、GLM约37.60GB；这些是十进制字节估计，不含激活、工作区和运行开销。不能据此直接宣布某模型单卡可跑，也不据“四卡”预先指定一模型一卡。14B需要根据实际显存作多卡／其他合格部署安排。

GLM旧自定义实现与现有运行环境的兼容性只做了tokenizer与接口层检查；官方也提示新transformers使用HF集成版本可减少兼容问题。这里没有切换到另一个转换权重快照或修改旧实现。后续若需转换或适配，应记录新版本与桥接，不静默替换候选模型。

尚未完成：模型前向兼容性、GPU数值资格、输出遵从性、正式显存与设备绑定；GLM本地实现与上游文件的差异需要在执行适配中处理。三模型CPU通过不改变这些状态，也不等于本次科学模型清单已经获得人工采纳。
