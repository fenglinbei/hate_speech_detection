# 模型适配与已知边界

| 项目 | Qwen3-8B／14B | 本地GLM-4-9B-Chat |
|---|---|---|
| 模板 | 各自已审核的原生模板，enable_thinking=False | 已审核的本地原生模板 |
| 有／无token | 18830／42192 | 98318／98479 |
| 正常取分 | model.model的最后有效隐藏状态，经lm_head投影 | model.transformer的最后有效隐藏状态，经其output_layer投影 |
| pad token | 151643 | 151329 |
| 计算 | 原生FP32，eager attention，batch1，use_cache=False；读数转FP64 | 相同显式约束，使用本地原生eager实现 |
| 格式诊断 | 手动贪心续接，使用冻结的本地EOS集合 | 相同方法，避免依赖旧generate接口 |

两个候选来自同一次完整词表logits，m=z无−z有；同时保留完整词表归一化概率、合法答案质量和候选内支持度。主margin不追加EOS、不使用NCC、不调分类阈值、不按词数归一化。m=0保持平局；工程未决与错误分类分开报告。

所有input_ids、attention_mask与position_ids显式登记；最后有效索引取mask的最后一个1。GLM的return_last_logit直接取张量最后一格，右补齐时那一格可能是pad，因此本适配显式选择最后有效隐藏状态后投影。Qwen也使用同一明确位置操作。微型随机CPU模型已与原生forward对应位置比较，但真实权重与CUDA内核仍需实际工程资格。

本地GLM没有在当前Transformers版本中直接继承GenerationMixin。保留原本地文件，直接调用其原生transformer及输出层进行评分，并以逐步argmax完成格式诊断。没有修改模型源文件、下载转换模型或静默升级模板。其微型模型已通过当前from_pretrained调用路径和safetensors重载；这不证明完整9B权重必然在计划设备上载入成功。

模型配置中的采样偏好不控制本次测量：评分无采样；格式诊断为确定性argmax，每步重新计算完整前缀，最多8 token。每条诊断须恰好一个有／无token后紧接本地EOS；判对判错不作为格式资格条件。

完整模型载入使用本地safetensors，逐项核查missing／unexpected／mismatched keys、参数名称、浮点dtype和设备放置。绑定、载入及收尾验证权重全哈希；CPU准备复用此前23个分片的全哈希并核查metadata哈希和权重stat，没有宣称本轮重新散列了全部权重。

只冻结接口、代码、包版本和资格规则；CUDA版本、设备UUID、具体张量放置及真实显存峰值在实际运行时记录。微型CPU测试覆盖双卡映射的名称分配，不替代实际跨卡搬运或数值验收。
