# 内容替换数据审核包

从 [REVIEW.md](REVIEW.md) 开始逐条审核。44 项、88 个输入；新 GPU 运行尚未启动。此目录封存后只读；反馈、修订和采用另存版本。

CPU 复核：`PYTHONUTF8=1 .conda/stage1-p0/bin/python scripts/review/audit_case_content_replacement_v1.py --draft /data/liaozijie/hate_speech_detection/reviews/case-content-replacement-v1/draft-01`。此命令不加载模型权重、不初始化 CUDA、不写回审核结论。
