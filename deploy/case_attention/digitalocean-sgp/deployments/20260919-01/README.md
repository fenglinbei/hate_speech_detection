# 2026-09-19 增量发布完成

已上线：<https://hsd.fenglin.pro/>。当前版本 `incremental-20260919-01`，发布清单 SHA256 `78f2e2d5056ed557f79858045dcf4c555e57e4e854aa63385e17419ff93a83f9`。页面保留首轮 12 条件，加入内容替换实验 88 条件、完整报告与 258 条比较。

[直接打开本轮的 36 层平均](https://hsd.fenglin.pro/#round=replacement&request=ccr-541-LD-base&mode=mean&parts=parts&ceiling=0.01)。层平均是原始 FP64 注意力沿 36 层的等权算术平均，支持单头／32 头平均、质量／密度、读取位置、条件差分、材料／槽位对齐、排序和 SVG 导出。

界面增加干预类型筛选、同条件基线快捷比较、完整原文、可分享视图链接和手机布局。缓存限制为 4 条件、每条件 2 层及 1 份平均文件；平均视图只请求一份平均文件。旧的 912 个数据文件原样继承，科学结果与审核记录未修改。本次全部工作为 CPU／静态网站处理，无 GPU 运行。

## 验收

- 100 个条件的 20,778,048 个平均 token 数值全量比对；两种求平均实现的最大差异 3.8858e-16，片段重建最大差异 4.4409e-16。
- 独立检查 100 条完整输入和分数、原始片段汇总及 1,219,200 个重算平均值；全部 3,387 个新增资产及 gzip 摘要通过。
- 本地及真实 HTTPS 浏览器分别比对 10,648 个展示值，涵盖逐层／平均、头选择、密度、NA、条件／位置差分、换序对齐、快速切换、Unicode 原文、SVG、深链接和手机布局。页面无脚本异常。
- 线上首次加载实测 11.432 秒，只代表本次链路及冷启动。登录、原始文件摘要和普通 DNS 路径的 TLS 证书检查通过，未登录仍返回 401。
- 切换前后 249 个受保护文件的摘要一致；两份审核会话、现有认证、审核及 PDF 后端的 PID／启动时间／重启次数不变，PDF HTTPS 持续返回 200 且正文摘要相同。只 reload Nginx。
- 临时明文凭据已删除，本机预览已停止。只清理此次重复上传的大包；全部旧版本、线上数据和本地归档保留。服务器余量约 7.18 GB。

## 回退与来源

线上静态目录：`/opt/hsd-case-attention/releases/incremental-20260919-01/public`。
现有 Nginx 配置 SHA256：`38151c0010e6a0a0d03c241f72fab4f12c1c878e5d0d4c8865f918840525901e`。
前一发布为 `layers-20260918-03`；旧配置保存在同目录的 `previous-nginx.conf`，以及远端 `/opt/hsd-case-attention/deployments/incremental-20260919-01/previous-nginx.conf`。

回退前核对配置仍匹配本次摘要，再恢复该配置并执行 `nginx -t` 与 Nginx reload。只回退页面，不恢复任何历史审核会话，不重启审核或 PDF 服务，不重跑实验。

来源为 `reviews/case-attention-v1/results-01`、`reviews/case-content-replacement-v1/results-01` 及其 `report-02`。所有源数据和冻结清单保持原样。实现及操作说明见 `../../incremental-content-v1/README.md`。本目录的 `closeout.json` 索引最终回执与文件摘要；上传期间的 UI 布局修正另有 `ui-package.json`，原大包摘要和初始清单保留在 `payload-package.json`／`payload-release-manifest.json`。
