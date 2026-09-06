# WP3 G3 候选来源零代理抓取报告（2026-08-30）

> 状态：`CANDIDATE ACQUISITION ONLY / NOT APPLIED TO CATALOG`
>
> 范围：`development-only / form-only / label-free / non-lexicon`

## 1. 结果

来源登记文档共有 45 次 URL 引用、42 个唯一 URL。3 个 URL 在不同章节重复，网络请求按精确 URL
去重后各执行一次完整 GET。工作区在抓取前没有可验证的真实来源快照，因此 42 个 URL 全部进入本轮获取。

保留产物：

- capture：`wp3capture-d6ded677ee5667be53ec884b329adc82573be3fb33bc725c05c5a6975beb1900`
- payload manifest SHA-256：`ef3987cb9beeda7fe6892a20effc9c5d7b24a9272101570157f513d5ece81297`
- manifest SHA-256：`c44ecc05736c1b23137e433801e6a6463009992c4ffa46e86b96bc0047ba7937`
- 体积：约 9.8 MiB
- 原始响应文件：37
- UTF-8 正文投影：33

汇总：

| 结果 | 数量 |
| --- | ---: |
| 可用静态正文、有效 PDF 或有效 JSON | 32 |
| 保存了响应但仍需处理 | 5 |
| 未取得响应 | 5 |
| 合计 | 42 |

细分状态：

| 状态 | 数量 |
| --- | ---: |
| `downloaded_usable` | 32 |
| `downloaded_needs_browser_archive` | 2 |
| `downloaded_needs_review` | 1 |
| `downloaded_error_or_challenge` | 1 |
| `http_error` | 1 |
| `network_error` | 2 |
| `unsafe_target` | 3 |

## 2. 零代理与完整性边界

- 进程清除了大小写 `HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY`；Requests 使用
  `trust_env=false`。
- 每个 URL 一次 physical attempt，无自动重试；只允许有界 HTTP/HTTPS 跳转。
- TLS 校验保持开启；没有使用 `verify=false`、忽略证书或代理替换页。
- 响应保存原始实体字节，并绑定 requested/final URL、完整跳转链、状态码、响应头白名单、大小、SHA-256、
  正文投影和获取时间。
- 7 个教育部中英文页面发生同站 HTTPS→HTTP 降级；正文有效，但 manifest 明确记录
  `transport_downgrade=true`，不能静默当作 HTTPS 证据。
- 两份成功 PDF 均通过 `%PDF-` 与尾部 `%%EOF` 容器检查。当前环境没有 `pdfinfo/qpdf`，正式纳入前仍应
  做独立 PDF 结构和正文重放检查。
- GLM 价格页不属于 G3 来源。按用户后续修订，本地价格页面响应及中断的浏览器临时文件已经删除；保留
  capture 中没有 `bigmodel.cn/pricing` URL、HTML 或正文。该 URL 只作为费用估算参考。

零代理配置只能证明客户端未使用环境代理；不能证明宿主网络不存在透明网关或特殊 DNS 路由。

## 3. 原“待人工下载”清单

### P0

6/6 均取得完整可见正文：

- 教育部 BBS 用字用语调查；
- 两个原固定教育部来源；
- 网络语言是非访谈；
- 2019、2020 年度十大网络用语。

六页均由 HTTPS 降级到同站 HTTP，必须按新 source policy 记录传输风险后再决定是否纳入。

### P1

- 成功：Sci-open 2025 论文页面，正文可见且标题匹配。
- 失败：FX361 原 URL 302 到站内 404；保存了 404 响应，不能作正文证据。
- 失败：人民网传媒页与全国政协页均因证书主机名不匹配而停止；没有关闭 TLS 校验。

### P2

- 成功：教育部英文站 2020 网络用语榜单，HTTPS 降级到同站 HTTP。
- 部分：CTgoodjobs 原 HTML 已保存，脚本内能检出 YYDS、NSDD、XSWL、ZQSG、PLGG、PLJJ、BDJW、
  栓Q、尊嘟假嘟等预期词，但静态可见正文不足；可先离线解析内嵌数据，再决定是否仍需 MHTML。
- 失败：北邮页面返回 HTTP 412。

## 4. 其他未完整项

| URL/来源 | 当前结果 | 后续方式 |
| --- | --- | --- |
| CHIME raw JSON | 本机 DNS 映射为非公网地址，安全拒绝 | 在另一网络下载原始 JSON，绑定 commit、license、URL 和 SHA |
| 两个中文 Wikipedia 页面 | 本机 DNS 映射为非公网地址，安全拒绝 | 由另一网络保存具体 `oldid` revision，不接受未锁定 latest 页 |
| Hanspub PDF | HTTP 200 但正文是 `acw_sc__v2` JavaScript/WAF challenge | 用户提供 publisher PDF 或浏览器完整归档 |
| LingoAce 原文章 | 307 到无关的英语课程页 | 视为原文章已迁移/失效；查找发布者存档需另行修订 allowlist |
| 人民网、全国政协 | TLS hostname mismatch | 用户提供原始归档，或另审 HTTP 官方入口；禁止跳过证书校验 |
| FX361 | 站内 404 | 降级为不可用候选，除非用户提供历史归档 |
| 北邮 | HTTP 412 | 用户浏览器归档或项目附件；无正文时不进入候选队列 |

对 5 个“已有响应但正文不完整”的页面另做了一次无代理 Chromium 补抓，结果为 4 个硬超时、1 个浏览器
错误，未恢复任何 rendered DOM。失败凭据封存在：

- render：`wp3render-dafeac19c73591bbaba0240a5283aa8cee78d0f5e1fadfe9cf31b0bae49821d1`
- payload manifest SHA-256：`ebf352ab7319f9367da05fc5b415f7ff1c1105515b95b4983801709d7a082893`

## 5. 下一步

1. 先实现并审核 `wp3-g3-public-source-catalog/v2`：增加 `acquisition_mode`、`source_role`、最终协议和
   downgrade 元数据；不能把本 capture 直接当成正式 source bundle。
2. 从 32 个可用页面中按 `direct_evidence / candidate_pool / prevalence_or_taxonomy / method_only`
   分流；`method_only` 和纯流行度材料不得生成 form relation。
3. 为可接纳页面生成本地 evidence spans，再进入逐条 form review；禁止根据本文或搜索摘要直接生成
   canonical mapping。
4. 优先补 CHIME raw JSON、两个 Wikipedia oldid 与 Hanspub PDF；它们分别影响高覆盖候选池、当前固定
   七来源闭集和高密度谐音关系。
5. GLM 价格页保持 reference-only，不生成本地 pricing snapshot 或 executable pricing bundle。现有
   successor runner 仍要求可重放的价格 evidence；在模型预检前需另行正式修订该预算契约，不能用空
   snapshot 绕过。

