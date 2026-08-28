# Stage 1 P0 本地人工裁决界面

该工具只处理冻结的人类队列：20 条 group–hate 二轮盲审和 4 条 field-type
结构复核。它不会向浏览器发送模型票据、队列原因、置信度、legacy coupling hint，
也不会展示或改写 sealed 自动行。

## 启动

在仓库根目录运行：

```bash
.conda/stage1-p0/bin/python tools/stage1_adjudication_ui/server.py \
  --workspace-root .
```

终端会打印本地地址，默认是 `http://127.0.0.1:8765/`。服务只允许绑定 loopback；
若端口占用，可传 `--port 0` 自动选择空闲端口。用 `Ctrl+C` 停止。

只检查冻结输入而不启动页面：

```bash
.conda/stage1-p0/bin/python tools/stage1_adjudication_ui/server.py \
  --workspace-root . --check
```

## 保存与导出

- 每条裁决先显示核对摘要，人工确认后才写入。
- 选择“修正后通过”会启用可修正路径；单一路径会自动选中，多路径需先勾选要修改的
  路径，其对应值控件才会启用。
- 提交成功会立即更新该条的完成状态与总进度；若仍有待办，界面自动切到下一条，并在
  成功提示中写明刚完成的 case alias。
- 服务端生成 UTC 秒级 `reviewed_at`，使用冻结 reviewer ID，并调用现有 Stage 1
  validator；只有通过后才以读写描述符取得排他锁并原子替换相应 JSONL，兼容当前
  NFS 工作区的锁语义。
- 已确认条目在网页中锁定，不允许静默覆盖；若确需更正，应先停止并走显式复核流程。
- 已确认结果直接保存到现有 20 行与 4 行人审文件；未确认表单只保存在当前浏览器的
  workspace-fingerprint 草稿中，列表可筛选草稿并可清除当前草稿。
- 两部分均完成后，“导出结果包”会下载 ZIP，内含两个可直接交给现有 CLI 校验的
  JSONL、hash manifest 和后续说明。导出会绑定页面看到的两套 revision；其他标签页或
  进程改动后必须先重新加载。

页面不会拼接最终 34 行，也不会签署 reviewer declaration。完成 24 条后，继续严格按
[`stage1-p0-human-review-handoff.md`](../../docs/research/experiment-plans/stage1-p0-human-review-handoff.md)
执行 validate、merge-human、merge-data-adjudication 与 prepare-data-declaration；人类核对
声明 hash 后才可确认 attestation。

## 安全边界

- 固定路由和固定两套 workspace，不接受浏览器提供的任意文件路径。
- 无 CDN、遥测、外部请求或第三方前端依赖。
- CSP、Host/Origin 检查、随机会话 token、请求体上限和 revision/CAS 防并发覆盖。
- 每次读取、提交和导出都会重新解析 workspace，并核对冻结 audit、packet、review ref
  及 payload manifest；导出直接使用刚完成权威校验的内存快照。
- 动态语料只作为文本节点渲染；控制字符与双向覆盖字符在服务端转义。
- 正式裁决文件变化、冻结依赖漂移或语义校验失败时，本次请求不会写入。

## 自检

测试全部在临时副本上运行，不会修改正式裁决文件：

```bash
.conda/stage1-p0/bin/python -m unittest -v \
  tools/stage1_adjudication_ui/test_server.py
```
