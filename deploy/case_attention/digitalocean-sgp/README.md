# 注意力可视化部署到 hsd.fenglin.pro

2026-09-18，用户明确授权替换现有页面，并指定网页登录账号。
入口为 **https://hsd.fenglin.pro/**，用户名 `liaozijie`；密码按本次用户指令设置，不保存在仓库或发布包中。

线上入口现在是 #541 / #3169 注意力可视化。打开后自动加载默认条件，无需手动选择本地文件。用户进一步确认带宽受限时应优化展示逻辑，现已采用 **汇总按条件加载、token 权重按当前层加载** 的版本。跨条件对比仅取对方汇总，切换头或读取位置复用已有层数据。全部 12 个原始 `*.view.json` 仍与封存结果逐字节相同；汇总值不重算，每层显示字节与原 FP32 显示副本严格一致，并在浏览器校验 SHA256。研究源文件、实验数组和冻结清单没有修改。

真实 HTTPS 浏览器验收通过：最终版本首次加载 8.314 秒、切换案例 4.572 秒、跨条件对比 4.310 秒；前一轮分别为 7.459、3.480、2.038 秒。默认 #541 / LD 的数据压缩传输量从 38,254,492 字节降至 4,884,451 字节，减少 87.2%。实测耗时取决于链路及浏览器，不是性能保证。原整份下载方案的两次浏览器超时保留在回执中，没有作为成功验收。

前一轮浏览器验收期间，SGP 的 30 秒采样平均 CPU 使用率为 3.11%，峰值为 10.89%，最低可用内存 387.4 MiB。该次测量没有出现 CPU 或内存饱和；大文件传输是当时主要等待来源。因此继续在 SGP 直接托管，并减少单次请求量。将源站迁到当前服务器再通过 SGP 转发仍会经过 SGP 出口，本次未增加该链路。

目标服务器为 SSH 别名 `digitalocean-sgp`。Nginx 沿用现有域名、证书和安全响应头，使用独立 Basic Auth 密码哈希文件 `/etc/nginx/.htpasswd-hsd-case-attention`，HTML、JSON 和 gzip 资源均需要登录。预压缩文件用于传输，原始 JSON 仍保留并校验。缓存设为私有且每次使用前验证。

当前发布：`layers-20260918-03`，清单 SHA256 为 `c2875336c640fa5c78b34f40b6805fa94a3071e83edbdd8549e7174870e1faaf`。该版本还补齐了直接点击两张热图单元格时的层数据加载。原始整份数据发布包 SHA256 为 `8a9de828b07879672a60616d39b269565362ebdbf4f46d54676f8ffddcce2695`，作为历史版本保留。

- 当前静态目录：`/opt/hsd-case-attention/releases/layers-20260918-03/public/`。
- 当前切换回执与前一可视化版本配置：`/opt/hsd-case-attention/deployments/layers-20260918-03/`。
- 原审核页面的配置备份：`/opt/hsd-case-attention/deployments/8a9de828b07879672a60616d39b269565362ebdbf4f46d54676f8ffddcce2695/previous-nginx.conf`。
- 本地执行与浏览器回执：`deployments/20260918-01/`。
- 科学结果来源：`reviews/case-attention-v1/results-01/`；结果 manifest SHA256 为 `c3bbf0b231f63f80a6d0394e98d24feaa79aea50647058b5eea976a7d6dfc455`。

原审核后端、当前代码链接、两层审核会话及原密码哈希均保留。未停止、重启或改写审核服务与 PDF 服务。仅切换该域名的 Nginx 路由并平滑 reload。

若需回退页面，先核对站点配置仍为本次发布版本，再选择对应目标的 `previous-nginx.conf`，执行 `nginx -t` 后 reload Nginx。当前配置 SHA256 为 `33f1293645c4ac2fc7d9f1ed2a2fab0befe69effc490740405b69db48fef7b20`。03 备份回到 02 分层版本，02 备份回到整份下载可视化；原包目录的备份回到旧审核页面。不要恢复任何历史审核会话备份，也不要重跑历史审核导入或实验。

`build_release.py` 验证封存清单后打包；`activate_release.py` 通过标准输入接收临时凭据，验证发布包和现有站点配置，持有原部署锁，备份后切换，失败时恢复旧配置。`prepare_layer_release.py` 在独立目录生成无损按层文件及相同汇总，`patch_layer_interactions.py` 在新的独立版本修正热图点击回调，`activate_layer_release.py` 只切换静态根目录。`smoke_live.cjs` 保留前一轮验收，`smoke_live_v2.cjs` 验证最终版本的真实 HTTPS 页面、自动加载、完整 prompt、数值轴、热图点击、阶段与条件对比、分层数据校验及 SVG 导出。`external-https-final.json` 另行验证普通网络路径上的证书、未登录 401 和已登录页面哈希。临时明文凭据在最终检查后删除；继续使用 SGP 直接托管，没有新建当前计算服务器的 Web 服务或转发隧道。
