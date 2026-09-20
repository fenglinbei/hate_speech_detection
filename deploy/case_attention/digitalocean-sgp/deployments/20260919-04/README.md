# 查询表示替换增量部署完成

[全部层交互图](https://hsd.fenglin.pro/reports/patching/) · [完整报告](https://hsd.fenglin.pro/reports/patching/REPORT.html)。发布 `incremental-20260919-04`，沿用原登录。

旧127个注意力条件、应用脚本、目录和全部科学数据保持字节一致，首页仅添加一个入口。新页是四份既有输入的288个跨条件替换、288个自替换和144个位置差值，不是288个新样本。所有层、两方向、前置对照、NA、SVG、深链接、手机和报告下载均核验。

本地与真实HTTPS各核对77,510值，其中旧回归74,198、新页面3,312。两边完整科学检查清单相同；旧缓存遥测仅核对原限制。

仅切换hsd Nginx root并平滑reload。249项受保护文件、认证、两个人审会话、审核/PDF服务身份/启动/重启次数及PDF HTTPS响应前后完全相同。临时明文、预览、浏览器已清理，只删除本次已验证展开的重复上传包，保留旧发布和本地归档。

发布清单SHA256 `ad94deed70330b52c94fa8664ab695404c3d47ffc179d59ab4596a4ce69179be`；当前Nginx SHA256 `4886ad48ddcfd666578a720c176b5958d570e1a06054b76338d393e9ccd8d866`。回退仅在核对当前配置后恢复previous-nginx.conf，nginx -t并reload至incremental-20260919-03；禁止回退人工记录或重启PDF/审核服务。
