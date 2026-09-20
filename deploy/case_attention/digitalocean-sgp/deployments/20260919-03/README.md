# 第四轮增量部署完成

本轮入口：[释义顺序与机制](https://hsd.fenglin.pro/#round=presentation&request=hpm-Q01-D04&mode=mean&parts=parts&ceiling=0.01)。[完整报告](https://hsd.fenglin.pro/reports/presentation/) · [机制页面](https://hsd.fenglin.pro/reports/presentation/mechanism.html)。当前发布 `incremental-20260919-03`，沿用既有登录。

原三轮109个条件全部保留，加入本轮18个新测量，共127个条件。全部层、头、层平均、注意力质量/密度、差分、NA、SVG、完整prompt、报告和下载均已核验。机制页面另提供逐层投影、来源向量与配对表示距离。

CPU逐条核对127份输入及分数，独立重算1,432,704个聚合均值和2,897,280个新增token均值；8,400个继承文件原样保留，710项新增资产及gzip摘要通过。本地和真实HTTPS完整浏览器各核对74,198个数值，含全部18条件和56,412个机制值，并通过旧轮次回归、深链接、报告/PDF、手机与SVG检查。

仅切换hsd Nginx静态root并平滑reload。249个受保护文件、原登录、两个审核会话、审核/PDF进程启动时间和重启计数及PDF HTTPS响应前后严格相同。发布清单摘要 `acfe6e0c51bfa7ea8e5629143ed68e9e224dc3318f78fea8396e47e7725a5a7b`；当前Nginx配置摘要 `3148d7abd1ba3a3ec395e2fe300293b3436cb6a47cc9fb2c5b48f1d4f3ee5cc5`。

科学输入与封存结果未在部署中改变，GPU运行保持终态。缺失不是零；注意力和局部向量投影不能单独建立因果路径。第二阶段激活替换未执行。

临时明文凭据已删除，本轮预览及浏览器均已退出。只删除本轮已经验证/展开的重复上传包，保留本地归档、旧发布和所有人工记录。

页面回退目标为incremental-20260919-02：核对当前Nginx摘要后，恢复本目录或远端部署目录的previous-nginx.conf，执行nginx -t并reload。不要回退人工审核会话，不重启审核/PDF服务，不重跑已完成GPU实验。
