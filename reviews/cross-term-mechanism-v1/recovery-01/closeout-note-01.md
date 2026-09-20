独立数值审计和7项回归检查通过后，科学收尾的首次CPU预检查停止：归档脚本先创建recovery-01/manifest.json，再枚举该目录，误把尚未写入的manifest本身作为0字节制品纳入，形成无效自引用。失败日志保留在../closeout-preflight-console-02.log；当时没有创建科学结果选择器或closeout目录。

保留原manifest.json，不修改其记录。新manifest-02.json在打开输出文件之前计算完整文件清单，将原错误manifest作为历史制品保存，同时明确排除自身。新的closeout_cross_term_mechanism_v3.py只把恢复归档引用改为manifest-02.json、收尾目录改为closeout-03，其余v2检查逻辑不变。原v2源码与运行前pin保留。

这是CPU归档元数据的修复；独立审计结果、全部科学原始数据与图表、数值门槛和GPU运行均未改变。完整数值审计没有重跑，GPU没有重启。
