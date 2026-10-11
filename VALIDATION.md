# 2026-10-11 权重更新验证

训练源SHA256与sidecar一致；导出模型tensor全部与源逐位相等；严格加载通过；两个真实jaw CPU前向有限非空。参考support更新为v3，仅用于迁移回归。

二阶段baseline raw55k与训练源及旧包SHA256一致。validation中的旧GPU输出为epoch80历史结果，不能作为新版验收。新版未运行GPU级联评测。
