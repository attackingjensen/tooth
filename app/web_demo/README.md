# C80 牙冠网页

从包根目录配置env.local并运行`./start.sh local`。上传STL/GLB，点选基牙与两侧邻牙，估计并手工调整共享裁剪框，预览后选择修复牙位FDI并确认生成。服务器默认调用C80 SurfaceCrownNet epoch80→TRELLIS raw55k，结果保持输入world坐标。

FDI必须为恒牙11–18、21–28、31–38或41–48；非法或缺失会在API入队前拒绝。多GPU队列、访问隔离、CSRF、取消与原导出功能沿用原实现；首次启动在private/创建本机访问凭据，归档不带其他机器的凭据。

运行模型与迁移检查见包根README.md、ARCHITECTURE.md、VALIDATION.md。原候选框只按上传扫描SHA精确匹配，仍需预览和明确牙位。C80在生产人工框下存在退化，输出需人工检查。
