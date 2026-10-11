# C80 接 TRELLIS 交付验证（2026-10-09）

- 已用真实checkpoint内config严格加载C80 epoch80 armC，权重原样复制，未训练或转换权重；TRELLIS原生产raw55k strict=True全部键匹配。
- `verify.py --cpu-only`通过：包内不可变文件SHA、两个冻结旧bench人工框jaw的C80参考support逐坐标完全一致（14721/17004），XYZ/ZYX明确转换；不输入GT support。
- 网页/几何/后处理23项pytest通过，包括FDI缺失、0、19、50、分数、布尔值和字符串的拒绝、有效牙位入队、认证/CSRF/任务隔离与取消。前端app.js更改为显式传递FDI；未额外做浏览器自动化验收。
- 两个真实旧bench人工框从STL重新体素化，使用实际包内C80→jaw40视图编码→SLAT50步→mesh及后处理完整推理通过。case_001支持点14718、raw faces563400；case_002支持点17006、raw faces572852。与缓存参考support计数差3/2，来自STL导出/重新体素化和GPU浮点计算，不伪称整个CLI逐点等同于旧CPU缓存。
- 原始、处理后、粗牙冠均导出STL/GLB/OBJ，共18个出口；用共享frame核验世界bounds、面数与有限坐标，允许误差<0.0001mm，见`validation/export_validation.json`。每例另外含jaw/crown叠加GLB。
- 第一次GPU6尝试因可用状态变化仅停止了本任务推理，记录在`validation/attempt_gpu6_interrupted*`；正式两例分别在启动时空闲的GPU0/2完成，未终止他人任务。

证据：`validation/cpu_verify.log`、`web_tests.log`、`smoke_runs.json`、`smoke_runner.log`、每例run_metadata.json及export_validation.json。只读参考和测试fixture不含GT crown/support。实际CUDA kernel/renderer/codec由完整推理覆盖；目的机器仍要按README执行GPU verify及完整生成。

这些是部署功能验收，不是两病例准确度排名或16例临床质量认证。已有C80人工框退化与多组件风险仍存在；新50/20/20/10增强采样尚未训练，包内不声称增强收益。旧SS推理结果和旧验证数字不作为本包C80结果。

CPU文档复核：在本包根目录运行README中的`verify.py --cpu-only`。包清单覆盖不可变代码、权重、文档和fixture；运行日志、验证输出、独立见证和缓存不列入不可变哈希，以便目标机器复测。
