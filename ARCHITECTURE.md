# C80 接 TRELLIS

一阶段权重`models/c80/c80_epoch80.pt`原样来自`native_surface_full_BC50_20261007/formal/C/best.pt`，epoch80、armC。代码在`app/c80_models/`；严格使用checkpoint内config实例化，strict=True加载。32/64/128三尺度主干、3个交换stage、每stage 2个残差block；jaw/template条件编码和FDI embedding。推理不给template，mask=0。回归TSDF头保留于权重/架构但不用于support，实际使用surface_head的logits≥0。没有SS扩散过程，没有SS噪声和SS预训练codec依赖。

## 共享坐标

row-vector约定：`model=(world-c)@R/L`，`world=model@R.T*L+c`。裁剪、jaw渲染、C80和SLAT均用同一frame。Open3D占据布局为XYZ；C80输入`[B,1,D,H,W]`采用ZYX，显式transpose(2,1,0)。C80输出DHW扫描后转成XYZ整数support，TRELLIS稀疏坐标为batch+XYZ。对GT/输出不独立居中或ICP。

`app/c80_runtime.py`负责转换，返回XYZ logits用于质量检查/粗网格和XYZ coords用于二阶段。阈值为≥0，与已有C80交付诊断一致；不是材料TSDF的零等值面，也不补齐、平滑或强制居中support。

## 二阶段

`app/runs/crown_slat_full/ckpts/denoiser_step0055000.pt`为原生产raw55k，不是EMA55k。`configs/crown/slat_geo_controlnet_crown_full.json`架构固定。jaw法线40视图、512渲染，local DINOv2 ViT-L/14-reg4，SLAT encoder posterior mean。原生jaw SLAT标准化后只在与C80 support精确重叠处映射，其他位置为零；不改成对所有预测位置直接编码的另一套bridge。

Euler CFG strength0，50步，8维逐点高斯噪声；反标准化后用TRELLIS mesh decoder输出模型坐标，再用同一frame回世界。后处理与旧生产包相同，原始mesh始终保留。

## 入口

- `app/infer_jaw_file_v2.py`：C80→jaw条件→SLAT→mesh与出口。
- `app/jaw_slat_condition.py`：从旧生产helper原样抽取渲染/编码函数，去掉训练数据硬编码入口。
- `app/web_demo/app.py`：显式FDI校验、任务队列及CLI传递。
- `verify.py`：包哈希、checkpoint严格加载、两个真实参考support及GPU前置检查。

模型初始化可能消耗全局随机数；二阶段使用独立Generator(seed+1)，不受其影响。C80是确定性的。模型替换不改变训练事实，尚未使用新50/20/20/10偏移采样重新训练。
