# C80 → TRELLIS 二阶段交付包（2026-10-09）

默认架构已替换为 **C80 SurfaceCrownNet v3 old-only step12320 → TRELLIS SLAT ControlNet raw55k → mesh decoder**。C80直接从裁剪jaw的64³表面占据和恒牙FDI预测牙冠表面support，不使用SS扩散、SS encoder/decoder或GT牙冠。二阶段继续采用生产40视图DINO/native jaw SLAT→精确坐标重叠映射→Euler50；保持原共享frame、原始world出口和受限补洞/Taubin后处理。

一阶段更新为 full0.7 基准、尺度抖动和中心偏移增强训练的 v3 old-only step12320。二阶段使用 baseline crown_slat_full raw55k，与旧包逐字节一致。来源与校验见 WEIGHT_UPDATE.json。新组合尚未重新进行 GPU 级联质量评测。

## 部署

1. 复制整个包，按[ENVIRONMENT.md](ENVIRONMENT.md)配置既有主推理与独立Open3D环境；权重齐全，推理不在线下载模型。
2. `cp env.example env.local`，填入目标机器解释器、GPU和端口。
3. `source env.local; "$CROWN_PYTHON" verify.py --cpu-only`：全包哈希、C80严格加载、两个真实jaw的参考support与XYZ/ZYX一致性。
4. 在空闲GPU执行 `CUDA_VISIBLE_DEVICES=0 "$CROWN_PYTHON" verify.py`，校验CUDA kernel、rasterizer及独立体素器。
5. `./start.sh local`，访问`http://服务器IP:8090`。首次运行生成自己的访问凭据；先上传、点选基牙/邻牙、预览裁剪，再选择FDI牙位并确认生成。

`./start.sh public`保留原临时公网演示入口。归档不包含本机env.local、访问码、网页历史任务或服务日志。迁移到目标硬件后仍须跑完整推理。

## CLI

先根据实际扫描估计并检查裁剪框：

```bash
source env.local
"$CROWN_PYTHON" app/prepare_jaw_frame.py --jaw /path/full.glb --anchor 18.735 -2.966 7.553 --diameter 9 --out_dir /path/prepared
```

可提供两次`--neighbor x y z`、`--normal x y z`或`--basis pca_middle`。然后显式指定修复牙位：

```bash
"$CROWN_PYTHON" app/infer_jaw_file_v2.py --jaw /path/prepared/prepared.stl --frame /path/prepared/frame.json --original_jaw /path/full.glb --fdi 16 --out_dir /path/result
```

FDI为11–18/21–28/31–38/41–48；缺失、0或非法值会拒绝，不猜测牙位。原`--ss_steps`已取消；C80为确定性一阶段，`--slat_steps`默认50，`--seed`默认1234，二阶段专属CUDA采样seed为1235。网页FDI会保存到任务并传入相同CLI。

## 输出与生产扫描

`stage1.npz`保存XYZ布局的jaw占据、表面logits和XYZ support；`stage1_coarse_world.*`是C80粗表面。`crown_raw_world.stl/glb/obj`保留二阶段原预测；`crown_smooth_world.*`为受限处理结果；`jaw_and_crown_world.glb`用于原扫描叠加。`run_metadata.json`记录C80架构、权重SHA、FDI、条件桥接、frame及几何诊断。

`production_cases/inputs`保留16例源扫描与完整性清单，旧手动框仅作为几何候选保存；不携带旧SS预测作为C80结果，不自动推断其FDI或修复目标。局部框尺度降低牙冠体素分辨率时仍会损失细节；偏移、完整牙冠覆盖和邻牙上下文需检查。

## 本机文档验收

工作目录：本交付包根目录。复用已安装解释器，不安装依赖：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 /public/model_group/z84378256/envs/3d_editing_agent/bin/python verify.py --cpu-only
```

预期退出0及`C80_BUNDLE_CPU_PASS`。完整GPU两例复测：设置`CROWN_VOXEL_PYTHON`后，以同一主解释器运行`validation/run_local_smoke.py`；脚本仅选择使用显存小于500MiB的GPU。结果在`validation/smoke_case_001/`和`smoke_case_002/`，这是旧bench人工框诊断，不是16例真实扫描质量认证。

## 2026-10-11 裁剪修正

网页、CLI与预存候选框统一full0.7：预计牙冠最长边d对应条件框边长L=d/0.7。原0.9候选保持中心、方向及预计牙冠尺寸，框边长扩大9/7，须重新预览从原扫描裁剪。旧任务须重新估框和预览。二阶段仍为旧full0.9 raw55k，裁剪修正不改变其训练规格。

## v3默认裁剪配方

默认使用显式预计牙冠bbox中心、预计最长边d、已按训练约定对齐的输入方向，L=d/0.7；无PCA重新定向、无法向中心内移。网页须填写最长边，点选后校正中心；原始扫描未经对齐时不能直接宣称与训练分布一致。咬合/PCA选项仍为显式可选启发式。

CLI：`python app/prepare_jaw_frame.py --jaw aligned.stl --center X Y Z --diameter 9 --out_dir prepared`。

该部署配方对应v3 identity档。训练的随机尺度/中心增强不在推理时随机执行；训练根据GT整冠保证一格余量的操作，在无GT部署时只能由用户预览确认。

## 两种网页生成路径

上传后选择FDI，可点击“无需点选·直接生成”：按局部输入bbox中心、最长边/0.95归一化，保持输入方向；元数据明确direct_local_input_bbox，非GT牙冠full0.7框。也可选择FDI后点选中心成框：按牙位提供尺寸初值，L=d/0.7，可调整、预览确认再生成。牙位尺寸为启发式初值，并非经过验证的定位器。
