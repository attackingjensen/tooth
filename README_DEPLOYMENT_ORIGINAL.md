# C80 → TRELLIS 二阶段交付包（2026-10-09）

默认架构已替换为 **C80 SurfaceCrownNet epoch80 armC → TRELLIS SLAT ControlNet raw55k → mesh decoder**。C80直接从裁剪jaw的64³表面占据和恒牙FDI预测牙冠表面support，不使用SS扩散、SS encoder/decoder或GT牙冠。二阶段继续采用生产40视图DINO/native jaw SLAT→精确坐标重叠映射→Euler50；保持原共享frame、原始world出口和受限补洞/Taubin后处理。

这是模型与推理链路的部署替换，使用已有C80权重，尚未进行新增裁剪增强训练。当前模型在人工框下仍存在退化，不将包验证等同于临床质量验收。架构与坐标约定见[ARCHITECTURE.md](ARCHITECTURE.md)，实际检查见[VALIDATION.md](VALIDATION.md)。

## 部署

1. 复制整个包，按[ENVIRONMENT.md](ENVIRONMENT.md)配置既有主推理与独立Open3D环境；本次仅发布两个自训练权重；推理前另需准备三个基础权重。
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
