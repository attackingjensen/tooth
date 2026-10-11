# 迁移环境清单（独立文档）

## 硬件与系统
- Linux x86_64；当前验证硬件 NVIDIA L20X，驱动 570.172.08。
- NVIDIA CUDA GPU，建议至少 48 GB 显存；调度器默认要求空闲显存 ≥35000 MiB。48 GB 是部署建议，不是已完成的最低显存测试。
- 建议主机内存 32 GB 以上；磁盘至少 25 GB（环境另计）。
- Python 3.10；CUDA Toolkit 12.4 / nvcc；GCC/G++、make、ninja、git。
- 系统动态库：libGL.so.1、libglib-2.0.so.0、libgomp.so.1。例如 Debian/Ubuntu 安装 libgl1 libglib2.0-0 libgomp1 build-essential。

## 两个 Python 环境
1. 主推理环境：PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、flash-attn 2.7.3、spconv-cu124 2.3.8；其余见 environment/requirements-runtime.txt。
2. 专用体素化环境：Python 3.10、numpy 1.26.4、Open3D 0.19.0，见 environment/requirements-voxel.txt。不要合并：旧主环境的 Open3D 体素化曾段错误。
主推理环境中的Open3D 0.17.0仅用于补洞的点到三角面距离查询，另需scipy、scikit-image；体素化仍通过Open3D 0.19.0独立解释器执行。
3. nvdiffrast 和 utils3d 的实际源码已放 third_party/；nvdiffrast 需在目标 CUDA 环境编译。

## 安装参考（已有 Python 3.10 和 CUDA Toolkit 12.4）
```bash
python3.10 -m venv /opt/crown-runtime
python3.10 -m venv /opt/crown-voxel
/opt/crown-runtime/bin/python -m pip install --upgrade pip setuptools wheel
/opt/crown-runtime/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
/opt/crown-runtime/bin/python -m pip install packaging ninja
/opt/crown-runtime/bin/python -m pip install flash-attn==2.7.3 --no-build-isolation
/opt/crown-runtime/bin/python -m pip install spconv-cu124==2.3.8
/opt/crown-runtime/bin/python -m pip install -r environment/requirements-runtime.txt
/opt/crown-runtime/bin/python -m pip install --no-build-isolation ./third_party/nvdiffrast ./third_party/utils3d
/opt/crown-voxel/bin/python -m pip install -r environment/requirements-voxel.txt
```
安装需要可访问软件包源；本次仅发布两个自训练权重，运行前需另行准备三个上游基础权重。若更换 PyTorch/CUDA/GPU 架构，CUDA 扩展需重新编译并验收。
完整旧环境快照 environment/*freeze-reference.txt 仅供排查，含旧路径与不相关软件，不应直接整体 pip install。

## 网络
内网：开放 TCP 8090；公网固定入口：域名 + HTTPS 反向代理/VPN。
临时外网演示：已附 Linux x86_64 cloudflared，运行 ./start.sh public，需要服务器能连接 Cloudflare；临时域名会变，不能视为固定生产域名。
