# UniAD 2.0 FFN 编译器原型

## 当前状态与范围

- 开发分支：`codex/uniad-cuda-mlir`
- 当前编译单元：UniAD 2.0 `stage2_e2e` 的一个 BEV encoder FFN。
- 静态输入：`float32 [1, 40000, 256]`，其中 `40000 = 200 × 200` 个 BEV token。
- 中间宽度：`512`；输出：`float32 [1, 40000, 256]`。
- 推理语义：`MatMul(256→512) + Bias + ReLU → MatMul(512→256) + Bias + Residual + LayerNorm`。配置中的 dropout 在推理模式下为恒等操作。
- `--layer` 可选择 encoder 的第 0～5 层；一次导出和编译只处理其中一层。

这个原型用于验证 UniAD 子图的权重导出、编译、封装、正确性和 GPU 性能测量。它**不是完整 UniAD 编译器**：BEV attention、deformable attention、tracking、mapping、motion、occupancy 和 planning 等模块仍由原有 PyTorch/MMCV 流程执行，因此本工具的结果不能作为完整 UniAD 的端到端延迟、吞吐量或精度数据。

## 两阶段实现

### 第一阶段：CUDA/HXP + cuBLAS

导出器从可信的 UniAD checkpoint 中提取指定 encoder 层的两组全连接权重、bias 和 LayerNorm 参数，转成固定布局的 NumPy 数据，并生成带 SHA-256 校验的数据包。CUDA backend 将同一 FFN 图编译成 HXP 包；运行时把两个 MatMul 明确交给 CuPy/cuBLAS，其余算子使用生成的 CUDA 调度。编译报告同时记录输入契约、后端选择、dispatch 数量和产物摘要，便于复现与审计。

这一阶段是后续优化的正确性与性能基线。基准程序会检查运行时实际选择了 `cupy-cublas`，不会把 CUDA MatMul fallback 的结果当作 cuBLAS 基线。

### 第二阶段：MLIR + cuBLAS + 融合核

MLIR backend 针对同一 FFN 生成可审查的 tensor MLIR、运行时 ABI MLIR、调用清单和 CUDA kernel 源码。当前可执行路径是经过完整性校验的 MLIR runtime-call plan，共四次 GPU 调用：

1. `gac_cublas_fc1_matmul_f32`：第一个 MatMul，调用 CuPy/cuBLAS。
2. `gac_bias_relu_inplace_f32`：融合 `Bias + ReLU`，由 NVRTC 编译 CUDA kernel。
3. `gac_cublas_fc2_matmul_f32`：第二个 MatMul，调用 CuPy/cuBLAS。
4. `gac_bias_residual_layer_norm_f32`：融合 `Bias + Residual + LayerNorm`，每行由一个 CUDA block 完成归约。

这将原始的七算子语义压缩成四次调用，并减少中间结果写回与 kernel launch。这里的 MLIR 文件描述并校验 ABI 和调用关系，实际 dispatch 由校验后的 call records 驱动；当前融合 LayerNorm kernel 由 NVRTC 编译，尚未把整段 MLIR 直接 lowering 为 NVVM/PTX。

## 两个运行环境

UniAD 与编译器依赖不同，建议严格分开环境，通过数据包交换权重和可选的 golden sample。

| 环境 | 用途 | 主要版本 |
| --- | --- | --- |
| UniAD 环境 | 读取 checkpoint、提取权重、可选生成 Torch eager golden | Python 3.9、PyTorch 2.0.1+cu118、CUDA 11.8 运行栈 |
| GAC 编译环境 | 生成 HXP/MLIR 产物并在 GPU 上执行基准 | Python 3.11、CUDA 12、CuPy/cuBLAS/NVRTC |

`requirements-compiler.txt` 固定了当前兼容的 General AI Compiler 版本。也可以不安装 GAC 包，在编译和基准命令中用 `--gac-root` 指向一个包含 `src/gac` 的 GAC checkout。导出阶段不导入 GAC，也不需要 CUDA 12 环境。

## 使用方法

以下命令均从 UniAD 仓库根目录执行。生成物默认放在已被 `.gitignore` 排除的 `artifacts/uniad_compiler/` 下。

没有 checkpoint 时，可以先生成与生产形状完全相同的确定性合成 bundle，仅用于编译和性能链路验证：

```bash
python tools/uniad_compiler/make_synthetic_bundle.py artifacts/uniad_compiler/synthetic-layer0.bundle --seed 20261011
```

1. 在 UniAD Python 3.9 环境导出第 0 层 FFN。`--with-synthetic-golden` 会额外生成确定性的 `[1,40000,256]` 输入及 Torch eager 输出，未压缩数据约 82 MiB；不需要该数据时可省略此参数。

   ```bash
   python tools/uniad_compiler/export_ffn.py ckpts/uniad_base_e2e.pth artifacts/uniad_compiler/layer0.bundle --layer 0 --with-synthetic-golden
   ```

2. 切换到 GAC Python 3.11 / CUDA 12 环境，同时生成 CUDA/HXP 和 MLIR CUDA 产物。把 `/path/to/General-AI-Compiler` 换成实际 checkout 路径。

   ```bash
   python tools/uniad_compiler/compile_ffn.py artifacts/uniad_compiler/layer0.bundle artifacts/uniad_compiler/layer0-compiled --backend both --gac-root /path/to/General-AI-Compiler
   ```

3. 在 SM86 GPU 上运行配对基准。它先校验两条路径与 golden（若数据包未带 golden，则使用 GAC NumPy reference）的输出一致，再交替两种后端的测量顺序，以减少先后顺序造成的偏差。

   ```bash
   python tools/uniad_compiler/benchmark_ffn.py artifacts/uniad_compiler/layer0.bundle --gac-root /path/to/General-AI-Compiler --device 0 --warmup 10 --iterations 20 --repeats 10 --output artifacts/uniad_compiler/layer0-benchmark.json
   ```

基准报告区分两种时间边界：resident GPU 时间保持输入、权重和 workspace 常驻显存；public `run()` 时间包含参数检查、H2D、计算、D2H 和同步。正确性阈值为 `rtol=1e-4`、`atol=1e-5`。`--repeats` 必须是偶数，才能保证两种测量顺序样本数相同。

## 性能数据边界

已经在 RTX 3080 / SM86 上使用上述 seed `20261011` 的 bundle 完成**生产形状、合成权重和合成输入**的真机验证。两个独立进程分别改变首次后端顺序，得到下列 resident GPU 中位数：

| 首次顺序 | MLIR（4 calls） | CUDA/HXP + cuBLAS（6 dispatches） | MLIR 加速比 |
| --- | ---: | ---: | ---: |
| MLIR first | 1635.330 µs | 2074.368 µs | 1.26440× |
| CUDA first | 1645.388 µs | 2075.340 µs | 1.26061× |

两条路径的最大绝对误差都是 `9.5367431640625e-7`。这是单层 FFN 的设备常驻计算时间；包含约 41 MB 输入 H2D 和约 41 MB 输出 D2H 的 public `run()` wall time约为 34 ms，不能和 resident 数字混用。原始 JSON、逐 repeat 顺序、源文件摘要和协议说明见 [`validation/uniad-compiler-rtx3080-2026-10-11`](validation/uniad-compiler-rtx3080-2026-10-11/README.md)。

当前远端仍缺少 UniAD checkpoint 和 nuScenes dataset，所以尚未完成真实权重 golden、完整 UniAD 精度或端到端速度。上表不能换算成完整 UniAD 帧率，也不是 RTX 3070 的设备实测。

## 下一步优化

- 使用 cuBLASLt epilogue 融合第一层 MatMul、bias 和 ReLU，把当前四调用路径进一步压缩为三调用路径。
- 将融合 LayerNorm 从 NVRTC CUDA 源码迁移到 MLIR GPU/NVVM lowering，并产出可缓存、可验证的 PTX。
- 通过 DLPack 接入 PyTorch tensor，并在 PyTorch current stream 上执行，避免不必要的 H2D/D2H 拷贝、额外 stream 和同步。
- 在单层 FFN 结果稳定后逐步覆盖 encoder 多层及 attention 子图，最后再讨论整网端到端性能。
