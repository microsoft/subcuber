# Cooperative FP16 LCMA 修复与评审入口

本分支在 SubCuber 历史提交 `b0672e829c49db76512096dc43d97ab1b733e02b` 上恢复完整仓库，加入实测兼容修改和累计正确性修复。上游许可证、源码和固定子模块关系保留。

评审对象为 `HopperF16InterleavedPresumCooperativeMaxFusion_4x256_4x256_OptNo`。五类根因落在四个源码文件中；这不是对仓库全部 kernel 的修复声明。

- [技术说明：现象、因果证据、逐项修改和验证限制](lcma_review/TECHNICAL_REPORT.md)
- [原版与修复版正确性表](lcma_review/results/AB正确性表.csv)
- [逐点性能和波动表](lcma_review/results/性能回归表.csv)
- [机器可读历史结果](lcma_review/results/summary.json)
- [累计正确性补丁](lcma_review/patches/lcma_full_fix.patch)、[独立兼容补丁](lcma_review/patches/compatibility.patch)
- [提交、环境与补丁指纹](lcma_review/provenance.json)、[1161 项最终源码指纹](lcma_review/source_sha256.json)

历史 H100 发布构建中，14 组冻结输入 × 两入口 × 三次，修复版 84/84 通过；原版 42/84 通过。原本正确的 A 场景维持 42/42；原本失败的 B 场景从 0/42 变为 42/42。参考为禁用 TF32 的 FP32 GEMM，判据未放宽。

性能存在代价：A 场景 prepared GPU span 延迟等权几何平均约增加 1.60%，其中 6144³ 约增加 10.37%；workspace 不变。错误输出的原版 B 耗时不作为有效性能基线。

本次仓库装配已逐项核验 1161 个文件与历史最终实测源码一致。新增便携脚本仅完成本地语法、准备状态和命令装配检查，**没有重新在 H100 上编译或运行这个完整检出**。历史结果与新构建验证应分别记录。

## 克隆与准备依赖

在短路径目录克隆，避免 Windows 超长路径；实际 CUDA 编译和 GPU 运行使用授权的 Linux 容器。

```bash
git -c core.autocrlf=false -c core.longpaths=true clone \
  --branch lcma-cooperative-fix https://github.com/li-yuelang/subcuber_debug.git
cd subcuber_debug
python3 lcma_review/prepare.py
python3 lcma_review/verify_source.py
```

`prepare.py` 初始化 `.gitmodules` 固定的 CUTLASS/GoogleTest 提交，然后幂等应用原项目自带的 `cutlass.patch`。兼容修改和正确性修改已经在本分支源码中，**不需要再次应用** `lcma_review/patches/` 下的归档补丁。脚本不执行 reset，不覆盖未知依赖修改。

CUTLASS 内四个文件的修改来自原项目依赖补丁；父仓库仍记录原 CUTLASS gitlink。因此克隆后 CUTLASS 出现 dirty 状态是预期现象，不应将其误认为依赖源码已随父仓库提交。

## 编译本次配置

历史环境为 CUDA 12.9.86、G++ 10.5、Python 3.11.16、PyTorch 2.10.0+cu129、H100。新环境版本差异应单独记录，不能假设生成完全相同的二进制。

```bash
# 不编译、不使用 GPU：打印完整构建参数并检查源码。
python3 lcma_review/build.py --nvcc /usr/local/cuda/bin/nvcc --cxx g++-10 --dry-run

# 实际编译：不运行 GPU kernel。
python3 lcma_review/build.py --nvcc /usr/local/cuda/bin/nvcc --cxx g++-10 \
  --output lcma_review/build/fixed.so
```

如果 CUDA 安装未包含 `curand_kernel.h`，使用 `--curand-include /实际路径/nvidia/curand/include`。没有擅自安装依赖或修改精度的步骤。成功构建会保存新二进制 SHA256、完整命令、编译器版本和日志。桥接文件复制自历史实验；timing 桥接增加的接口只做 host 端 initialize/batch。

## 正确性复现

先由机器管理者提供授权容器，独占一张 H100。运行器强制检查容器、显式指定的 GPU UUID、无现有计算进程，并非阻塞持有与历史实验相同的 `/tmp/yuelang_kda_gpu_activity.lock`。不要绕过隔离或与其他 GPU 工作并行。脚本不会建立 SSH 连接或启动保活。

以下命令中的 `GPU-当前授权UUID` 必须替换为实际授权设备，不能照搬历史节点信息：

```bash
# 新抽样：生成方式和判据不变，但不自动等同于历史字节级输入。
timeout 300s python3 lcma_review/check.py \
  --so lcma_review/build/fixed.so --shape 4096 4096 4096 --seed 1729 \
  --expected-gpu-uuid GPU-当前授权UUID --out lcma_review/runs/new_4096_fixed

# 使用上一次保存的同一输入与参考文件再次检查。
timeout 300s python3 lcma_review/check.py \
  --so lcma_review/build/fixed.so --shape 4096 4096 4096 --seed 1729 \
  --input-dir lcma_review/runs/new_4096_fixed \
  --expected-gpu-uuid GPU-当前授权UUID --out lcma_review/runs/replay_4096_fixed
```

每个输出目录必须是新的。运行器保存 A/B、FP32 参考、文件及数据 SHA256、逐次完整输出（按值指纹去重）、行列错误统计、数值结果和构建库指纹。输入文件可能较大，不能提交到 Git。

每次分别调用原项目 helper 和适配入口，各三次；参考输入相同。输出须有限，所有元素满足 `abs(error) <= 0.01 + 0.01*abs(reference)`，且相对 L2 不超过 `0.001`。运行失败、接口拒绝与数值不通过不能混为一类。

历史输入只发布了 [指纹清单](lcma_review/frozen_inputs.json)，没有上传数 GB 数组；持有原冻结数据时，将三个 `.npy` 和 `input_manifest.json` 放在同一目录后使用 `--input-dir`。脚本以文件名解析，不依赖旧机器绝对路径。`historical_replay` 只有三个文件指纹均匹配历史清单时才为真。随机种子相同不保证跨环境字节相同。

## 构建兼容基线用于 A/B 比较

`lcma-tested-baseline` 分支停在历史提交加模板兼容修改，尚未加入正确性修复。与修复版使用同一份依赖准备、构建参数和桥接：

```bash
git fetch origin
git worktree add ../subcuber_baseline origin/lcma-tested-baseline
python3 lcma_review/prepare.py --root ../subcuber_baseline
python3 lcma_review/verify_source.py --root ../subcuber_baseline --version original
python3 lcma_review/build.py --source-root ../subcuber_baseline --version original \
  --nvcc /usr/local/cuda/bin/nvcc --cxx g++-10 --output lcma_review/build/original.so
```

对原版也用 `check.py`，只替换 `--so` 和新输出目录，`--input-dir` 指向固定的同一组数据。原版 B 预计数值失败；非零退出后仍保留结果，不改阈值重跑为“通过”。A/B 归类仅适用于历史冻结输入，新抽样若改变结果应调查并单独记录。

本运行器用于正确性复现，不替代历史性能回归实验。历史性能使用 9 轮同节点交错配对，GPU event 跨度与完整 helper host 时间分开统计，详见技术说明和 CSV。新性能实验须先固定计划并验证正确性。

## 评审顺序与范围

1. 查看历史基线与修复分支的 diff，核对四个文件内的改动。
2. 阅读技术说明中“现象 → 预测 → 单因素验证 → 累计修复”的证据链。
3. 在完整源码上准备固定子模块并核验指纹，再编译、复现。
4. 追加邻近尺寸、其他输入分布或调度前，明确新用例、支持条件和验收标准。

当前未证明任意 shape、任意数值分布、多 stream 或其他配置正确；也没有将本配置结果推广到全部 SubCuber 实现。
