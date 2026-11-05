# RIVE：小模型在验证之下的自我改进

RIVE（Retrospective Inference and Verified Evolution）研究的问题是：语言模型能否在可验证的反馈之下改进自己，并且在固定的评测上得到可以确认的提升。

本仓库是这项研究在单张消费级显卡上的实验代码和结果。
- **硬件：** 一张 RTX 4070，12 GB 显存。
- **研究对象：** Qwen3.5-4B，以及部分实验里的 Qwen3.5-9B。

每个实验的判据都在运行之前写定。负结果也照实报告。

## 主要结果

| 结论 | 依据 |
|---|---|
| 一层改进存在 | 阶段 A 的路由改写在密封集上提高 8.2 个百分点（p = 0.0045）。T1 上四种配置版改进器各比静态智能体多解约 26 题（测试份 216 题）。 |
| 改进器之间的差别量不出来 | T1 上四种改进器平均多解 25.5–26.1 题，相互的配对差 p ≥ .86。设计上应当更好的正对照，也没有比基线好（−5.4 题，p = 0.5）。 |
| 原因是改进空间浅、测量能力弱 | 第一轮打开“按失败测试重试”之后，几乎没有改动再过门槛。最后的智能体之间，选择份分数和测试份分数的相关系数是 0.02。测出 3 题的差别需要 35–48 个种子。 |
| 改进器的事前预测不可靠 | 三个 LLM 改进器的 Brier 分数是 0.40–0.51，一直报实际留下比例的 Brier 分数是 0.19–0.23。 |
| 外部分析能修好具体的障碍 | 思考模式“想不完”时，从残差流里去掉一个方向（秩 1）。4B 在选择份上从 53 题升到 126 题。在 9B 上，解出的题数没有达到事先写定的门槛。 |
| 老师教学和自学微调没有带来提升 | DeepSeek 教学、拒绝采样自学、带思考过程的教学，在选择份上都低于原模型或在波动以内。 |
| 进行中 | 按测试结果给奖励的强化学习，训练覆盖编程、数学、逻辑和文字四个领域，按易、中、难分级评测。 |

详细的数字、判据和出处见文档：

- [`docs/technical-report.md`](docs/technical-report.md)：负结果技术报告。在这个规模上，改进器之间的差别为什么量不出来。
- [`docs/experiments.md`](docs/experiments.md)：各个实验的摘要，包括问题、事先写定的判据、结果和结论。

## 方法要点

- **事先写定的判据：** 每个实验在运行前写定读法和门槛，结果按写定的读法判。看到数据之后加的分析，都标为事后分析，不作判据。
- **题包分离：**
  - 开发份用来调试和选检查点；
  - 选择份用来判定；
  - 测试份只用来确认。
- **噪声估计：** 评测不能逐位复现（Qwen3.5 的线性注意力层不支持 vLLM 的批不变模式）。所以门槛按重复评测测得的波动来定。程序题评测份约 250 题，两次评测之差的标准差约 7.4 题，提升的门槛定为 15 题。
- **记账：** 来自外部模型或外部分析的帮助（老师的答案、方向编辑）单独记录，不算作模型的自我改进。

## 环境

- WSL2 上的 Ubuntu，Python 3.12。
- vLLM 0.32.0（FP8 或在线 MXFP4 量化），transformers 5.17，PyTorch 2.13（CUDA 13.0），bitsandbytes 0.50.2，peft。
- Docker：候选代码和判题都在断网、限资源的容器里运行。
- 数据不在仓库里，需另行下载：
  - LiveCodeBench（release v1 和 test6）
  - MBPP（sanitized）
  - MATH、MATH-500
  - BFCL v3
  - 模型权重：Qwen3.5-4B、Qwen3.5-9B（Hugging Face）

## 运行

```
python -m pytest -q                                  # 单元测试和沙箱测试（需要 Docker）
bash scripts/serve.sh                                # 4B 的 vLLM 服务（FP8）
bash scripts/serve9b.sh                              # 9B 的 vLLM 服务（MXFP4，64k 上下文）
python scripts/rive_t2.py scan                       # T2 的配置扫描
python scripts/rive_think_fix.py run t0 tb tw        # 思考截断的几种处理（6.13）
python scripts/rive_9b.py run think16k_s0            # 9B 的评测（6.23）
python scripts/watchdog.py --heartbeat <文件> -- bash scripts/rive_rl_chain.sh   # 强化学习（6.25）
python scripts/watchdog.py --heartbeat <文件> -- bash scripts/rive_rl2_chain.sh  # 强化学习第二版（6.26）
```

各脚本开头的说明写有用法和输出位置。说明里的 `docs/rive-design.md section 6.x` 指项目内部的实验记录，没有公开；编号和 [`docs/experiments.md`](docs/experiments.md) 的小节相同。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `rive/benchmark/` | 题目的划分、判题、失败分类、统计 |
| `rive/sandbox/` | 断网容器里的候选沙箱和宿主侧客户端 |
| `rive/runtime/` | 用候选的工作流和模型跑题 |
| `rive/research/` | 研究循环（诊断、假设、设计、实现、验证、复盘）和随机修改对照 |
| `rive/evolution/`、`rive/archive/` | 阶段 C 的演化控制器，以及记录全部候选和评测的 SQLite 档案 |
| `rive/meta_rsi/` | 配置版改进器、RESET-META 评测、GEPA 基线、权重编辑工具 |
| `rive/infra/` | 事件日志、运行清单、外部模型的调用网关 |
| `candidate_template/` | 阶段 A 的初始候选 |
| `scripts/` | 各个实验的脚本和运行链 |
| `tests/` | 单元测试和沙箱测试 |
| `results/` | 各个实验的汇总结果（JSON） |

## 局限

- 只有一个模型家族（Qwen3.5 的 4B 和 9B）和一张显卡，任务以 Python 编程题为主。结论不外推到别的规模和领域。
- 评测不能逐位复现。结论只针对超过门槛的差别，更小的差别这里量不出来。
- 逻辑题和文字题是用本仓库里的生成器出的，不是公认的评测集。
