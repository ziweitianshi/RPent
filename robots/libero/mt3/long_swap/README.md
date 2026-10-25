# LIBERO-PRO Long Swap：MT3 轨迹库检索重放

本目录用于复现 `libero_10_swap`（Long Swap）上的 MT3 轨迹重放结果：
保留真值位姿 + 场景快照的旧版复现（官方 seed 10–49：**310/400**），并新增[严格感知版本](#严格感知版本统一配置--v2r)（官方 seed 0–9）。

旧版真值位姿方法不调用 VLA，也不调用大模型规划。对每个回合：
1. 用模拟器真值物体位姿，从轨迹库里检索场景最近的一条成功轨迹；
2. 按交互把轨迹切成段，每段按参考物体的位姿变化（x、y、z、偏航）迁移到当前场景；
3. 对齐到瓶颈点，沿演示的接近通道进入，再闭环跟踪迁移后的目标位姿。

## 结果

seed 10–49，每任务 40 个回合，单回合，上限 10000 步，场景从快照恢复：

| 任务 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 成功 / 40 | 32 | 36 | 34 | 37 | 30 | 40 | 38 | 24 | 33 | 6 | 310 / 400 |

seed 10–49 在方法定型前没有用来调参。排行榜用的是 seed 0–9，可以用 `--seeds 0-9` 运行。

不恢复场景快照（`--no-scenes`，官方原生初始化）时约为 **308/400（77.0%）**：task 0–8 是 299/360，逐任务 32 / 35 / 33 / 36 / 30 / 40 / 38 / 24 / 31；task 9 是 9/40。
原生初始化下，同一个 seed 的固定装置位置在不同进程里会有差别（见"场景快照"），所以这个数字每次跑会有几个回合的波动。

task 9 的数字是修复"重复平移"之后的结果（见"已知问题"）。修复前是快照 5/40、原生初始化 1/40，总分分别是 309 和 300。

## 信息条件（引用结果时请一并说明）

- 使用模拟器的**真值物体位姿**；task 9 的路线选择还使用了模拟器的**真值碰撞几何**。排行榜上的方法使用相机图像，两者不能直接比排名。
- 轨迹库由两部分组成：
  - 官方 LIBERO 人类演示，录制于原布局；
  - 在**新采样**的 swap 场景上试错、并经过重放验证的成功轨迹。新场景和 seed 0–49 的初始状态不重叠，最小差 ≥ 1.3 cm。
- task 9 带有针对该任务的人工设定：抓取绕手指闭合轴倾斜、放置段沿微波炉 y 方向平移 4 cm、关门段多推 4 cm、门段对齐多重规划 2 次。

## 文件

| 路径 | 内容 |
|---|---|
| `configs/tasks_0_8/mt3.json` | task 0–8：检索 + 20 cm 接近通道 |
| `configs/task_9/mt3.json` | task 9：在上面的基础上，加最小干预选路、抓取倾斜、抓取补偿、放置平移和关门修正 |
| `run_long_swap.py` | 并行运行，汇总到 `<out>/results.json` |
| 数据包 `long_swap_data.tar.gz`（从 [Release 10.22](https://github.com/ziweitianshi/RPent/releases/tag/10.22) 下载） | `library/libero_10/task_XX.npz`：每个任务的轨迹库（task 9 带 `placement_offset_recorded` 标记）；`scenes/libero_10_swap_t{task}_s{seed}.json`：500 个场景快照 |

## 运行

1. 按 [LIBERO 使用文档](../../../../docs/source-zh/rst_source/usage/libero.rst) 安装 RPent 的 LIBERO 依赖（LIBERO-PRO、robosuite、MuJoCo、EGL）。不需要 VLA 和 SAM3 checkpoint。
2. 下载并解压数据包（72 MB），例如：

```bash
curl -LO https://github.com/ziweitianshi/RPent/releases/download/10.22/long_swap_data.tar.gz
echo "5a555fb5410495bbb5d795ffbca513f86d97f1d394c90f3442ba38a990682a76  long_swap_data.tar.gz" | sha256sum -c -
tar xzf long_swap_data.tar.gz   # 得到 long_swap_data/library 和 long_swap_data/scenes
```
3. 在仓库根目录运行：

```bash
python robots/libero/mt3/long_swap/run_long_swap.py \
  --out runs/long_swap_s10_49 --data $PWD/long_swap_data \
  --seeds 10-49 --procs 40 --egl-devices 0
```

- 每个回合约占 1–2 个 CPU 核，GPU 只用来做 EGL 渲染。在 128 核的机器上，并发超过约 80 时，环境服务启动会超时。
- 跑完 400 个回合约需 40–60 分钟。task 9 每个回合需要 5–8 分钟，主要花在路线候选的碰撞检查上。
- 运行时会在本目录创建软链接 `data`，指向 `--data`。配置里的库路径 `../../data/library` 就是相对于它的。

## 场景快照

`RPENT_MT3_FIXTURE_SEED_POLICY=rpent`（官方初始化）下，LIBERO 摆放固定装置时用的是 NumPy 的全局随机数，**同一个 seed 在不同进程里，固定装置的位置可能不同**。

脚本默认通过 `RPENT_MT3_SCENE_SNAPSHOT` 从 `scenes/` 恢复场景，回合可以逐个复现：在 8 个回合上核对过，和原始结果完全一致。加 `--no-scenes` 会改用官方原生初始化，成绩见上文（约 308/400）。

## 已知问题

- 已修复：task 9 的库里有 20 条轨迹，录制时就已经包含 +4 cm 的放置平移，以前配置里的 `ref_offset` 会再平移一次。现在库里的 `meta.placement_offset_recorded` 标出了这些轨迹，迁移时对它们不再应用 `ref_offset`。这要求数据包是带标记的新版本。
- task 9 在 dy（杯子 y − 微波炉 y）小于 0.26 m 的场景上成功率最低。剩下的失败主要是放置路线和微波炉碰撞。
- task 7 比只用人类演示时低（33 → 24/40）：生成轨迹里罐子在手中偏低几厘米，放进篮子时卡在篮沿上。

## 严格感知版本：统一配置 + v2r

该版本所有任务使用同一份算法配置和同一个 v2r 库，`align_step=0.005`，不启用单任务插入技能、ref_offset、选路检查、CP-Gen 或手内校正。相机图像由 Astra（gpt-6-astra）识别物体，SAM3 + 深度反投影 / 配准得到位姿；每回合调用 Astra 一次，使用外部和腕部相机，起始物体估计冻结，活动部件重新配准，同一固定装置共享配准。运行控制不读取仿真物体真值；末尾松手。代码基线为 `73b92f1`。

### 感知评测结果

LIBERO-PRO `libero_10_swap`，官方 seed 0–9 × t0–t9，每回合上限 10000 步，原生初始化。seed 0–9 在开发中已多次使用，不能作为未见测试集。

| 评测口径 | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | 合计 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 步长 0.01，完整评测 | 7 | 7 | 7 | 7 | 9 | 9 | 8 | 6 | 5 | 4 | 69/100 |
| 步长 0.005，完整评测 | 5 | 7 | 10 | 8 | 10 | 10 | 8 | 8 | 7 | 0 | **73/100** |
| 步长 0.005，仅 t0/t9 重复评测 | 7 | — | — | — | — | — | — | — | — | 3 | 10/20 |
| **同配置多次评测，逐任务取最好** | **7** | **7** | **10** | **8** | **10** | **10** | **8** | **8** | **7** | **3** | **78/100** |

78/100 是从两次相同 0.005 配置的评测中，t0/t9 取重复评测、其余取完整评测，属于逐任务选择后的拼合成绩，**不是单次完整评测结果，也不是逐 seed 取最好**。逐回合结果和来源保存在 [results/](results/)，选取规则见 [strict_align005_best_per_task.json](results/strict_align005_best_per_task.json)。

73/100 的 100 个有效回合均正常结束；其中 22 个受模型认证/额度故障影响的回合在恢复登录后单独补跑，替换无效记录；一次 Astra JSON 解析失败仍得到任务成功。t0/t9 重复评测的 20 回合全部正常结束，Astra 无认证或解析错误。相同 seed 的固定装置仍可能不同，感知输出也有波动，因此这些数字不保证逐回合复现，也不能直接证明两个步长存在稳定优劣。旧版 310/400 使用真值位姿和快照，信息条件及 seed 范围均不同。

### 感知数据包与依赖

额外需要 SAM3 checkpoint、SAM3 服务、可调用 gpt-6-astra 的 Codex 登录，以及 `long_swap_perception_data.tar.gz`。该包包含 `library/`（v2r）、`perception.json`、点云模型和配准参考，约 235 MiB；不包含 SAM3 权重。SHA256：

```text
553931462dee19e334407d7eefabc3360b7ebf6e9f470becde15c978b1a6d8ea
```

**此感知数据包尚未上传 Release。** Release 10.22 的 `long_swap_data.tar.gz` 是旧真值评测的数据，不能替代感知数据包。拿到感知数据包后在仓库根目录解压：

```bash
echo "553931462dee19e334407d7eefabc3360b7ebf6e9f470becde15c978b1a6d8ea  long_swap_perception_data.tar.gz" | sha256sum -c -
tar xzf long_swap_perception_data.tar.gz
```

按 [LIBERO 文档](../../../../docs/source-zh/rst_source/usage/libero.rst) 安装依赖；本版本还使用 SciPy / Open3D 点云配准和 Codex CLI。感知配准还需要原 MT3 仓库及其 PointNet 权重（按该仓库安装依赖和权重）：

```bash
git clone https://github.com/kamil-dreczkowski/learning_thousand_tasks.git
git -C learning_thousand_tasks checkout a4653d5
git -C learning_thousand_tasks apply --unidiff-zero "$PWD/robots/libero/mt3/long_swap/dependencies/learning_thousand_tasks_libero.patch"
export RPENT_MT3_ORIG_REPO="$PWD/learning_thousand_tasks"
```

补丁对应本次评测使用的 `libero-adapt` 提交 `658dae0`：兼容 HWC 图像、小掩码和当前 PyTorch 权重加载方式。模型权重不包含在感知数据包中。

先在当前用户环境中确认 Codex 登录有效，然后在单独终端启动 SAM3：

```bash
export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
python -m rpent.robots.components.sam3_server \
  --cuda-device 3 --transport http --host 127.0.0.1 --port 8114
```

看到 `RPC server listening` 后运行：

```bash
python robots/libero/mt3/long_swap/run_perception.py \
  --out runs/long_swap_strict_align005 \
  --data "$PWD/long_swap_perception_data" \
  --tasks 0-9 --seeds 0-9 --procs 50 --egl-devices 0,2,3 \
  --sam3-endpoint http://127.0.0.1:8114
```

`--align-step 0.01` 可用于对照；`--tasks 0,9` 可仅重测两个任务。运行器将数据相对路径展开到各任务配置，保存命令、SHA、日志及逐步更新的 `results.json`。输出目录必须为空；结果中的 `locator_failed` 标记模型调用/解析异常，外部认证或额度错误应先排除再解读算法效果。Numba 缓存关闭方式与原评测一致，通过该入口的进程局部启动模块设置，不修改系统 robosuite 安装。
