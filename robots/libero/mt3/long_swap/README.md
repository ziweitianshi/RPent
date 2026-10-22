# LIBERO-PRO Long Swap：MT3 轨迹库检索重放

本目录用于复现 `libero_10_swap`（Long Swap）上的 MT3 轨迹重放结果：
在官方初始状态 seed 10–49 上 **310/400（77.5%）**。

方法不调用 VLA，也不调用大模型规划。对每个回合：
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

脚本默认通过 `RPENT_MT3_SCENE_SNAPSHOT` 从 `scenes/` 恢复场景，回合可以逐个复现：在 8 个回合上核对过，和原始结果完全一致。加 `--no-scenes` 会改用官方原生初始化，成绩见上文（300/400）。

## 已知问题

- 已修复：task 9 的库里有 20 条轨迹，录制时就已经包含 +4 cm 的放置平移，以前配置里的 `ref_offset` 会再平移一次。现在库里的 `meta.placement_offset_recorded` 标出了这些轨迹，迁移时对它们不再应用 `ref_offset`。这要求数据包是带标记的新版本。
- task 9 在 dy（杯子 y − 微波炉 y）小于 0.26 m 的场景上成功率最低。剩下的失败主要是放置路线和微波炉碰撞。
- task 7 比只用人类演示时低（33 → 24/40）：生成轨迹里罐子在手中偏低几厘米，放进篮子时卡在篮沿上。
