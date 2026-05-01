# PINN Quantum Potential Inversion

本项目是一个量子力学课程设计程序，用于演示如何利用物理信息神经网络（Physics-Informed Neural Networks, PINN）从少量波函数离散采样点中反演一维定态量子体系的势能函数。

项目参考论文《基于物理信息神经网络的量子势能反演》的思路：先用有限差分法生成已知势能下的波函数数据，再训练神经网络同时学习波函数 `psi(x)`、势能 `V(x)` 和能量本征值 `E`，最后将模型导出为 ONNX，用图形界面进行推理和可视化。

## 快速使用

1. 安装依赖：

```bash
pip install numpy torch matplotlib pyqt5 onnx onnxruntime openvino
```

如果需要 DirectML 推理，可额外安装：

```bash
pip install onnxruntime-directml
```

2. 生成训练数据：

```bash
python generate_data.py
```

在界面中选择势能类型和参数，点击计算并导出 `wavefunctions.csv`。

3. 训练模型：

```bash
python train_pinn.py
```

训练完成后会绘制势能和波函数，并导出 `pinn3.onnx`。

4. 使用 ONNX 模型推理：

```bash
python infer_onnx.py
```

在界面中选择 `.onnx` 模型，即可绘制反演得到的势能曲线和波函数曲线。

## 方法概述

一维定态薛定谔方程为：

```text
-1/2 * psi''(x) + V(x) * psi(x) = E * psi(x)
```

代码中取 `hbar = 1`、`m = 1`。模型主要包含三部分：

- `PsiNet`：波函数网络，输入坐标 `x`，输出某个能级的波函数 `psi_n(x)`。
- `VNet`：势能网络，输入坐标 `x`，输出未知势能 `V(x)`。
- `E_params`：每个本征态对应一个可训练能量参数 `E_n`。

训练损失由多项物理约束组成：

- PDE 残差损失：约束网络输出满足薛定谔方程。
- 数据拟合损失：约束预测波函数贴近 `wavefunctions.csv` 中的离散数据。
- 边界条件损失：约束区间边界处波函数接近 0。
- 归一化损失：约束每个波函数满足概率归一化。
- 正交性损失：约束不同本征态之间相互正交。
- 势能平滑损失：抑制反演势能出现非物理振荡。

训练采用两阶段策略：第一阶段让能量本征值自由训练得到粗略估计；随后利用内积公式重新计算能量本征值，并作为第二阶段训练的初始值，以提高反演稳定性。

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `generate_data.py` | PyQt 图形界面程序。通过有限差分法求解一维定态薛定谔方程，支持谐振子、斜坡势、双高斯势阱等势能类型，导出少量波函数采样数据到 `wavefunctions.csv`。 |
| `train_pinn.py` | 主训练脚本，使用 FP32 精度训练 PINN。读取 `wavefunctions.csv`，训练 `PsiNet`、`VNet` 和能量参数，绘制反演势能与波函数，并导出 `pinn3.onnx`。 |
| `train_pinn_bf16.py` | BF16 自动混合精度训练版本。适合在支持 BF16 的 GPU 上尝试加速，但对本项目这种小网络和二阶自动微分任务不一定更快。 |
| `train_pinn_fp16.py` | FP16 自动混合精度训练版本，使用 `GradScaler` 降低梯度下溢风险。PINN 中二阶导数对精度敏感，运行时需要观察 loss 是否出现 `nan`。 |
| `train_pinn_tf32.py` | FP32 + TF32 加速版本。保持张量和参数为 FP32，仅在 NVIDIA GPU 上允许矩阵乘法使用 TF32 加速，通常比 FP16/BF16 更稳。 |
| `infer_onnx.py` | PyQt + ONNX Runtime 推理界面。可选择 ONNX 模型，输入或绘制坐标范围，输出势能 `V(x)`、多个波函数 `psi_n(x)` 和能量本征值 `E_n`。 |
| `verify_energy_eigenvalues.py` | 用有限差分法计算一维谐振子的前若干个能量本征值，并与解析结果对比，用于验证数值求解精度。 |
| `check_openvino_devices.py` | 使用 OpenVINO 检测当前可用推理设备。 |
| `wavefunctions.csv` | 训练输入数据，由 `generate_data.py` 导出，包含坐标列和多个波函数采样列。 |
| `pinn1.onnx`、`pinn2.onnx`、`pinn3.onnx` | 已导出的 ONNX 模型文件，可被 `infer_onnx.py` 加载。 |
| `训练结果图/` | 保存训练或实验得到的势能、波函数可视化结果。 |

## 精度版本选择

- 推荐优先使用 `train_pinn.py` 或 `train_pinn_tf32.py`。PINN 训练包含二阶导数和积分约束，FP32/TF32 通常更稳定。
- `train_pinn_bf16.py` 可用于支持 BF16 的新 GPU，但不保证提速。
- `train_pinn_fp16.py` 可能更快，但数值风险更高。若 loss 出现 `nan`，建议退回 FP32/TF32。

## 论文对应关系

本项目代码对应论文中的主要模块：

- 正问题数据生成：通过有限差分构造哈密顿矩阵，求解已知势能下的波函数与能量本征值。
- 反问题模型建立：用多个波函数网络和一个共享势能网络构造 PINN。
- 物理约束构造：在损失函数中加入 PDE、数据、边界、归一化、正交性和平滑项。
- 两阶段训练：先粗略学习能量，再用内积公式修正能量初值并继续训练。
- 部署推理：将组合模型导出 ONNX，提供图形界面进行曲线绘制和单点查询。
