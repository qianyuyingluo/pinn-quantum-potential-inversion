import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx as torch_onnx   # 新增：导出 ONNX 用
import matplotlib.pyplot as plt
import time  # <<< 新增：用于计时

# =========================
# 0. 配置区域：可按需要修改
# =========================

DATA_FILE = "wavefunctions.csv"   # 生成好的波函数数据

# PDE 残差点 / 归一化点 / 平滑点数量
n_colloc = 10000
n_norm   = 10000
n_smooth = 10000

# 两阶段训练轮数
epochs_stage1 = 10000    # 阶段1：同时训练 ψ、V、E（E 从默认初值开始）
epochs_stage2 = 20000   # 阶段2：用数值积分得到的 E 作为初值，继续训练 ψ、V、E

# 损失权重
lambda_pde    = 20.0
lambda_data   = 8.0
lambda_norm   = 5.0
lambda_bc     = 1.0
lambda_orth   = 5.0
lambda_smooth = 1e-3   # 如果是方势阱、斜坡势阱不太光滑，可以适当减小或设为 0

# 绘图范围
V_plot_min = -4.0
V_plot_max =  4.0
psi_plot_min = -4.0
psi_plot_max =  4.0

# 阶段1 能量本征值的初始猜测
default_E_init = 0


# =========================
# 1. 设备选择
# =========================

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(">>> 使用 NVIDIA GPU (cuda)")
#elif hasattr(torch, "xpu") and torch.xpu.is_available():
    #device = torch.device("xpu")
    #print(">>> 使用 Intel XPU (xpu)")
else:
    device = torch.device("cpu")
    print(">>> 使用 CPU")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
print(">>> 使用 FP32 + TF32 加速训练")


# =========================
# 2. 读取数据
# =========================

print(f">>> 从 {DATA_FILE} 读取数据...")
data_np = np.loadtxt(DATA_FILE, delimiter=",", skiprows=1)

x_data_np   = data_np[:, 0]
psi_cols_np = data_np[:, 1:]

N_state = psi_cols_np.shape[1]
N_data  = x_data_np.shape[0]

print(f">>> 检测到 {N_state} 个本征态, 每个态有 {N_data} 个数据点")

x_data = torch.tensor(x_data_np, dtype=torch.float32, device=device).view(-1, 1)
psi_exp_list = []
for i in range(N_state):
    psi_exp = torch.tensor(psi_cols_np[:, i], dtype=torch.float32, device=device).view(-1, 1)
    psi_exp_list.append(psi_exp)

x_min, x_max = x_data_np.min(), x_data_np.max()
a_train, b_train = float(x_min), float(x_max)
print(f">>> 训练区间: [{a_train:.3f}, {b_train:.3f}]")

colloc_x = torch.linspace(a_train, b_train, n_colloc, device=device).view(-1, 1)
norm_x   = torch.linspace(a_train, b_train, n_norm,   device=device).view(-1, 1)
smooth_x = torch.linspace(a_train, b_train, n_smooth, device=device).view(-1, 1)

dx_norm = (b_train - a_train) / (n_norm - 1)


# =========================
# 3. 定义网络结构
# =========================

class PsiNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

class VNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

# 新增：把 V_net + 多个 ψ_net + E 封装成一个组合网络，方便导出 ONNX
class CombinedNet(nn.Module):
    """
    输入:
        x: [N, 1]
    输出:
        V(x):      [N, 1]
        psi_all(x):[N, N_state]  每一列是一个本征态的波函数
        E:         [N_state]     能量本征值（与 x 无关）
    """
    def __init__(self, V_net, psi_nets, E_params):
        super().__init__()
        self.V_net = V_net
        self.psi_nets = nn.ModuleList(psi_nets)

        # 把最终学到的 E_params 固定下来，作为常数输出
        with torch.no_grad():
            E_list = []
            for E in E_params:
                # E 是形状 [1] 的 Parameter，这里拉成标量再拼接
                E_cpu = E.detach().view(1).cpu()
                E_list.append(E_cpu)
            E_cat = torch.cat(E_list, dim=0)   # [N_state]
        # 注册成 Parameter（requires_grad=False），这样会被一并导出到 ONNX
        self.E = nn.Parameter(E_cat, requires_grad=False)

    def forward(self, x):
        # x: [N, 1]
        V = self.V_net(x)  # [N,1]

        psi_outputs = []
        for net in self.psi_nets:
            psi_outputs.append(net(x))   # 每个 [N,1]
        psi_all = torch.cat(psi_outputs, dim=1)  # [N, N_state]

        # E 不依赖 x，直接返回 [N_state]
        return V, psi_all, self.E


psi_nets = [PsiNet().to(device) for _ in range(N_state)]
V_net = VNet().to(device)

# 能量本征值作为可训练参数（两阶段都参与训练）
E_params = nn.ParameterList([
    nn.Parameter(torch.tensor([default_E_init], dtype=torch.float32, device=device))
    for _ in range(N_state)
])

print(">>> 阶段1: 能量本征值初始值 E_n：")
for n, E in enumerate(E_params):
    print(f"    E_{n} = {E.item():.6f}")

# 阶段1优化器：ψ、V、E 一起训练，学习率相同
params_stage1 = list(V_net.parameters())
for net in psi_nets:
    params_stage1 += list(net.parameters())
params_stage1 += list(E_params)

optimizer = optim.Adam(params_stage1, lr=1e-3)


# =========================
# 4. 用内积公式重算能量 E_n 的函数
# =========================

def compute_refined_energies(psi_nets, V_net, a, b, N_eval=1201):
    """
    用公式
        E = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
    数值积分重算每个态的能量本征值。
    """
    x = torch.linspace(a, b, N_eval, device=device).view(-1, 1)
    x.requires_grad_(True)

    V = V_net(x)
    dx = (b - a) / (N_eval - 1)

    refined_E = []

    for net in psi_nets:
        psi = net(x)

        grad_psi = torch.autograd.grad(
            psi, x,
            grad_outputs=torch.ones_like(psi),
            create_graph=True
        )[0]
        grad2_psi = torch.autograd.grad(
            grad_psi, x,
            grad_outputs=torch.ones_like(grad_psi),
            create_graph=True
        )[0]

        Hpsi = -0.5 * grad2_psi + V * psi

        num = torch.sum(psi * Hpsi) * dx
        den = torch.sum(psi * psi) * dx
        E_est = (num / den).detach()
        refined_E.append(E_est)

    return refined_E


# =========================
# 5. 单轮训练函数（复用阶段1/阶段2）
# =========================

def train_one_epoch(epoch, phase_name, start_time):
    """
    增加了 start_time：用于在每 1000 个 epoch 打印累计耗时
    """
    global optimizer

    optimizer.zero_grad()

    # PDE 残差
    colloc_x.requires_grad_(True)
    V_colloc = V_net(colloc_x)
    loss_pde = 0.0

    for n in range(N_state):
        psi_n = psi_nets[n](colloc_x)

        grad_psi = torch.autograd.grad(
            psi_n, colloc_x,
            grad_outputs=torch.ones_like(psi_n),
            create_graph=True
        )[0]
        grad2_psi = torch.autograd.grad(
            grad_psi, colloc_x,
            grad_outputs=torch.ones_like(grad_psi),
            create_graph=True
        )[0]

        E_n = E_params[n]  # 阶段1/2 都用这个
        res_n = -0.5 * grad2_psi + V_colloc * psi_n - E_n * psi_n
        loss_pde += torch.mean(res_n**2)

    # 数据拟合
    loss_data = 0.0
    for n in range(N_state):
        psi_pred_data = psi_nets[n](x_data)
        psi_exp = psi_exp_list[n]
        loss_data += torch.mean((psi_pred_data - psi_exp)**2)

    # 边界条件 ψ(a)=ψ(b)=0
    xa = torch.tensor([[a_train]], dtype=torch.float32, device=device)
    xb = torch.tensor([[b_train]], dtype=torch.float32, device=device)

    loss_bc = 0.0
    for net in psi_nets:
        psi_a = net(xa)
        psi_b = net(xb)
        loss_bc += psi_a.pow(2) + psi_b.pow(2)

    # 归一化
    psi_norm_list = []
    loss_norm = 0.0
    for net in psi_nets:
        psi_norm = net(norm_x).squeeze()
        psi_norm_list.append(psi_norm)
        integral = torch.sum(psi_norm**2) * dx_norm
        loss_norm += (integral - 1.0)**2

    # 正交性
    loss_orth = 0.0
    if N_state >= 2:
        for i in range(N_state):
            for j in range(i+1, N_state):
                inner_ij = torch.sum(psi_norm_list[i] * psi_norm_list[j]) * dx_norm
                loss_orth += inner_ij**2

    # 势能平滑
    smooth_x.requires_grad_(True)
    V_smooth = V_net(smooth_x)
    grad_V = torch.autograd.grad(
        V_smooth, smooth_x,
        grad_outputs=torch.ones_like(V_smooth),
        create_graph=True
    )[0]
    loss_smooth_val = torch.mean(grad_V**2)

    loss = (lambda_pde    * loss_pde
          + lambda_data   * loss_data
          + lambda_norm   * loss_norm
          + lambda_bc     * loss_bc
          + lambda_orth   * loss_orth
          + lambda_smooth * loss_smooth_val)

    loss.backward()
    optimizer.step()

    # 每 1000 个 epoch（以及第 1 个）打印一次损失和用时
    if epoch % 1000 == 0 or epoch == 1:
        elapsed = time.time() - start_time
        E_values = ", ".join([f"E_{i}={E_params[i].item():.3f}" for i in range(N_state)])
        print(
            f"[{phase_name}] Epoch {epoch:5d} | "
            f"Total={loss.item():.3e} | "
            f"PDE={loss_pde.item():.3e} | "
            f"Data={loss_data.item():.3e} | "
            f"Norm={loss_norm.item():.3e} | "
            f"Orth={loss_orth.item():.3e} | "
            f"Time={elapsed:7.2f}s | "
            f"{E_values}"
        )


# =========================
# 6. 阶段 1：自动学习 E
# =========================

print("\n========== 阶段 1：自动学习 E（粗略） ==========\n")
start_time_stage1 = time.time()  # <<< 记录阶段1开始时间
for epoch in range(1, epochs_stage1 + 1):
    train_one_epoch(epoch, phase_name="Stage1", start_time=start_time_stage1)

print("\n>>> 阶段1结束，当前（自动学到的）E_n：")
for n, E in enumerate(E_params):
    print(f"    E_{n} (learned) = {E.item():.6f}")


# =========================
# 7. 内积公式重算能量，作为第二阶段的初始值
# =========================

print("\n>>> 使用内积公式重算能量本征值（refined E）...")
refined_E = compute_refined_energies(psi_nets, V_net, a_train, b_train, N_eval=1601)

print(">>> 重算得到的 E_n（refined）：")
for n, E in enumerate(refined_E):
    print(f"    E_{n} (refined) = {E.item():.6f}")

print("\n>>> 用重算的 E_n 覆盖 E_params，作为阶段2的初始值（继续训练 E）")
for n, E in enumerate(refined_E):
    E_params[n].data = E.clone().view_as(E_params[n].data)
    E_params[n].requires_grad = True  # 第二阶段继续训练能量本征值


# =========================
# 8. 阶段 2：用 refined E 作为初始值，继续训练 ψ、V、E
# =========================

# 第二阶段重新设置优化器：ψ、V 学习率较大，E 学习率小一点
optimizer = optim.Adam([
    {"params": V_net.parameters(), "lr": 1e-3},
    {"params": [p for net in psi_nets for p in net.parameters()], "lr": 1e-3},
    {"params": E_params, "lr": 1e-4},   # E 继续训练，但步子小一些
])

print("\n========== 阶段 2：固定初始 E（来自积分），继续训练 ψ、V、E ==========\n")
start_time_stage2 = time.time()  # <<< 记录阶段2开始时间
for epoch in range(1, epochs_stage2 + 1):
    train_one_epoch(epoch, phase_name="Stage2", start_time=start_time_stage2)

print("\n>>> 阶段2结束，最终 E_n：")
for n, E in enumerate(E_params):
    print(f"    E_{n} (final) = {E.item():.6f}")


# =========================
# 9. 画最终 ψ 和 V（只画 NN 推理结果）
# =========================

for net in psi_nets:
    net.eval()
V_net.eval()

grid_x = np.linspace(a_train, b_train, 400)
grid_x_tensor = torch.tensor(grid_x, dtype=torch.float32, device=device).view(-1, 1)

with torch.no_grad():
    V_pred_full = V_net(grid_x_tensor).cpu().numpy().flatten()
    psi_pred_full_list = []
    for net in psi_nets:
        psi_pred_full = net(grid_x_tensor).cpu().numpy().flatten()
        psi_pred_full_list.append(psi_pred_full)

# 势能：按 V_plot_min/max 截取
mask_V = (grid_x >= V_plot_min) & (grid_x <= V_plot_max)
x_V = grid_x[mask_V]
V_plot = V_pred_full[mask_V]

plt.figure(figsize=(8, 6))
plt.plot(x_V, V_plot, label="Recovered V(x)")
plt.xlabel("x")
plt.ylabel("V(x)")
plt.title(f"Recovered Potential, x ∈ [{V_plot_min}, {V_plot_max}]")
plt.legend()
plt.grid(True)

# 波函数：按 psi_plot_min/max 截取
mask_psi = (grid_x >= psi_plot_min) & (grid_x <= psi_plot_max)
x_psi = grid_x[mask_psi]

for n in range(N_state):
    psi_full = psi_pred_full_list[n]
    psi_plot = psi_full[mask_psi]

    plt.figure(figsize=(8, 6))
    plt.plot(x_psi, psi_plot, label=f"Predicted psi{n}(x)")
    plt.xlabel("x")
    plt.ylabel(f"psi{n}(x)")
    plt.title(f"Predicted Wavefunction state {n}, x ∈ [{psi_plot_min}, {psi_plot_max}]")
    plt.legend()
    plt.grid(True)

plt.show()


# =========================
# 10. 导出 ONNX 模型（输入 x，输出 V(x)、psi(x)、E）
# =========================

print("\n>>> 正在导出 ONNX 模型...")

# 先把网络全部搬到 CPU，避免 ONNX 导出时出现 device 冲突
for net in psi_nets:
    net.cpu()
V_net.cpu()

# 组合模型：V_net + psi_nets + E
combined_model = CombinedNet(V_net, psi_nets, E_params)
combined_model.eval()

# 准备一个示例输入（只用来建图，长度随便）
dummy_x = torch.linspace(a_train, b_train, steps=10).view(-1, 1)  # [10,1] on CPU

onnx_filename = "pinn3.onnx"

torch_onnx.export(
    combined_model,
    dummy_x,
    onnx_filename,
    input_names=["x"],              # 用户输入名
    output_names=["V", "psi", "E"], # 势能、波函数、能量本征值
    dynamic_axes={
        "x":   {0: "N"},   # 第 0 维长度可变：N 个坐标点
        "V":   {0: "N"},
        "psi": {0: "N"},
        # E 不依赖 N，是 [N_state]，不需要 dynamic_axes
    },
    opset_version=17
)

print(f">>> 已导出 ONNX 模型到 {onnx_filename}")
print(">>> ONNX 输入：x 形状 [N,1]；输出：V[N,1], psi[N,N_state], E[N_state]")
