import numpy as np

def build_ho_hamiltonian(N=2000, x_min=-8.0, x_max=8.0,
                         hbar=1.0, m=1.0, omega=1.0):
    """
    构造一维谐振子的有限差分哈密顿矩阵
    V(x) = 1/2 m ω^2 x^2
    区间 [x_min, x_max] 上 N 个网格点
    """
    # 坐标网格
    x = np.linspace(x_min, x_max, N)
    dx = x[1] - x[0]

    # 动能算符的有限差分系数 a = ħ² / (2mΔx²)
    a = hbar**2 / (2.0 * m * dx**2)

    # 主对角：2a + V(x)
    V = 0.5 * m * omega**2 * x**2
    main_diag = 2.0 * a + V

    # 上下对角：-a
    off_diag = -a * np.ones(N - 1)

    # 构造三对角矩阵 H
    H = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    return H, x

def main():
    # ===== 数值参数，可以根据需要自己调 =====
    N = 1600          # 网格点数，越大越精细，但计算越慢
    x_min = -8.0
    x_max =  8.0
    hbar = 1.0
    m = 1.0
    omega = 1.0

    # 构造哈密顿矩阵
    H, x = build_ho_hamiltonian(N, x_min, x_max, hbar, m, omega)

    # 对称实矩阵，用 eigh
    E, psi = np.linalg.eigh(H)

    # 只取前 10 个本征值
    print("一维谐振子数值能量本征值（ħ = m = ω = 1）")
    print("n   E_num           E_exact        |E_num - E_exact|")
    print("-----------------------------------------------------")
    for n in range(10):
        E_num = E[n]
        E_exact = (n + 0.5) * hbar * omega
        err = abs(E_num - E_exact)
        print(f"{n:<2d}  {E_num: .10f}  {E_exact: .10f}  {err: .3e}")

if __name__ == "__main__":
    main()
