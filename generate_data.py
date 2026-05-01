import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ================== matplotlib 中文字体设置 ==================
# 尽量列出几种常见中文字体，有一个存在就能正常显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QComboBox,
    QPushButton, QTextEdit, QGridLayout, QMessageBox
)


# ================== 通用数值求解函数（计算网格） ==================
def solve_schrodinger(pot_type, params):
    """
    pot_type: 'harmonic', 'slope', 'double_gaussian'
    params: 字典，包含 x_min_calc, x_max_calc, N_calc, m, hbar, 以及各势能的参数
    返回: x_full_calc, V_full_calc, E2, psi_full_calc
    - x_full_calc: (N_calc+2,) 计算网格上的坐标（含边界，边界 ψ=0）
    - V_full_calc: (N_calc+2,) 计算网格上的势能
    - E2: 前两个本征能量 [E0, E1]
    - psi_full_calc: (N_calc+2, 2) 两个本征态（边界 ψ=0）
    """

    x_min = float(params.get("x_min_calc", -5.0))
    x_max = float(params.get("x_max_calc",  5.0))
    N     = int(params.get("N_calc", 400))
    m     = float(params.get("m", 1.0))
    hbar  = float(params.get("hbar", 1.0))

    if N < 20:
        raise ValueError("计算网格点 N_calc 太小，建议 N_calc >= 100。")

    dx = (x_max - x_min) / (N + 1)
    x_inner = np.linspace(x_min + dx, x_max - dx, N)  # 内部网格点

    # 势能 V(x) on inner grid
    if pot_type == "harmonic":
        omega = float(params.get("omega", 1.0))
        V_inner = 0.5 * m * omega**2 * x_inner**2

    elif pot_type == "slope":
        k = float(params.get("k", 0.5))
        V_inner = k * x_inner

    elif pot_type == "double_gaussian":
        V0    = float(params.get("V0", 5.0))     # 深度（正数）
        a     = float(params.get("a", 1.5))      # 两个势阱中心位置 ±a
        sigma = float(params.get("sigma", 0.5))  # 宽度
        # 双高斯势阱：通常取为负势
        V_inner = -V0 * (
            np.exp(-(x_inner - a)**2 / (2 * sigma**2)) +
            np.exp(-(x_inner + a)**2 / (2 * sigma**2))
        )
    else:
        raise ValueError("未知势能类型")

    # 动能算符有限差分
    kin_const = hbar**2 / (2 * m * dx**2)
    diag = 2 * kin_const + V_inner
    off  = -kin_const * np.ones(N - 1)
    H = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)

    # 本征值 / 本征向量
    E, psi = np.linalg.eigh(H)

    # 取前两个本征态并归一化
    psi0 = psi[:, 0].copy()
    psi1 = psi[:, 1].copy()

    norm0 = np.sqrt(np.trapezoid(np.abs(psi0)**2, x_inner))
    norm1 = np.sqrt(np.trapezoid(np.abs(psi1)**2, x_inner))
    psi0 /= norm0
    psi1 /= norm1

    # 补边界点（两端 ψ=0）
    x_full = np.linspace(x_min, x_max, N + 2)
    psi_full = np.zeros((N + 2, 2))
    psi_full[1:-1, 0] = psi0
    psi_full[1:-1, 1] = psi1

    # 势能在 full 网格上的值
    if pot_type == "harmonic":
        omega = float(params.get("omega", 1.0))
        V_full = 0.5 * m * omega**2 * x_full**2
    elif pot_type == "slope":
        k = float(params.get("k", 0.5))
        V_full = k * x_full
    elif pot_type == "double_gaussian":
        V0    = float(params.get("V0", 5.0))
        a     = float(params.get("a", 1.5))
        sigma = float(params.get("sigma", 0.5))
        V_full = -V0 * (
            np.exp(-(x_full - a)**2 / (2 * sigma**2)) +
            np.exp(-(x_full + a)**2 / (2 * sigma**2))
        )
    else:
        V_full = np.zeros_like(x_full)

    return x_full, V_full, E[:2], psi_full


# ================== PyQt 主窗口 ==================
class SchrodingerWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("一维薛定谔方程数值求解器（PyQt 版）")

        central = QWidget()
        self.setCentralWidget(central)

        layout = QGridLayout(central)
        row = 0

        # 势能类型（中文）
        layout.addWidget(QLabel("势能类型:"), row, 0)
        self.pot_combo = QComboBox()
        self.pot_combo.addItems(["谐振子", "斜坡势", "双高斯势阱"])
        layout.addWidget(self.pot_combo, row, 1)
        row += 1

        # ===== 计算网格参数 =====
        layout.addWidget(QLabel("计算 x_min:"), row, 0)
        self.xmin_calc_edit = QLineEdit("-4.0")
        layout.addWidget(self.xmin_calc_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("计算 x_max:"), row, 0)
        self.xmax_calc_edit = QLineEdit("4.0")
        layout.addWidget(self.xmax_calc_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("计算网格点 N_calc:"), row, 0)
        self.N_calc_edit = QLineEdit("2000")
        layout.addWidget(self.N_calc_edit, row, 1)
        row += 1

        # ===== 输出/绘图网格参数 =====
        layout.addWidget(QLabel("输出 x_min:"), row, 0)
        self.xmin_out_edit = QLineEdit("-4.0")
        layout.addWidget(self.xmin_out_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("输出 x_max:"), row, 0)
        self.xmax_out_edit = QLineEdit("4.0")
        layout.addWidget(self.xmax_out_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("输出点数 N_out:"), row, 0)
        self.N_out_edit = QLineEdit("40")
        layout.addWidget(self.N_out_edit, row, 1)
        row += 1

        # 物理参数
        layout.addWidget(QLabel("质量 m:"), row, 0)
        self.m_edit = QLineEdit("1.0")
        layout.addWidget(self.m_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("ℏ (hbar):"), row, 0)
        self.hbar_edit = QLineEdit("1.0")
        layout.addWidget(self.hbar_edit, row, 1)
        row += 1

        # 势能参数
        layout.addWidget(QLabel("谐振子 ω:"), row, 0)
        self.omega_edit = QLineEdit("1.0")
        layout.addWidget(self.omega_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("斜坡 k:"), row, 0)
        self.k_edit = QLineEdit("1")
        layout.addWidget(self.k_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("双高斯 V0:"), row, 0)
        self.V0_edit = QLineEdit("5.0")
        layout.addWidget(self.V0_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("双高斯 a:"), row, 0)
        self.a_edit = QLineEdit("1.5")
        layout.addWidget(self.a_edit, row, 1)
        row += 1

        layout.addWidget(QLabel("双高斯 σ:"), row, 0)
        self.sigma_edit = QLineEdit("0.5")
        layout.addWidget(self.sigma_edit, row, 1)
        row += 1

        # 输出文件名
        layout.addWidget(QLabel("输出 CSV 文件名:"), row, 0)
        self.file_edit = QLineEdit("wavefunctions.csv")
        layout.addWidget(self.file_edit, row, 1)
        row += 1

        # 按钮
        self.run_button = QPushButton("计算 + 导出 + 绘图")
        self.run_button.clicked.connect(self.on_run)
        layout.addWidget(self.run_button, row, 0, 1, 2)
        row += 1

        # 状态信息
        layout.addWidget(QLabel("状态信息:"), row, 0, 1, 2)
        row += 1
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text, row, 0, 1, 2)

    def log(self, msg):
        self.info_text.append(msg)

    def _get_pot_type(self):
        """把中文选项映射为内部使用的英文标识"""
        label = self.pot_combo.currentText()
        if label == "谐振子":
            return "harmonic", label
        elif label == "斜坡势":
            return "slope", label
        elif label == "双高斯势阱":
            return "double_gaussian", label
        else:
            raise ValueError("未知的势能类型选项")

    def on_run(self):
        try:
            pot_type, pot_label = self._get_pot_type()

            # 计算网格参数
            x_min_calc = float(self.xmin_calc_edit.text())
            x_max_calc = float(self.xmax_calc_edit.text())
            N_calc     = int(self.N_calc_edit.text())

            # 输出/绘图网格参数
            x_min_out = float(self.xmin_out_edit.text())
            x_max_out = float(self.xmax_out_edit.text())
            N_out     = int(self.N_out_edit.text())

            if not (x_min_calc < x_max_calc):
                raise ValueError("计算区间需要满足 x_min_calc < x_max_calc。")
            if not (x_min_out < x_max_out):
                raise ValueError("输出区间需要满足 x_min_out < x_max_out。")
            if x_min_out < x_min_calc or x_max_out > x_max_calc:
                raise ValueError("输出区间必须在计算区间内部：\n[x_min_calc, x_max_calc] 要包含 [x_min_out, x_max_out]。")

            if N_out < 10:
                raise ValueError("输出点数 N_out 太小，建议 N_out >= 50。")

            m     = float(self.m_edit.text())
            hbar  = float(self.hbar_edit.text())
            omega = float(self.omega_edit.text())
            k     = float(self.k_edit.text())
            V0    = float(self.V0_edit.text())
            a     = float(self.a_edit.text())
            sigma = float(self.sigma_edit.text())

            # 传入计算参数
            params = dict(
                x_min_calc=x_min_calc,
                x_max_calc=x_max_calc,
                N_calc=N_calc,
                m=m,
                hbar=hbar,
                omega=omega,
                k=k,
                V0=V0,
                a=a,
                sigma=sigma,
            )

            # ===== 数值求解（在计算网格上） =====
            x_full_calc, V_full_calc, E2, psi_full_calc = solve_schrodinger(pot_type, params)

            # ===== 在输出网格上插值 =====
            x_out = np.linspace(x_min_out, x_max_out, N_out)
            # 势能和波函数在输出坐标的值
            V_out    = np.interp(x_out, x_full_calc, V_full_calc)
            psi1_out = np.interp(x_out, x_full_calc, psi_full_calc[:, 0])
            psi2_out = np.interp(x_out, x_full_calc, psi_full_calc[:, 1])

            # ===== 导出 CSV =====
            filename = self.file_edit.text().strip()
            if not filename:
                filename = "wavefunctions.csv"

            data = np.column_stack([x_out, psi1_out, psi2_out])
            header = "x,psi1,psi2"
            np.savetxt(filename, data, delimiter=",", header=header, comments="")

            # 日志
            self.info_text.clear()
            self.log(f"势能类型: {pot_label} ({pot_type})")
            self.log(f"计算区间: [{x_min_calc}, {x_max_calc}], N_calc = {N_calc}")
            self.log(f"输出区间: [{x_min_out}, {x_max_out}], N_out  = {N_out}")
            self.log(f"基态能量 E0 ≈ {E2[0]:.6f}")
            self.log(f"第一激发态 E1 ≈ {E2[1]:.6f}")
            self.log(f"数据已保存到: {filename}")

            # ===== 用 matplotlib 分两个窗口绘图 =====

            # 1）势能图
            figV, axV = plt.subplots(figsize=(8, 8))
            axV.plot(x_out, V_out, label="V(x)")
            axV.set_xlabel("位置 x")
            axV.set_ylabel("势能 V(x)")
            axV.set_title(f"{pot_label} 的势能曲线")
            axV.grid(True)
            axV.legend()
            figV.tight_layout()

            # 2）波函数图
            figPsi, axPsi = plt.subplots(figsize=(8, 8))
            axPsi.plot(x_out, psi1_out, label=f"基态 ψ1(x), E0≈{E2[0]:.3f}")
            axPsi.plot(x_out, psi2_out, label=f"第一激发态 ψ2(x), E1≈{E2[1]:.3f}")
            axPsi.set_xlabel("位置 x")
            axPsi.set_ylabel("波函数 ψ(x)")
            axPsi.set_title(f"{pot_label} 的前两个本征态波函数")
            axPsi.grid(True)
            axPsi.legend()
            figPsi.tight_layout()

            # 一次性展示所有图像窗口
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"计算失败：{e}")
            self.log(f"错误：{e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SchrodingerWindow()
    win.resize(800, 800)
    win.show()
    sys.exit(app.exec_())
