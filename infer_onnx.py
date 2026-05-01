import sys
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt

from matplotlib import rcParams
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTextEdit, QComboBox, QMessageBox,
    QCheckBox, QFileDialog
)

# =====================
# Matplotlib 中文配置
# =====================
rcParams['font.sans-serif'] = ['SimHei']   # Windows 常见中文字体
rcParams['axes.unicode_minus'] = False     # 正常显示负号

# 默认绘图范围与采样点数
DEFAULT_X_MIN = -4.0
DEFAULT_X_MAX =  4.0
DEFAULT_N_SAMPLES = 1000


class PinnOnnxInferGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PINN 势能与波函数 ONNX 推理（CPU / Intel 核显）")

        self.session = None
        self.input_name = None
        self.num_states = None
        self.onnx_path = None   # 由用户选择

        # 可用后端
        self.available_providers = ort.get_available_providers()
        print(">>> ONNX Runtime 可用后端：", self.available_providers)

        self._build_ui()

    # ==============
    # 构建界面
    # ==============
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 1. ONNX 模型选择
        model_layout = QHBoxLayout()
        self.model_label = QLabel("当前模型：未选择")
        btn_choose_model = QPushButton("选择 ONNX 模型")
        btn_choose_model.clicked.connect(self.choose_onnx_file)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(btn_choose_model)
        layout.addLayout(model_layout)

        # 2. 设备选择（CPU / Intel 核显）
        device_layout = QHBoxLayout()
        device_label = QLabel("推理设备：")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "Intel 核显（DirectML）"])
        self.device_combo.currentTextChanged.connect(self.on_device_changed)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        layout.addLayout(device_layout)

        # 3. 绘图范围与采样点设置
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("绘图 x 最小值："))
        self.x_min_edit = QLineEdit(str(DEFAULT_X_MIN))
        range_layout.addWidget(self.x_min_edit)

        range_layout.addWidget(QLabel("绘图 x 最大值："))
        self.x_max_edit = QLineEdit(str(DEFAULT_X_MAX))
        range_layout.addWidget(self.x_max_edit)

        range_layout.addWidget(QLabel("采样点数："))
        self.sample_edit = QLineEdit(str(DEFAULT_N_SAMPLES))
        range_layout.addWidget(self.sample_edit)

        layout.addLayout(range_layout)

        # 4. 是否把 V(0) 平移到 0
        shift_layout = QHBoxLayout()
        self.shift_checkbox = QCheckBox("将 x=0 处的势能平移到 0（V(x) ← V(x) - V(0)）")
        shift_layout.addWidget(self.shift_checkbox)
        shift_layout.addStretch()
        layout.addLayout(shift_layout)

        # 5. 单点查询 x
        x_layout = QHBoxLayout()
        x_label = QLabel("输入坐标 x：")
        self.x_edit = QLineEdit()
        self.x_edit.setPlaceholderText("例如：0.5")
        btn_calc = QPushButton("计算该点的势能与波函数")
        btn_calc.clicked.connect(self.run_single_inference)
        x_layout.addWidget(x_label)
        x_layout.addWidget(self.x_edit)
        x_layout.addWidget(btn_calc)
        layout.addLayout(x_layout)

        # 6. 结果显示
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("这里显示单点结果 / 绘图时使用的能量本征值等信息")
        layout.addWidget(self.result_text)

        # 7. 绘图按钮
        btn_plot = QPushButton("绘制势能与波函数曲线")
        btn_plot.clicked.connect(self.plot_curves)
        layout.addWidget(btn_plot)

        self.resize(800, 600)

    # ==============
    # 选择 ONNX 文件
    # ==============
    def choose_onnx_file(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "选择 ONNX 模型文件",
            "",
            "ONNX 模型 (*.onnx);;所有文件 (*)"
        )
        if not fname:
            return

        self.onnx_path = fname
        self.model_label.setText(f"当前模型：{fname}")
        # 每次选择新模型，用当前设备设置重新建 session
        self.create_session(self.device_combo.currentText())

    # ==============
    # 创建 / 切换 session
    # ==============
    def create_session(self, device_mode: str):
        if not self.onnx_path:
            # 还没有选模型
            return

        # 根据设备选项设置 providers
        if device_mode == "CPU":
            providers = ["CPUExecutionProvider"]
        elif device_mode.startswith("Intel 核显"):
            # 使用 DirectML 尝试跑在核显 / GPU 上
            if "DmlExecutionProvider" in self.available_providers:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            else:
                QMessageBox.warning(
                    self,
                    "警告",
                    "当前 ONNX Runtime 不支持 DirectML（缺少 DmlExecutionProvider），"
                    "将自动使用 CPU 执行。\n\n"
                    "建议安装 onnxruntime-directml：\n"
                    "    pip install onnxruntime-directml"
                )
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        try:
            print(f">>> 使用 providers = {providers}")
            self.session = ort.InferenceSession(
                self.onnx_path,
                providers=providers
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "错误",
                f"加载 ONNX 模型失败：\n{e}"
            )
            self.session = None
            return

        # 记录输入输出
        inps = self.session.get_inputs()
        if len(inps) != 1:
            QMessageBox.warning(
                self,
                "警告",
                f"模型输入数量不是 1，当前为 {len(inps)}，代码假设只有一个输入 x。"
            )
        self.input_name = inps[0].name

        outs = self.session.get_outputs()
        if len(outs) != 3:
            QMessageBox.warning(
                self,
                "警告",
                f"模型输出数量不是 3，当前为 {len(outs)}，代码假设输出为 V, psi, E。"
            )

        # 用一次小推理推断本征态数
        x_tmp = np.linspace(-0.1, 0.1, 3, dtype=np.float32).reshape(-1, 1)
        V_tmp, psi_tmp, E_tmp = self.session.run(
            None, {self.input_name: x_tmp}
        )
        self.num_states = psi_tmp.shape[1]
        print(f">>> 推断到本征态数量 N_state = {self.num_states}")
        print(">>> session 实际使用 providers:", self.session.get_providers())

        # 在结果框里提示一下
        self.result_text.setPlainText(
            f"已成功加载模型：{self.onnx_path}\n"
            f"本征态数量 N_state = {self.num_states}\n"
            f"当前推理设备选项：{device_mode}\n"
            f"session 实际 providers: {self.session.get_providers()}\n"
            f"可用 ONNX providers：{self.available_providers}"
        )

    def on_device_changed(self, text):
        # 切换设备时，如果已经选了模型，就重建 session
        self.create_session(text)

    # ==============
    # 解析 x 范围和采样点
    # ==============
    def get_plot_range_and_samples(self):
        try:
            x_min = float(self.x_min_edit.text().strip())
            x_max = float(self.x_max_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "警告", "x 最小值/最大值请输入实数。")
            return None, None, None

        if x_min >= x_max:
            QMessageBox.warning(self, "警告", "x 最小值必须小于最大值。")
            return None, None, None

        try:
            n_samples = int(self.sample_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, "警告", "采样点数请输入正整数。")
            return None, None, None

        if n_samples < 10:
            QMessageBox.warning(self, "警告", "采样点数太少，请至少输入 10。")
            return None, None, None

        return x_min, x_max, n_samples

    # ==============
    # 计算 x=0 时的 V(0)
    # ==============
    def get_V0(self):
        """
        用模型计算 V(0)。若失败则返回 None。
        """
        if self.session is None:
            return None

        x0 = np.array([[0.0]], dtype=np.float32)
        try:
            V0, _, _ = self.session.run(None, {self.input_name: x0})
            V0_val = float(V0[0, 0])
            return V0_val
        except Exception as e:
            QMessageBox.warning(self, "警告", f"计算 V(0) 失败：\n{e}")
            return None

    # ==============
    # 单点推理
    # ==============
    def run_single_inference(self):
        if self.session is None:
            QMessageBox.warning(self, "警告", "请先选择 ONNX 模型并加载成功。")
            return

        x_str = self.x_edit.text().strip()
        if not x_str:
            QMessageBox.warning(self, "警告", "请输入一个 x 坐标。")
            return

        try:
            x_val = float(x_str)
        except ValueError:
            QMessageBox.warning(self, "警告", "x 请输入一个实数，例如 0.5。")
            return

        x_arr = np.array([[x_val]], dtype=np.float32)

        try:
            V_pred, psi_pred, E_pred = self.session.run(
                None, {self.input_name: x_arr}
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"ONNX 推理失败：\n{e}")
            return

        V_val = float(V_pred[0, 0])
        psi_vec = psi_pred[0, :]      # [N_state]
        E_vec = E_pred                # [N_state]

        # 是否平移：V(x) ← V(x) - V(0)
        V_shift_info = ""
        if self.shift_checkbox.isChecked():
            V0 = self.get_V0()
            if V0 is not None:
                V_val_shifted = V_val - V0
                V_shift_info = (
                    f"（已平移，使 V(0)=0，对应该点平移后势能为 {V_val_shifted:.6e}）"
                )
                V_val = V_val_shifted
            else:
                V_shift_info = "（尝试平移 V(0) 失败，当前结果为未平移值）"

        # 文本输出
        lines = []
        lines.append(f"【单点查询】")
        lines.append(f"x = {x_val:.6f}")
        lines.append(f"势能 V(x) = {V_val:.6e} {V_shift_info}")
        lines.append("")
        lines.append("各本征态波函数值：")
        for i in range(self.num_states):
            lines.append(f"  第 {i} 个本征态 ψ_{i}(x) = {psi_vec[i]:.6e}")
        lines.append("")
        lines.append("当前模型的能量本征值：")
        for i in range(self.num_states):
            lines.append(f"  E_{i} = {E_vec[i]:.6e}")

        self.result_text.setPlainText("\n".join(lines))

    # ==============
    # 绘制曲线：势能 + 波函数
    # ==============
    def plot_curves(self):
        if self.session is None:
            QMessageBox.warning(self, "警告", "请先选择 ONNX 模型并加载成功。")
            return

        x_min, x_max, n_samples = self.get_plot_range_and_samples()
        if x_min is None:
            return

        x = np.linspace(x_min, x_max, n_samples, dtype=np.float32).reshape(-1, 1)

        try:
            V_pred, psi_pred, E_pred = self.session.run(
                None, {self.input_name: x}
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"ONNX 推理失败：\n{e}")
            return

        V_pred = V_pred.reshape(-1)    # [N]
        # psi_pred: [N, N_state]

        # 如果勾选了平移，则在整个曲线中做 V(x) ← V(x) - V(0)
        V_shift_info = ""
        if self.shift_checkbox.isChecked():
            # 如果绘图范围包含 0，则直接在本次 x 上找 V(0)
            if x_min <= 0.0 <= x_max:
                idx0 = np.argmin(np.abs(x.reshape(-1) - 0.0))
                V0 = V_pred[idx0]
            else:
                # 否则额外算一次 V(0)
                V0 = self.get_V0()

            if V0 is not None:
                V_pred = V_pred - V0
                V_shift_info = "（已平移，使 V(0)=0 ）"
            else:
                V_shift_info = "（尝试平移 V(0) 失败，当前曲线为未平移值）"

        # -------- 势能图 --------
        plt.figure(figsize=(8, 8))
        plt.plot(x.reshape(-1), V_pred, label="势能 V(x)")
        plt.xlabel("位置 x")
        plt.ylabel("势能 V(x)")
        plt.title(f"PINN 反演得到的势能分布 {V_shift_info}")
        plt.grid(True)
        plt.legend()

        # -------- 波函数图 --------
        plt.figure(figsize=(8, 8))
        for i in range(self.num_states):
            plt.plot(
                x.reshape(-1),
                psi_pred[:, i],
                label=f"第 {i} 个本征态 ψ_{i}(x)"
            )
        plt.xlabel("位置 x")
        plt.ylabel("波函数 ψ(x)")
        plt.title("PINN 预测的各本征态波函数")
        plt.grid(True)
        plt.legend()

        plt.show()

        # 文本框输出能量本征值
        lines = []
        lines.append(f"【绘图信息】")
        lines.append(f"x 范围：[{x_min:.3f}, {x_max:.3f}]，采样点数：{n_samples}")
        lines.append(f"势能平移：{'已将 V(0) 平移到 0' if self.shift_checkbox.isChecked() else '未平移'}")
        lines.append("")
        lines.append("当前模型的能量本征值：")
        for i in range(self.num_states):
            lines.append(f"  E_{i} = {E_pred[i]:.6e}")
        self.result_text.setPlainText("\n".join(lines))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PinnOnnxInferGUI()
    win.show()
    sys.exit(app.exec_())
