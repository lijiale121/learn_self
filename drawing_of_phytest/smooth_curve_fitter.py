import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


class SmoothCurveFitter:
    """光滑曲线拟合器，用于拟合多项式光滑曲线（y = aₙxⁿ + ... + a₁x + a₀）"""

    def __init__(self):
        self.coefficients = None  # 多项式系数（从高次到低次）
        self.degree = None  # 多项式阶数
        self.r_squared = None  # 决定系数（拟合优度）
        self.x_data = None
        self.y_data = None

        # 设置中文字体，确保中文正常显示
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    def fit(self, x: List[float] or np.ndarray, y: List[float] or np.ndarray, degree: int = 3) -> None:
        """
        拟合多项式光滑曲线

        参数:
            x: x轴数据（自变量）
            y: y轴数据（因变量）
            degree: 多项式阶数（正整数，阶数越高曲线越灵活，但可能过拟合，建议3-5阶）
        """
        # 转换为numpy数组
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # 输入验证
        if len(x) != len(y):
            raise ValueError("x和y的数据长度必须一致")
        if len(x) < 2:
            raise ValueError("至少需要2个数据点才能进行拟合")
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("多项式阶数必须是正整数")
        if degree >= len(x):
            raise ValueError(f"阶数({degree})不能大于等于数据点数量({len(x)})，否则会过拟合")

        self.x_data = x
        self.y_data = y
        self.degree = degree

        # 多项式拟合（返回系数从高次到低次）
        self.coefficients = np.polyfit(x, y, degree)

        # 计算决定系数R²（拟合优度）
        y_pred = np.polyval(self.coefficients, x)  # 计算拟合值
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)  # 总平方和
        ss_residual = np.sum((y - y_pred) ** 2)  # 残差平方和
        self.r_squared = 1 - (ss_residual / ss_total)

    def get_coefficients(self) -> np.ndarray:
        """
        获取拟合得到的多项式系数

        返回:
            系数数组（从高次项到低次项，例如[3,2,1]对应3x²+2x+1）
        """
        if self.coefficients is None:
            raise RuntimeError("请先调用fit()方法进行拟合")
        return self.coefficients

    def get_r_squared(self) -> float:
        """
        获取决定系数R²（拟合优度，越接近1拟合效果越好）

        返回:
            决定系数R²
        """
        if self.r_squared is None:
            raise RuntimeError("请先调用fit()方法进行拟合")
        return self.r_squared

    def plot(self,
             title: str,
             x_label: str,
             y_label: str,
             data_legend: str,
             curve_legend: str,
             show_points: bool = True,
             save_path: Optional[str] = None) -> None:
        """
        绘制拟合结果可视化图（支持完全自定义图表文本）

        参数:
            title: 图表标题（必须提供）
            x_label: x轴标签（必须提供）
            y_label: y轴标签（必须提供）
            data_legend: 原始数据点的图例名称（必须提供）
            curve_legend: 拟合曲线的图例名称（必须提供）
            show_points: 是否显示原始数据点
            save_path: 图表保存路径（None表示不保存）
        """
        if self.coefficients is None or self.x_data is None or self.y_data is None:
            raise RuntimeError("请先调用fit()方法进行拟合")

        # 创建画布
        plt.figure(figsize=(10, 6))

        # 绘制原始数据点
        if show_points:
            plt.scatter(self.x_data, self.y_data, color='blue', label=data_legend)

        # 生成光滑曲线的x值（更密集的点）
        x_min, x_max = np.min(self.x_data), np.max(self.x_data)
        x_smooth = np.linspace(x_min, x_max, 500)  # 500个点确保曲线光滑
        y_smooth = np.polyval(self.coefficients, x_smooth)

        # 绘制拟合曲线
        plt.plot(x_smooth, y_smooth, color='red', label=curve_legend)

        # 添加多项式公式标注（简化显示，只保留前3位小数）
        formula = "y = "
        for i, coeff in enumerate(self.coefficients):
            power = self.degree - i
            if power == 0:
                term = f"{coeff:.3f}"
            elif power == 1:
                term = f"{coeff:.3f}x"
            else:
                term = f"{coeff:.3f}x^{power}"

            if i == 0:
                formula += term  # 第一项不带符号
            else:
                formula += f" + {term}" if coeff >= 0 else f" - {abs(coeff):.3f}x^{power}"

        # 添加标注
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.text(0.05, 0.95, f'R² = {self.r_squared:.4f}\n{formula}',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8),
                 verticalalignment='top')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def __str__(self) -> str:
        """返回拟合结果的字符串表示"""
        if self.coefficients is None:
            return "尚未进行拟合，请先调用fit()方法"

        # 构建多项式公式字符串
        formula = "多项式拟合结果: "
        for i, coeff in enumerate(self.coefficients):
            power = self.degree - i
            if power == 0:
                term = f"{coeff:.6f}"
            elif power == 1:
                term = f"{coeff:.6f}x"
            else:
                term = f"{coeff:.6f}x^{power}"

            if i == 0:
                formula += term
            else:
                formula += f" + {term}" if coeff >= 0 else f" - {abs(coeff):.6f}x^{power}"

        return f"{formula}\n决定系数R²: {self.r_squared:.6f}"


# 示例用法（包含中文测试）
if __name__ == "__main__":
    # 生成示例数据（非线性关系）
    x = np.linspace(0, 10, 20)
    y = 2 * x ** 2 - 3 * x + 5 + np.random.normal(0, 5, size=20)  # 二次函数加噪声

    # 创建拟合器并拟合（使用3阶多项式）
    fitter = SmoothCurveFitter()
    fitter.fit(x, y, degree=3)  # 可根据数据特点调整阶数（3-5阶常用）

    # 输出结果
    print(fitter)

    # 可视化（使用中文测试）
    fitter.plot(
        title="浓度与时间关系曲线",
        x_label="时间 (min)",
        y_label="浓度 (mg/L)",
        data_legend="实测数据点",
        curve_legend="光滑拟合曲线"
    )