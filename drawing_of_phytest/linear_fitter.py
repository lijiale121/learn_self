import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List


class LinearFitter:
    """一次函数拟合器，用于拟合y = kx + b形式的线性函数"""

    def __init__(self):
        self.k = None  # 斜率
        self.b = None  # 截距
        self.r_squared = None  # 决定系数（拟合优度）
        self.x_data = None
        self.y_data = None

        # 设置中文字体，确保中文正常显示
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

    def fit(self, x: List[float] or np.ndarray, y: List[float] or np.ndarray) -> None:
        """
        拟合一次函数 y = kx + b

        参数:
            x: x轴数据（自变量）
            y: y轴数据（因变量）
        """
        # 转换为numpy数组
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # 检查输入数据长度是否一致
        if len(x) != len(y):
            raise ValueError("x和y的数据长度必须一致")
        if len(x) < 2:
            raise ValueError("至少需要2个数据点才能进行线性拟合")

        self.x_data = x
        self.y_data = y

        # 计算均值
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        # 计算斜率k和截距b
        numerator = np.sum((x - x_mean) * (y - y_mean))  # 分子
        denominator = np.sum((x - x_mean) ** 2)  # 分母

        self.k = numerator / denominator
        self.b = y_mean - self.k * x_mean

        # 计算决定系数R²（拟合优度）
        y_pred = self.k * x + self.b  # 直接计算拟合值
        ss_total = np.sum((y - y_mean) ** 2)  # 总平方和
        ss_residual = np.sum((y - y_pred) ** 2)  # 残差平方和
        self.r_squared = 1 - (ss_residual / ss_total)

    def get_parameters(self) -> Tuple[float, float]:
        """
        获取拟合得到的参数

        返回:
            (k, b): 斜率和截距
        """
        if self.k is None or self.b is None:
            raise RuntimeError("请先调用fit()方法进行拟合")
        return (self.k, self.b)

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
             line_legend: str,
             show_points: bool = True,
             save_path: Optional[str] = None) -> None:
        """
        绘制拟合结果可视化图（支持完全自定义图表文本）

        参数:
            title: 图表标题（必须提供）
            x_label: x轴标签（必须提供）
            y_label: y轴标签（必须提供）
            data_legend: 原始数据点的图例名称（必须提供）
            line_legend: 拟合直线的图例名称（必须提供）
            show_points: 是否显示原始数据点
            save_path: 图表保存路径（None表示不保存）
        """
        if self.k is None or self.b is None or self.x_data is None or self.y_data is None:
            raise RuntimeError("请先调用fit()方法进行拟合")

        # 创建画布
        plt.figure(figsize=(10, 6))

        # 绘制原始数据点
        if show_points:
            plt.scatter(self.x_data, self.y_data, color='blue', label=data_legend)

        # 绘制拟合直线
        x_range = np.linspace(min(self.x_data), max(self.x_data), 100)
        y_fit = self.k * x_range + self.b
        plt.plot(x_range, y_fit, color='red', label=f'{line_legend}: y = {self.k:.4f}x + {self.b:.4f}')

        # 添加标注
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.text(0.05, 0.95, f'R² = {self.r_squared:.4f}',
                 transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        # 保存图片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def __str__(self) -> str:
        """返回拟合结果的字符串表示"""
        if self.k is None or self.b is None:
            return "尚未进行拟合，请先调用fit()方法"
        return f"一次函数拟合结果: y = {self.k:.6f}x + {self.b:.6f}\n决定系数R²: {self.r_squared:.6f}"


# 示例用法（包含中文测试）
if __name__ == "__main__":
    # 生成示例数据
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    y = [2.1, 4.0, 5.9, 8.2, 10.1, 11.8, 14.2, 15.9, 18.1]  # 近似y=2x+0.1

    # 创建拟合器并拟合
    fitter = LinearFitter()
    fitter.fit(x, y)

    # 输出结果
    print(fitter)

    # 可视化（使用中文测试）
    fitter.plot(
        title="温度与反应速率关系",
        x_label="温度 (°C)",
        y_label="反应速率 (mol/L·s)",
        data_legend="实验测量点",
        line_legend="线性拟合曲线"
    )