import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def plot_arrays(arrays, x_labels, legend_labels, save_path=None):
    """
    绘制柱状图，支持自定义 x 轴编号、图例内容、配色，并导出为 SVG 图。

    参数:
    arrays: 二维列表，包含要绘制的数据。
    x_labels: 列表，x 轴的标签内容。
    legend_labels: 列表，图例的标签内容。
    save_path: 字符串，保存图片的路径（包括文件名）。如果为 None，则不保存。
    """
    # 设置图片的长宽
    plt.figure(figsize=(12, 6))  # 宽度为 12 英寸，高度为 6 英寸

    # 设置 x 轴坐标
    num_groups = len(arrays[0])  # 每组的数据点数量
    group_spacing = 30  # 增加组与组之间的间距，使 x 轴变长
    bar_width = 5  # 增加柱子的宽度
    spacing = 0.2  # 柱子之间的间距

    # 自定义颜色列表
    colors = ['#4155c6', '#ff7e79', '#2ca02c', '#ffcc53', '#8dc0ff', '#7991ff', '#e377c2']

    # 计算 x 轴的位置
    x = np.arange(num_groups) * group_spacing  # 每组之间的间距加大

    # 创建柱状图
    for i, arr in enumerate(arrays):
        # 调整每个数组的 x 坐标，使其分开
        x_positions = x + i * (bar_width + spacing)
        plt.bar(x_positions, arr, width=bar_width, label=legend_labels[i], color=colors[i])

    # 添加标题和标签
    plt.title('本次实验所有模型的结果对比')
    plt.xticks(x + (len(arrays) - 1) * (bar_width + spacing) / 2, x_labels)  # 使用自定义的 x 轴标签

    # 调整 y 轴刻度
    plt.ylim(0.6, 1)  # 设置 y 轴范围
    plt.yticks(np.arange(0.6, 1, 0.1))  # 设置 y 轴刻度密度

    # 设置图例在最上方横着排成一列
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(legend_labels), fontsize='small')

    # 显示图表
    plt.tight_layout()  # 调整布局，避免图例被裁剪

    # 如果提供了保存路径，则保存为 SVG 图
    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        print(f"图表已保存为 {save_path}")
    else:
        plt.show()

# 示例输入
arrays = [
    [0.9242243265848599, 0.91246547849864, 0.8549848942598187, 0.8533358539854947, 0.8570234113712375,
     0.8279483037156704, 0.7081370036648205],  # dual-GATGCN
    [0.9242243265848599, 0.91246547849864, 0.8549848942598187, 0.8533358539854947, 0.8570234113712375,
     0.8279483037156704, 0.7081370036648205],  # dual-GAT
    [0.9219638149868196, 0.9158080849317594, 0.8530966767371602, 0.8532670343341042, 0.8395061728395061,
     0.8573680063041765, 0.705935015610048],  # GATGCN
    [0.9181782804204232, 0.9104399384688956, 0.849320241691843, 0.8482897521130591, 0.8565573770491803,
     0.8234830575256107, 0.6976539213486748],  # dual-GCN
    [0.917289969833441, 0.9110132952403832, 0.8542296072507553, 0.8536004722417943, 0.8546184738955823, 0.8384554767533491, 0.7077347936757383],  # GAT
]

# 自定义 x 轴标签
x_labels = ['AUROC', 'AUPR', 'ACC', 'BACC', 'PREC', 'TPR', 'KAPPA']

# 自定义图例标签
legend_labels = ['dual-GAT-GCN', 'dual-GAT', 'GAT-GCN', 'dual-GCN', 'AttenSyn']

# 调用函数绘制柱状图并保存为 SVG
plot_arrays(arrays, x_labels, legend_labels, save_path='../output_chart.svg')