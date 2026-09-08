"""读取基准测试 CSV 数据并生成耗时与航迹长度对比图。"""

import os
import glob
import pandas as pd
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_DIR = "results"

FONT_SIZE = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

plt.rcParams['font.size'] = FONT_SIZE
plt.rcParams['axes.labelsize'] = FONT_SIZE
plt.rcParams['xtick.labelsize'] = FONT_SIZE
plt.rcParams['ytick.labelsize'] = FONT_SIZE
plt.rcParams['legend.fontsize'] = FONT_SIZE
plt.rcParams['axes.titlesize'] = FONT_SIZE


def plot_benchmark_results():
    csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not csv_files:
        print("没有找到 CSV 文件。")
        return

    sns.set_style("whitegrid", {'grid.linestyle': '--', 'grid.alpha': 0.5})

    for file_path in csv_files:
        print(f"正在处理: {file_path}")
        df = pd.read_csv(file_path)
        df.loc[df['Success'] == 0, ['Time_s', 'Length_m']] = pd.NA

        filename = os.path.basename(file_path)
        parts = filename.split('_')
        env_name = "_".join(parts[1:-2]) if len(parts) > 2 else "Result"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        p1 = sns.lineplot(
            data=df, x='Iteration', y='Time_s', hue='Algorithm',
            linewidth=1.8, ax=axes[0], legend=True
        )
        axes[0].set_ylabel("time/s")
        axes[0].set_xlabel("iteration")
        axes[0].set_xlim(df['Iteration'].min(), df['Iteration'].max())
        axes[0].get_legend().remove()

        p2 = sns.lineplot(
            data=df, x='Iteration', y='Length_m', hue='Algorithm',
            linewidth=1.8, ax=axes[1], legend=True
        )
        axes[1].set_ylabel("length/m")
        axes[1].set_xlabel("iteration")
        axes[1].set_xlim(df['Iteration'].min(), df['Iteration'].max())
        axes[1].get_legend().remove()

        handles, labels = axes[0].get_legend_handles_labels()

        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.05), ncol=4,
                   frameon=True, fontsize=FONT_SIZE)

        sns.despine(left=False, bottom=False)
        plt.tight_layout()

        plt.subplots_adjust(top=0.88)

        save_path = os.path.join(INPUT_DIR, f"clean_plot_{env_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存: {save_path}")
        # plt.show()


if __name__ == "__main__":
    plot_benchmark_results()
