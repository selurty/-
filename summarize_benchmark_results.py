""" 汇总最新基准测试数据并生成论文用统计表 """

import os
import glob
import pandas as pd

RESULTS_DIR = "results"


def get_latest_csv(env_name):
    """获取指定环境下最新生成的 CSV 文件 """
    search_pattern = os.path.join(RESULTS_DIR, f"results_{env_name}_*.csv")
    files = glob.glob(search_pattern)
    if not files:
        return None
    # 按文件修改时间排序，获取最新的文件
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def get_metrics(df, alg_name):
    """计算指定算法的 Time, Length, Nodes 平均值 (仅统计成功情况) """
    alg_df = df[(df['Algorithm'] == alg_name) & (df['Success'] == 1)]
    if alg_df.empty:
        return "N/A", "N/A", "N/A"

    avg_time = round(alg_df['Time_s'].mean(), 4)
    avg_len = round(alg_df['Length_m'].mean(), 2)
    avg_nodes = round(alg_df['Nodes'].mean(), 1)

    return str(avg_time), str(avg_len), str(avg_nodes)


def get_constraint_metrics(df, alg_name):
    """计算禁飞区/限高约束场景的合规统计。"""
    alg_all = df[df['Algorithm'] == alg_name]
    if alg_all.empty:
        return "N/A", "N/A", "N/A", "N/A", "N/A"

    compliance = round(alg_all.get('Constraint_OK', alg_all['Success']).mean() * 100, 1)
    violations = round(alg_all.get('Constraint_Violations', pd.Series([0] * len(alg_all))).mean(), 1)

    alg_ok = alg_all[alg_all['Success'] == 1]
    if alg_ok.empty:
        return "N/A", "N/A", "N/A", str(compliance), str(violations)

    avg_time = round(alg_ok['Time_s'].mean(), 4)
    avg_len = round(alg_ok['Length_m'].mean(), 2)
    avg_nodes = round(alg_ok['Nodes'].mean(), 1)

    return str(avg_time), str(avg_len), str(avg_nodes), str(compliance), str(violations)


def main():
    print("正在读取最新实验数据...\n")

    csv_simple = get_latest_csv("Simple_Env")
    csv_complex = get_latest_csv("Complex_Env")
    csv_nofly = get_latest_csv("NoFly_Constraint_Env")

    if not csv_simple or not csv_complex:
        print("未找到完整的测试结果文件。请先运行批量测试，并确认 results 文件夹中已有对应环境的数据。")
        return

    df_simple = pd.read_csv(csv_simple)
    df_complex = pd.read_csv(csv_complex)

    # 算法名称映射 (左侧: CSV里的原名，右侧: 表格里的展示名)
    algorithms = [
        ("SRD-HS (Proposed)", "SRD-HS(Proposed)"),
        ("Bi-RRT*", "Bi-RRT*"),
        ("A*", "A*"),
        ("GWO", "GWO")
    ]

    # 用于保存汇总表的数据结构
    summary_data = []

    # ================= 打印控制台表格 =================
    header_line = "-" * 88
    print("表 1 本方法设计的三维航迹规划方法的定量实验结果")
    print(header_line)
    print(f"| {'环境':<18} | {'简单城市环境 (Simple_Env)':<28} | {'复杂城市环境 (Complex_Env)':<28} |")
    print(header_line)
    print(
        f"| {'算法性能指标':<16} | {'Time':<8} | {'Length':<8} | {'Nodes':<6} | {'Time':<8} | {'Length':<8} | {'Nodes':<6} |")
    print(header_line)

    for csv_name, display_name in algorithms:
        s_time, s_len, s_nodes = get_metrics(df_simple, csv_name)
        c_time, c_len, c_nodes = get_metrics(df_complex, csv_name)

        print(
            f"| {display_name:<18} | {s_time:<8} | {s_len:<8} | {s_nodes:<6} | {c_time:<8} | {c_len:<8} | {c_nodes:<6} |")

        # 存入列表用于导出CSV
        summary_data.append({
            "Algorithm": display_name,
            "Simple_Time": s_time, "Simple_Length": s_len, "Simple_Nodes": s_nodes,
            "Complex_Time": c_time, "Complex_Length": c_len, "Complex_Nodes": c_nodes
        })

    print(header_line)

    # ================= 导出为双层表头的 CSV =================
    # 使用 MultiIndex 创建类似图片中的多级表头结构
    columns = pd.MultiIndex.from_tuples([
        ("环境", "算法 \\ 性能指标"),
        ("简单城市环境", "Time"), ("简单城市环境", "Length"), ("简单城市环境", "Nodes"),
        ("复杂城市环境", "Time"), ("复杂城市环境", "Length"), ("复杂城市环境", "Nodes")
    ])

    export_df = pd.DataFrame([
        [row["Algorithm"], row["Simple_Time"], row["Simple_Length"], row["Simple_Nodes"],
         row["Complex_Time"], row["Complex_Length"], row["Complex_Nodes"]]
        for row in summary_data
    ], columns=columns)

    output_file = os.path.join(RESULTS_DIR, "summary_table_for_paper.csv")
    export_df.to_csv(output_file, index=False, encoding='utf-8-sig')  # utf-8-sig 保证 Excel 打开不乱码

    print(f"\n汇总表格已生成: {output_file}")

    if csv_nofly:
        df_nofly = pd.read_csv(csv_nofly)
        constraint_rows = []

        print("\n表 2 禁飞区/限高约束环境下的合规验证结果")
        print("-" * 94)
        print(f"| {'算法':<18} | {'Time':<8} | {'Length':<8} | {'Nodes':<8} | {'Compliance/%':<13} | {'Violations':<10} |")
        print("-" * 94)

        for csv_name, display_name in algorithms:
            time_v, len_v, nodes_v, compliance_v, violations_v = get_constraint_metrics(df_nofly, csv_name)
            print(f"| {display_name:<18} | {time_v:<8} | {len_v:<8} | {nodes_v:<8} | {compliance_v:<13} | {violations_v:<10} |")
            constraint_rows.append({
                "Algorithm": display_name,
                "Time": time_v,
                "Length": len_v,
                "Nodes": nodes_v,
                "Compliance_%": compliance_v,
                "Constraint_Violations": violations_v
            })

        print("-" * 94)

        constraint_output = os.path.join(RESULTS_DIR, "summary_constraint_table_for_paper.csv")
        pd.DataFrame(constraint_rows).to_csv(constraint_output, index=False, encoding='utf-8-sig')
        print(f"禁飞区约束汇总表已生成: {constraint_output}")

    print("提示: 可使用 Excel 打开 CSV 文件，并将内容整理为论文表格。")


if __name__ == "__main__":
    main()
