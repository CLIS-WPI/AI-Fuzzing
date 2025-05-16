# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from collections import Counter
import ast
import os

# تنظیمات
INPUT_CSV = "fuzzing_results_v23_fixed_index_error.csv"  # Updated to match simulation output
OUTPUT_REPORT = "analysis_report.txt"
QOS_SINR_THRESHOLD = 2.0

def safe_eval(x):
    """تبدیل ایمن رشته‌های لیست به لیست‌های پایتون"""
    try:
        return ast.literal_eval(x) if isinstance(x, str) else x
    except (ValueError, SyntaxError):
        return []

def parse_vulnerabilities(vuln_str):
    """تجزیه آسیب‌پذیری‌ها و استخراج تعداد و UEهای Ping-Pong"""
    vulnerabilities = safe_eval(vuln_str)
    vuln_counts = Counter()
    ping_pong_ues = set()
    for vuln in vulnerabilities:
        vuln_type = vuln.split(":")[0]
        vuln_counts[vuln_type] += 1
        if vuln_type == "Ping-Pong":
            try:
                ue_str = vuln.split("UEs: ")[1].strip(")")
                ues = safe_eval(ue_str)
                ping_pong_ues.update(ues)
            except (IndexError, ValueError):
                continue
    return vuln_counts, ping_pong_ues

def analyze_ue_positions(df):
    """تحلیل موقعیت UEها برای شناسایی UEهای نزدیک لبه‌های سلول"""
    report = ["\n--- UE Position Analysis ---"]
    bs_pos = np.array([[0, 0], [100, 0], [50, 86.6], [150, 86.6], [100, 43.3]])
    edge_ues = {}
    for scenario in df['scenario'].unique():
        edge_ues[scenario] = []
        report.append(f"\nScenario: {scenario}")
        scenario_df = df[df['scenario'] == scenario]
        for idx, row in scenario_df.iterrows():
            try:
                ue_locs = safe_eval(row['ue_locations_str'])
                for ue_idx, loc in enumerate(ue_locs):
                    distances = np.linalg.norm(np.array(loc) - bs_pos, axis=1)
                    min_distance = np.min(distances)
                    if min_distance > 100:  # Updated for 100m radius
                        report.append(
                            f"Iter {row['iteration']}, UE {ue_idx}: Distance {min_distance:.2f}m (Edge)"
                        )
                        edge_ues[scenario].append((ue_idx, min_distance))
            except Exception as e:
                report.append(f"Error processing UE positions in iter {row['iteration']}: {e}")
                continue
        if edge_ues[scenario]:
            report.append(f"Summary: {len(edge_ues[scenario])} edge UEs detected.")
            edge_ues_ids = sorted(set(ue_id for ue_id, _ in edge_ues[scenario]))
            edge_ues_counts = Counter(ue_id for ue_id, _ in edge_ues[scenario])
            report.append(f"Edge UE IDs: {edge_ues_ids}")
            report.append(f"Edge UEs Frequency: {dict(edge_ues_counts)}")
        else:
            report.append("Summary: No edge UEs detected.")
    return report, edge_ues

def analyze_csv():
    """تحلیل فایل CSV و تولید گزارش جامع"""
    if not os.path.exists(INPUT_CSV):
        return f"Error: File {INPUT_CSV} not found."

    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        return f"Error: Failed to read {INPUT_CSV}: {e}"

    report = []

    # 1. آمار کلی
    report.append("=== General Statistics ===")
    report.append(f"Total Rows: {len(df)}")
    report.append(f"Scenarios: {', '.join(df['scenario'].unique())}")
    report.append(f"Fuzzer Types: {', '.join(df['fuzzer_type'].unique())}")
    report.append(f"Algorithms: {', '.join(df['algorithm'].unique())}")
    report.append("")

    key_metrics = [
        'avg_overall_sinr', 'avg_high_prio_sinr', 'fairness_index',
        'handover_count_iter', 'num_ues_below_qos', 'sinr_5th_percentile',
        'sinr_50th_percentile', 'sinr_95th_percentile', 'load_min', 'load_max', 'load_std'
    ]
    report.append("Key Metrics Summary:")
    for metric in key_metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                mean_val = values.mean()
                median_val = values.median()
                min_val = values.min()
                max_val = values.max()
                report.append(
                    f"  {metric.replace('_', ' ').title()}: "
                    f"Mean={mean_val:.2f}, Median={median_val:.2f}, Min={min_val:.2f}, Max={max_val:.2f}"
                )
            else:
                report.append(f"  {metric.replace('_', ' ').title()}: No valid data")
    report.append("")

    # 2. تحلیل آسیب‌پذیری‌ها
    report.append("=== Vulnerability Analysis ===")
    all_vuln_counts = Counter()
    all_ping_pong_ues = set()
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        report.append(f"\nScenario: {scenario}")
        for fuzzer in scenario_df['fuzzer_type'].unique():
            fuzzer_df = scenario_df[scenario_df['fuzzer_type'] == fuzzer]
            report.append(f"  Fuzzer: {fuzzer}")
            for algo in fuzzer_df['algorithm'].unique():
                algo_df = fuzzer_df[fuzzer_df['algorithm'] == algo]
                vuln_counts = Counter()
                ping_pong_ues = set()
                for _, row in algo_df.iterrows():
                    counts, ues = parse_vulnerabilities(row['vulnerabilities'])
                    vuln_counts.update(counts)
                    ping_pong_ues.update(ues)
                    all_vuln_counts.update(counts)
                    all_ping_pong_ues.update(ues)
                report.append(f"    Algorithm: {algo}")
                if vuln_counts:
                    for vuln_type, count in vuln_counts.items():
                        report.append(f"      {vuln_type}: {count} occurrences")
                    if ping_pong_ues:
                        report.append(f"      Ping-Pong UEs: {sorted(ping_pong_ues)}")
                else:
                    report.append("      No vulnerabilities detected.")
    report.append("\nOverall Vulnerability Counts:")
    for vuln_type, count in all_vuln_counts.items():
        report.append(f"  {vuln_type}: {count} occurrences")
    report.append(f"Overall Ping-Pong UEs: {sorted(all_ping_pong_ues)}")
    report.append("")

    # 3. تحلیل SINR
    report.append("=== SINR Analysis ===")
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        report.append(f"\nScenario: {scenario}")
        all_sinr_values = []
        for _, row in scenario_df.iterrows():
            sinr_list = safe_eval(row['assigned_sinr_list_str'])
            all_sinr_values.extend([x for x in sinr_list if np.isfinite(x)])
        if all_sinr_values:
            sinr_np = np.array(all_sinr_values)
            report.append(f"  Mean SINR: {sinr_np.mean():.2f} dB")
            report.append(f"  5th Percentile: {np.percentile(sinr_np, 5):.2f} dB")
            report.append(f"  Median: {np.median(sinr_np):.2f} dB")
            report.append(f"  95th Percentile: {np.percentile(sinr_np, 95):.2f} dB")
            below_qos = np.sum(sinr_np < QOS_SINR_THRESHOLD) / len(sinr_np) * 100
            report.append(f"  % UEs below QoS ({QOS_SINR_THRESHOLD} dB): {below_qos:.2f}%")
        else:
            report.append("  No valid SINR data available.")
        for fuzzer in scenario_df['fuzzer_type'].unique():
            fuzzer_df = scenario_df[scenario_df['fuzzer_type'] == fuzzer]
            report.append(f"  Fuzzer: {fuzzer}")
            for algo in fuzzer_df['algorithm'].unique():
                algo_df = fuzzer_df[fuzzer_df['algorithm'] == algo]
                mean_overall = algo_df['avg_overall_sinr'].mean()
                mean_high_prio = algo_df['avg_high_prio_sinr'].mean()
                report.append(
                    f"    Algorithm: {algo}, Avg Overall SINR: {mean_overall:.2f} dB, "
                    f"Avg High-Prio SINR: {mean_high_prio:.2f} dB"
                )
    report.append("")

    # 4. تحلیل Handover
    report.append("=== Handover Analysis ===")
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        total_hos = scenario_df['handover_count_iter'].sum()
        mean_hos = scenario_df['handover_count_iter'].mean()
        high_ho_iters = len(scenario_df[scenario_df['handover_count_iter'] > 10])
        report.append(f"\nScenario: {scenario}")
        report.append(f"  Total Handovers: {total_hos}")
        report.append(f"  Mean Handovers per Iteration: {mean_hos:.2f}")
        report.append(f"  Iterations with High Handovers (>10): {high_ho_iters}")
        for fuzzer in scenario_df['fuzzer_type'].unique():
            fuzzer_df = scenario_df[scenario_df['fuzzer_type'] == fuzzer]
            for algo in fuzzer_df['algorithm'].unique():
                algo_df = fuzzer_df[fuzzer_df['algorithm'] == algo]
                mean_hos_algo = algo_df['handover_count_iter'].mean()
                report.append(f"    Fuzzer: {fuzzer}, Algo: {algo}, Mean Handovers: {mean_hos_algo:.2f}")
    report.append("")

    # 5. تحلیل بار سلول‌ها
    report.append("=== Cell Load Analysis ===")
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        report.append(f"\nScenario: {scenario}")
        report.append(f"  Mean Load Min: {scenario_df['load_min'].mean():.2f}")
        report.append(f"  Mean Load Max: {scenario_df['load_max'].mean():.2f}")
        report.append(f"  Mean Load Std: {scenario_df['load_std'].mean():.2f}")
        high_std_iters = len(scenario_df[scenario_df['load_std'] > 0.1])
        report.append(f"  Iterations with High Load Std (>0.1): {high_std_iters}")
        for fuzzer in scenario_df['fuzzer_type'].unique():
            fuzzer_df = scenario_df[scenario_df['fuzzer_type'] == fuzzer]
            for algo in fuzzer_df['algorithm'].unique():
                algo_df = fuzzer_df[fuzzer_df['algorithm'] == algo]
                mean_std = algo_df['load_std'].mean()
                report.append(f"    Fuzzer: {fuzzer}, Algo: {algo}, Mean Load Std: {mean_std:.2f}")
    report.append("")

    # 6. مقایسه فازرها و الگوریتم‌ها
    report.append("=== Fuzzer and Algorithm Comparison ===")
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        report.append(f"\nScenario: {scenario}")
        for metric in ['avg_overall_sinr', 'fairness_index', 'handover_count_iter']:
            report.append(f"  {metric.replace('_', ' ').title()}:")
            for fuzzer in scenario_df['fuzzer_type'].unique():
                fuzzer_df = scenario_df[scenario_df['fuzzer_type'] == fuzzer]
                for algo in fuzzer_df['algorithm'].unique():
                    algo_df = fuzzer_df[fuzzer_df['algorithm'] == algo]
                    mean_val = algo_df[metric].mean()
                    report.append(f"    Fuzzer: {fuzzer}, Algo: {algo}, Mean: {mean_val:.2f}")
    report.append("")

    # 7. تحلیل موقعیت UEها
    ue_position_report, edge_ues = analyze_ue_positions(df)
    report.extend(ue_position_report)

    # 8. پیشنهادات برای بهبود
    report.append("=== Recommendations ===")
    if df['handover_count_iter'].mean() > 7:
        report.append(
            "- High handover rates detected. Consider increasing the hysteresis margin "
            "(currently 8 dB) to 10 dB or adjusting the TTT (currently 0.3s) to 0.4s to reduce ping-pong effects."
        )
    if df['num_ues_below_qos'].mean() > 3:
        report.append(
            "- Many UEs have SINR below QoS threshold (2 dB). Consider increasing TX_POWER_DBM "
            "(currently 40 dBm) to 43 dBm or adding more BSs (e.g., NUM_CELLS=6) to reduce interference."
        )
    if df['fairness_index'].mean() < 0.3:
        report.append(
            "- Low fairness index detected. The utility-based algorithm performs better (mean fairness ~0.29). "
            "Alternatively, reduce load_threshold in TrafficSteering to 0.15 for better load balancing."
        )
    if len(all_ping_pong_ues) > 5:
        edge_ues_all = set(ue_id for scenario in edge_ues for ue_id, _ in edge_ues[scenario])
        ping_pong_edge_overlap = all_ping_pong_ues.intersection(edge_ues_all)
        report.append(
            f"- Frequent ping-pong detected for UEs {sorted(all_ping_pong_ues)}. "
            f"Edge UEs: {sorted(edge_ues_all)}. Overlapping UEs: {sorted(ping_pong_edge_overlap)}. "
            "These UEs are likely near cell edges. Consider increasing ping_pong_window (currently 12) to 15 or adjusting ping_pong_threshold to 3."
        )
    report.append("")

    # ذخیره گزارش
    try:
        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
    except Exception as e:
        return f"Error: Failed to write report to {OUTPUT_REPORT}: {e}"

    return df, f"Analysis complete. Report saved to {OUTPUT_REPORT}"

def main():
    """تابع اصلی برای اجرای تحلیل"""
    result = analyze_csv()
    if isinstance(result, str):
        print(result)
    else:
        df, message = result
        print(message)

if __name__ == "__main__":
    main()