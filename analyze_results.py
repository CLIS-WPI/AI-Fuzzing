# -*- coding: utf-8 -*-
# Analysis Script for O-RAN Fuzzing Results
# Version 5.3: Ensuring plot_cdfs is defined and called correctly.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
from collections import Counter
import os
# from scipy import stats # Uncomment if you add statistical tests later

# --- Configuration ---
CSV_FILEPATH = 'fuzzing_results_v22_enhanced_metrics_tuned.csv' 
ANALYSIS_PLOT_DIR = f"analysis_output_{os.path.splitext(os.path.basename(CSV_FILEPATH))[0]}_v5_3_run"
SUMMARY_CSV_FILENAME = f"diagnostic_summary_{os.path.splitext(os.path.basename(CSV_FILEPATH))[0]}.csv"

SIMULATION_ITERATIONS = 100 
BS_LOCATIONS_2D = np.array([[0,0], [100,0], [50, 86.6]]) 
NUM_UES = 30 
NOISE_POWER_WATTS_ANALYSIS = 10**((-174 - 30) / 10) * 20e6

FLAGS = {
    "data_loading_error": False, "simulation_incomplete": False,
    "extreme_low_5th_perc_sinr_found": False, "extreme_high_95th_perc_sinr_found": False,
    "very_low_individual_ue_sinr_found": False, "very_high_individual_ue_sinr_found": False,
    "predominant_qos_violations": False, "few_ping_pongs_in_high_mobility": False,
    "ai_fuzzer_more_severe_qos": False, 
    "random_fuzzer_more_extreme_low_sinr": False, 
    "significant_ts_algo_difference_vuln": False,
    "significant_fuzzer_difference_vuln": False,
    "high_handover_rate_overall": False,
    "very_low_fairness_overall": False,
    "suspiciously_high_95th_percentile_sinr_values_present": False,
}
LOW_SINR_PERCENTILE_THRESHOLD = -20.0 
HIGH_SINR_PERCENTILE_THRESHOLD = 60.0 
INDIVIDUAL_EXTREME_LOW_SINR_THRESHOLD = -30.0
INDIVIDUAL_EXTREME_HIGH_SINR_THRESHOLD = 70.0 
HIGH_HO_RATE_THRESHOLD = 15.0 
LOW_FAIRNESS_THRESHOLD_OVERALL = 0.25
SIGNIFICANCE_LEVEL_P_VALUE = 0.05


def parse_list_string_column(series, column_name_for_debug="Unknown"):
    parsed_list = []
    for item_idx, item in enumerate(series):
        if isinstance(item, str):
            try:
                stripped_item = item.strip()
                if (stripped_item.startswith('[') and stripped_item.endswith(']')) or \
                   (stripped_item.startswith('{') and stripped_item.endswith('}')):
                    parsed_list.append(ast.literal_eval(stripped_item))
                else:
                    parsed_list.append(np.nan) 
            except (ValueError, SyntaxError):
                parsed_list.append(np.nan) 
        elif pd.isna(item) and column_name_for_debug == 'vulnerabilities':
            parsed_list.append([])
        else:
            parsed_list.append(item)
    return parsed_list

def load_and_preprocess_data(csv_filepath):
    global FLAGS, SIMULATION_ITERATIONS
    print(f"Loading data from: {csv_filepath}")
    if not os.path.exists(csv_filepath):
        print(f"ERROR: CSV file not found at {csv_filepath}"); FLAGS["data_loading_error"] = True; return None
    try:
        df = pd.read_csv(csv_filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        if 'iteration' in df.columns and not df['iteration'].empty:
            max_iter = df['iteration'].max()
            if pd.notna(max_iter):
                SIMULATION_ITERATIONS = int(max_iter) + 1
                print(f"Inferred SIMULATION_ITERATIONS from CSV: {SIMULATION_ITERATIONS}")
                if SIMULATION_ITERATIONS < 50: FLAGS["simulation_incomplete"] = True
        else:
            FLAGS["simulation_incomplete"] = True; print("Warning: Could not determine max iteration.")
    except Exception as e:
        print(f"Error loading CSV: {e}"); FLAGS["data_loading_error"] = True; return None
    str_list_cols = [
        'cell_loads_list_str', 'ue_locations_str', 'serving_cell_ids_str',
        'assigned_rsrp_list_str', 'assigned_sinr_list_str',
        'fuzzed_load_modifier_str', 'fuzzed_pos_modifier_str', 'vulnerabilities'
    ]
    for col in str_list_cols:
        if col in df.columns:
            df[col] = parse_list_string_column(df[col], column_name_for_debug=col)
        else: print(f"Warning: Expected column '{col}' not found.")
    if 'vulnerabilities' in df.columns:
        df['vulnerabilities'] = df['vulnerabilities'].apply(
            lambda x: [str(v) for v in x] if isinstance(x, list) else ([] if pd.isna(x) else [str(x)])
        )
        df['vulnerability_count'] = df['vulnerabilities'].apply(len)
    else: df['vulnerability_count'] = 0
    numeric_metrics_to_check = [
        'avg_overall_sinr', 'avg_high_prio_sinr', 'fairness_index',
        'handover_count_iter', 'num_ues_below_qos',
        'sinr_5th_percentile', 'sinr_50th_percentile', 'sinr_95th_percentile',
        'load_min', 'load_max', 'load_std'
    ]
    for col in numeric_metrics_to_check:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def overall_summary_stats(df):
    global FLAGS
    if df is None: return
    print("\n--- Dataframe Info ---"); df.info(verbose=True, show_counts=True)
    print("\n--- Descriptive Statistics (Numeric Columns) ---")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        desc_stats = df[numeric_cols].describe().transpose()
        print(desc_stats.to_string())
        if 'fairness_index' in desc_stats.index and pd.notna(desc_stats.loc['fairness_index', 'mean']) and desc_stats.loc['fairness_index', 'mean'] < LOW_FAIRNESS_THRESHOLD_OVERALL: FLAGS["very_low_fairness_overall"] = True
        if 'handover_count_iter' in desc_stats.index and pd.notna(desc_stats.loc['handover_count_iter', 'mean']) and desc_stats.loc['handover_count_iter', 'mean'] > HIGH_HO_RATE_THRESHOLD: FLAGS["high_handover_rate_overall"] = True
        if 'sinr_95th_percentile' in desc_stats.index and pd.notna(desc_stats.loc['sinr_95th_percentile', 'mean']) and desc_stats.loc['sinr_95th_percentile', 'mean'] > HIGH_SINR_PERCENTILE_THRESHOLD + 10 : FLAGS["suspiciously_high_95th_percentile_sinr_values_present"] = True
    else: print("No numeric columns found.")
    print("\n--- Missing Values Per Column (if any) ---")
    missing = df.isnull().sum(); print(missing[missing > 0].to_string() if missing.any() else "No missing values found.")

def analyze_vulnerabilities(df, output_dir):
    global FLAGS
    if df is None or 'vulnerabilities' not in df.columns: return
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Vulnerability Analysis ---")
    all_vulns_flat_list = [str(vuln) for sublist in df['vulnerabilities'].dropna() for vuln in sublist]
    vuln_counts = Counter(all_vulns_flat_list)
    print("Overall Vulnerability Counts:")
    if not vuln_counts: print("  No vulnerabilities detected.")
    else:
        for v_type, count in vuln_counts.most_common(): print(f"  {v_type}: {count}")
    total_vulns = sum(vuln_counts.values())
    if total_vulns > 0:
        qos_count = sum(count for v_type, count in vuln_counts.items() if "QoS Violation" in v_type)
        if (qos_count / total_vulns) > 0.5: FLAGS["predominant_qos_violations"] = True
    if 'scenario' in df.columns and 'High Mobility' in df['scenario'].unique():
        hm_df = df[df['scenario'] == 'High Mobility'].copy() 
        if not hm_df.empty and 'vulnerabilities' in hm_df.columns:
            hm_df['vulnerabilities'] = hm_df['vulnerabilities'].apply(lambda x: [str(v) for v in x] if isinstance(x, list) else [])
            hm_all_vulns_flat = [vuln for sublist in hm_df['vulnerabilities'].dropna() if isinstance(sublist, list) for vuln in sublist]
            hm_vuln_counts_scenario = Counter(hm_all_vulns_flat) 
            hm_ping_pong_count = sum(count for v_type, count in hm_vuln_counts_scenario.items() if "Ping-Pong" in v_type)
            hm_qos_count = sum(count for v_type, count in hm_vuln_counts_scenario.items() if "QoS Violation" in v_type)
            iters_with_any_vuln_hm = len(hm_df[hm_df['vulnerability_count'] > 0])
            if iters_with_any_vuln_hm > 0 and hm_ping_pong_count < (0.15 * iters_with_any_vuln_hm) and hm_qos_count > (0.3 * iters_with_any_vuln_hm):
                FLAGS["few_ping_pongs_in_high_mobility"] = True
    if vuln_counts:
        plt.figure(figsize=(14, 8)) 
        simplified_vuln_map = {}
        for v_str, count in vuln_counts.items():
            main_type = v_str.split(':')[0].strip(); simplified_vuln_map[main_type] = simplified_vuln_map.get(main_type, 0) + count
        if simplified_vuln_map:
            sns.barplot(x=list(simplified_vuln_map.keys()), y=list(simplified_vuln_map.values()))
            plt.title("Overall Distribution of Main Vulnerability Types"); plt.ylabel("Total Count"); plt.xlabel("Main Vulnerability Type")
            plt.xticks(rotation=30, ha="right"); plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "overall_main_vulnerability_types.png")); print(f"Saved plot: {os.path.join(output_dir, 'overall_main_vulnerability_types.png')}"), plt.close()
    if 'fuzzer_type' in df.columns and 'vulnerability_count' in df.columns:
        plt.figure(figsize=(8, 6))
        df.groupby('fuzzer_type')['vulnerability_count'].sum().plot(kind='bar')
        plt.title("Total Vulnerability Count by Fuzzer Type"); plt.ylabel("Total Vulnerabilities Detected"); plt.xlabel("Fuzzer Type")
        plt.xticks(rotation=0); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "vuln_count_by_fuzzer.png")); print(f"Saved plot: {os.path.join(output_dir, 'vuln_count_by_fuzzer.png')}"), plt.close()
    if all(col in df.columns for col in ['scenario', 'algorithm', 'fuzzer_type', 'vulnerability_count']):
        try:
            grouped_vulns = df.groupby(['scenario', 'algorithm', 'fuzzer_type'])['vulnerability_count'].sum().reset_index()
            if not grouped_vulns.empty:
                g = sns.catplot(data=grouped_vulns, x='algorithm', y='vulnerability_count', hue='fuzzer_type', col='scenario', kind='bar', height=5, aspect=0.9, legend_out=True)
                g.fig.suptitle("Total Vulnerabilities by Algorithm, Fuzzer, and Scenario", y=1.02); g.set_xticklabels(rotation=30, ha="right")
                plt.tight_layout(rect=[0,0,0.9,0.95]); plt.savefig(os.path.join(output_dir, "vuln_count_by_algo_fuzzer_scenario.png"))
                print(f"Saved plot: {os.path.join(output_dir, 'vuln_count_by_algo_fuzzer_scenario.png')}"), plt.close()
        except Exception as e: print(f"Error generating grouped vulnerability plot: {e}")

def analyze_key_metrics(df, output_dir):
    if df is None: return
    os.makedirs(output_dir, exist_ok=True)
    print("\n--- Key Metric Analysis ---")
    key_numerical_metrics = ['avg_overall_sinr', 'avg_high_prio_sinr', 'fairness_index', 'handover_count_iter', 'num_ues_below_qos', 'sinr_5th_percentile', 'sinr_50th_percentile', 'sinr_95th_percentile', 'load_min', 'load_max', 'load_std']
    for metric in key_numerical_metrics:
        if metric not in df.columns: print(f"Metric '{metric}' not found, skipping analysis."); continue
        print(f"\nAnalyzing metric: {metric}")
        try:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            if df[metric].isnull().all(): print(f"All values for metric '{metric}' are NaN, skipping."); continue
            grouped_stats = df.groupby(['scenario', 'fuzzer_type', 'algorithm'])[metric].agg(['mean', 'std', 'min', 'max'])
            if not grouped_stats.empty: print(grouped_stats.to_string())
            else: print(f"No data to aggregate for metric: {metric}")
            plt.figure(figsize=(13, 7)); sns.boxplot(data=df, x='scenario', y=metric, hue='fuzzer_type', dodge=True)
            plt.title(f"Distribution of {metric.replace('_', ' ').title()} by Scenario and Fuzzer"); plt.xticks(rotation=10, ha="right"); plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(os.path.join(output_dir, f"{metric}_dist_by_scenario_fuzzer.png")); print(f"Saved plot: {os.path.join(output_dir, f'{metric}_dist_by_scenario_fuzzer.png')}"), plt.close()
            plt.figure(figsize=(10, 7)); sns.violinplot(data=df, x='algorithm', y=metric, hue='fuzzer_type', dodge=True, inner="quartile")
            plt.title(f"Distribution of {metric.replace('_', ' ').title()} by Algorithm and Fuzzer"); plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
            plt.tight_layout(rect=[0, 0, 0.85, 1]); plt.savefig(os.path.join(output_dir, f"{metric}_dist_by_algo_fuzzer.png")); print(f"Saved plot: {os.path.join(output_dir, f'{metric}_dist_by_algo_fuzzer.png')}"), plt.close()
        except Exception as e: print(f"Error analyzing/plotting metric {metric}: {e}")

def diagnose_extreme_sinr_values(df, output_dir, 
                                 low_sinr_col='sinr_5th_percentile', low_sinr_threshold=-20.0, 
                                 high_sinr_col='sinr_95th_percentile', high_sinr_threshold=60.0,
                                 individual_low_thr=-30.0, individual_high_thr=70.0,
                                 num_samples_to_show=3):
    global FLAGS
    if df is None: return
    print(f"\n--- Diagnosing Extreme SINR Values (Low 5th %ile < {low_sinr_threshold}dB, High 95th %ile > {high_sinr_threshold}dB) ---")
    print(f"--- (Individual UE SINR thresholds: Low < {individual_low_thr}dB, High > {individual_high_thr}dB) ---")
    context_cols_base = ['scenario', 'iteration', 'fuzzer_type', 'algorithm']
    if low_sinr_col in df.columns and pd.to_numeric(df[low_sinr_col], errors='coerce').notna().any():
        df_copy = df.copy(); df_copy[low_sinr_col] = pd.to_numeric(df_copy[low_sinr_col], errors='coerce')
        low_sinr_percentile_cases = df_copy[df_copy[low_sinr_col] < low_sinr_threshold]
        print(f"\nFound {len(low_sinr_percentile_cases)} instances where '{low_sinr_col}' < {low_sinr_threshold} dB:")
        if not low_sinr_percentile_cases.empty:
            FLAGS["extreme_low_5th_perc_sinr_found"] = True
            print(f"  Details for top {min(num_samples_to_show, len(low_sinr_percentile_cases))} LOWEST 5th Percentile SINR cases:")
            for index, row in low_sinr_percentile_cases.nsmallest(num_samples_to_show, low_sinr_col).iterrows():
                print(f"\n    --- Case: Index {index} ---")
                for col_name in context_cols_base + [low_sinr_col]: print(f"      {col_name}: {row.get(col_name)}")
                ue_locations = row.get('ue_locations_str', []); assigned_sinrs = row.get('assigned_sinr_list_str', [])
                serving_cells = row.get('serving_cell_ids_str', []); assigned_rsrps = row.get('assigned_rsrp_list_str', [])
                if isinstance(assigned_sinrs, list) and assigned_sinrs:
                    valid_sinrs_np = np.array([s for s in assigned_sinrs if isinstance(s, (int, float)) and pd.notna(s)])
                    if valid_sinrs_np.size > 0:
                        min_sinr_val_in_list = np.min(valid_sinrs_np)
                        min_sinr_ue_indices = [i for i, s in enumerate(assigned_sinrs) if isinstance(s, (int, float)) and pd.notna(s) and s == min_sinr_val_in_list]
                        print(f"      Min Individual SINR in this step: {min_sinr_val_in_list:.2f} dB at UE(s) index: {min_sinr_ue_indices[:1]}")
                        if min_sinr_val_in_list < INDIVIDUAL_EXTREME_LOW_SINR_THRESHOLD: FLAGS["very_low_individual_ue_sinr_found"] = True
                        for ue_idx in min_sinr_ue_indices[:1]:
                             if isinstance(ue_locations, list) and ue_idx < len(ue_locations) and isinstance(ue_locations[ue_idx], list) and len(ue_locations[ue_idx])==2:
                                ue_pos = ue_locations[ue_idx]; print(f"        UE {ue_idx} Location: {[round(c,1) for c in ue_pos]}")
                                distances = [np.linalg.norm(np.array(ue_pos) - bs_pos) for bs_pos in BS_LOCATIONS_2D]
                                print(f"        UE {ue_idx} Distances to BSs: {[f'{d:.1f}m' for d in distances]}")
                             if isinstance(serving_cells, list) and ue_idx < len(serving_cells): print(f"        UE {ue_idx} Serving Cell ID: {serving_cells[ue_idx]}")
                             if isinstance(assigned_rsrps, list) and ue_idx < len(assigned_rsrps) and isinstance(assigned_rsrps[ue_idx], (int, float)) and pd.notna(assigned_rsrps[ue_idx]):
                                print(f"        UE {ue_idx} Assigned RSRP: {assigned_rsrps[ue_idx]:.2f} dBm")
                                signal_linear = 10**((assigned_rsrps[ue_idx]-30)/10)
                                sinr_linear = 10**(assigned_sinrs[ue_idx]/10)
                                if sinr_linear != 0 and abs(sinr_linear) > 1e-10 : 
                                    i_plus_n_linear = signal_linear / sinr_linear
                                    i_plus_n_dbm = 10 * np.log10(i_plus_n_linear * 1000) if i_plus_n_linear > 1e-30 else -np.inf
                                    print(f"        UE {ue_idx} Derived (I+N): {i_plus_n_linear:.2e} W ({i_plus_n_dbm:.2f} dBm vs NoiseFloor: {10*np.log10(NOISE_POWER_WATTS_ANALYSIS*1000):.2f} dBm)")
                print(f"      Fuzzed Load: {row.get('fuzzed_load_modifier_str')}")
                print(f"      Vulnerabilities: {row.get('vulnerabilities')}")
    if high_sinr_col in df.columns and df[high_sinr_col].notna().any():
        df_copy = df.copy(); df_copy[high_sinr_col] = pd.to_numeric(df_copy[high_sinr_col], errors='coerce')
        high_sinr_percentile_cases = df_copy[df_copy[high_sinr_col] > high_sinr_threshold]
        print(f"\nFound {len(high_sinr_percentile_cases)} instances where '{high_sinr_col}' > {high_sinr_threshold} dB:")
        if not high_sinr_percentile_cases.empty:
            FLAGS["extreme_high_95th_perc_sinr_found"] = True
            print(f"  Details for top {min(num_samples_to_show, len(high_sinr_percentile_cases))} HIGHEST 95th Percentile SINR cases:")
            for index, row in high_sinr_percentile_cases.nlargest(num_samples_to_show, high_sinr_col).iterrows():
                print(f"\n    --- Case: Index {index} ---")
                for col_name in context_cols_base + [high_sinr_col]: print(f"      {col_name}: {row.get(col_name)}")
                ue_locations = row.get('ue_locations_str', []); assigned_sinrs = row.get('assigned_sinr_list_str', [])
                serving_cells = row.get('serving_cell_ids_str', []); assigned_rsrps = row.get('assigned_rsrp_list_str', [])
                if isinstance(assigned_sinrs, list) and assigned_sinrs:
                    valid_sinrs_np = np.array([s for s in assigned_sinrs if isinstance(s, (int, float)) and pd.notna(s)])
                    if valid_sinrs_np.size > 0:
                        max_sinr_val_in_list = np.max(valid_sinrs_np)
                        max_sinr_ue_indices = [i for i, s in enumerate(assigned_sinrs) if isinstance(s, (int, float)) and pd.notna(s) and s == max_sinr_val_in_list]
                        print(f"      Max Individual SINR in this step: {max_sinr_val_in_list:.2f} dB at UE(s) index: {max_sinr_ue_indices[:1]}")
                        if max_sinr_val_in_list > INDIVIDUAL_EXTREME_HIGH_SINR_THRESHOLD: FLAGS["very_high_individual_ue_sinr_found"] = True
                        for ue_idx in max_sinr_ue_indices[:1]:
                            if isinstance(ue_locations, list) and ue_idx < len(ue_locations) and isinstance(ue_locations[ue_idx], list) and len(ue_locations[ue_idx])==2:
                                ue_pos = ue_locations[ue_idx]; print(f"        UE {ue_idx} Location: {[round(c,1) for c in ue_pos]}")
                                distances = [np.linalg.norm(np.array(ue_pos) - bs_pos) for bs_pos in BS_LOCATIONS_2D]
                                print(f"        UE {ue_idx} Distances to BSs: {[f'{d:.1f}m' for d in distances]}")
                            if isinstance(serving_cells, list) and ue_idx < len(serving_cells): print(f"        UE {ue_idx} Serving Cell ID: {serving_cells[ue_idx]}")
                            if isinstance(assigned_rsrps, list) and ue_idx < len(assigned_rsrps) and isinstance(assigned_rsrps[ue_idx],(int,float)) and pd.notna(assigned_rsrps[ue_idx]):
                                print(f"        UE {ue_idx} Assigned RSRP: {assigned_rsrps[ue_idx]:.2f} dBm")
                                signal_linear = 10**((assigned_rsrps[ue_idx]-30)/10)
                                sinr_linear = 10**(assigned_sinrs[ue_idx]/10)
                                if sinr_linear != 0 and abs(sinr_linear) > 1e-10: 
                                    i_plus_n_linear = signal_linear / sinr_linear
                                    i_plus_n_dbm = 10 * np.log10(i_plus_n_linear * 1000) if i_plus_n_linear > 1e-30 else -np.inf
                                    print(f"        UE {ue_idx} Derived (I+N): {i_plus_n_linear:.2e} W ({i_plus_n_dbm:.2f} dBm vs NoiseFloor: {10*np.log10(NOISE_POWER_WATTS_ANALYSIS*1000):.2f} dBm)")
                print(f"      Fuzzed Load: {row.get('fuzzed_load_modifier_str')}")
                print(f"      Vulnerabilities: {row.get('vulnerabilities')}")

def plot_cdfs(df, output_dir): # Corrected function name
    if df is None:
        print("CDF plotting skipped: DataFrame is None.")
        return
    print("\n--- Plotting CDFs for Key SINR Metrics ---")
    os.makedirs(output_dir, exist_ok=True)
    
    sinr_metrics_for_cdf = ['avg_overall_sinr', 'avg_high_prio_sinr', 
                            'sinr_5th_percentile', 'sinr_50th_percentile', 'sinr_95th_percentile']
    
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario].copy()
        for metric in sinr_metrics_for_cdf:
            if metric not in scenario_df.columns: 
                print(f"  Metric '{metric}' not found in scenario '{scenario}', skipping CDF plot.")
                continue
            
            scenario_df[metric] = pd.to_numeric(scenario_df[metric], errors='coerce')
            
            plt.figure(figsize=(12,8))
            legend_items_exist = False
            for algo in scenario_df['algorithm'].unique():
                for fuzzer in scenario_df['fuzzer_type'].unique():
                    subset_data = scenario_df[(scenario_df['algorithm']==algo) & (scenario_df['fuzzer_type']==fuzzer)][metric].dropna()
                    if not subset_data.empty:
                        sorted_data = np.sort(subset_data)
                        yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
                        plt.plot(sorted_data, yvals, marker='.', linestyle='-', markersize=4, alpha=0.7, label=f'{algo} ({fuzzer})')
                        legend_items_exist = True
            
            if not legend_items_exist:
                plt.close() 
                print(f"  No valid data to plot CDF for '{metric}' in scenario '{scenario}'.")
                continue

            plt.title(f'CDF of {metric.replace("_", " ").title()} - Scenario: {scenario}')
            plt.xlabel(metric.replace("_", " ").title() + " (dB where applicable)")
            plt.ylabel('Cumulative Probability P(X <= x)')
            plt.legend(loc='best') 
            plt.grid(True, which="both", ls="--", linewidth=0.5)
            
            if "sinr" in metric.lower() and not metric_data_numeric.empty: # Use metric_data_numeric here
                 q_low = metric_data_numeric.quantile(0.005) 
                 q_high = metric_data_numeric.quantile(0.995)
                 q_low = -60 if pd.isna(q_low) else q_low
                 q_high = 70 if pd.isna(q_high) else q_high
                 buffer = (q_high - q_low) * 0.1 if (q_high - q_low) > 0 and pd.notna(q_high - q_low) else 10
                 display_min = q_low - buffer; display_max = q_high + buffer
                 final_min = max(display_min, -70 if "5th" in metric else -50)
                 final_max = min(display_max, 70 if "95th" not in metric else 140)
                 if final_min >= final_max:
                     final_min = metric_data_numeric.min() - 5 if not metric_data_numeric.empty else -50
                     final_max = metric_data_numeric.max() + 5 if not metric_data_numeric.empty else (70 if "95th" not in metric else 140)
                     if final_min >= final_max: # Ultimate fallback
                         final_min = -50; final_max = (70 if "95th" not in metric else 140)
                         if "5th" in metric: final_min = -70
                 plt.xlim(final_min, final_max)
            
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f"CDF_{metric}_{scenario.replace(' ','_')}.png")
            try:
                plt.savefig(plot_filename); print(f"Saved plot: {plot_filename}")
            except Exception as e: print(f"Error saving CDF plot {plot_filename}: {e}")
            plt.close()

def generate_summary_csv_with_flags(df, output_filename):
    # (Identical to version 5)
    global FLAGS 
    if df is None: print("No data to generate summary CSV."); return
    print(f"\n--- Generating Summary CSV: {output_filename} ---")
    summary_list = []
    numeric_metrics_for_summary = ['avg_overall_sinr', 'avg_high_prio_sinr', 'fairness_index', 'handover_count_iter', 'num_ues_below_qos', 'sinr_5th_percentile', 'sinr_50th_percentile', 'sinr_95th_percentile', 'load_min', 'load_max', 'load_std']
    if not all(col in df.columns for col in ['scenario', 'fuzzer_type', 'algorithm']): return
    grouped = df.groupby(['scenario', 'fuzzer_type', 'algorithm'])
    for name, group in grouped:
        scenario, fuzzer_type, algorithm = name
        row = {'scenario': scenario, 'fuzzer_type': fuzzer_type, 'algorithm': algorithm}
        if 'vulnerabilities' in group.columns:
            group_vulns_flat = [str(v) for sublist in group['vulnerabilities'].dropna() for v in sublist]
            group_vuln_counts = Counter(group_vulns_flat)
            row['total_vuln_events'] = sum(group_vuln_counts.values()); row['unique_vuln_types_count'] = len(group_vuln_counts)
            row['avg_vuln_types_per_trigger_iter'] = group[group['vulnerability_count'] > 0]['vulnerability_count'].mean() if (group['vulnerability_count'] > 0).any() else 0
            row['count_qos_violations'] = sum(c for v, c in group_vuln_counts.items() if "QoS Violation" in v)
            row['count_ping_pong'] = sum(c for v, c in group_vuln_counts.items() if "Ping-Pong" in v)
            row['count_unfairness'] = sum(c for v, c in group_vuln_counts.items() if "Unfairness" in v)
        else:
            for key in ['total_vuln_events', 'unique_vuln_types_count', 'avg_vuln_types_per_trigger_iter', 'count_qos_violations', 'count_ping_pong', 'count_unfairness']: row[key] = 0
        for metric in numeric_metrics_for_summary:
            if metric in group.columns and group[metric].notna().any():
                row[f'mean_{metric}'] = group[metric].mean(); row[f'std_{metric}'] = group[metric].std()
                row[f'min_{metric}'] = group[metric].min(); row[f'max_{metric}'] = group[metric].max()
            else:
                for agg_type in ['mean', 'std', 'min', 'max']: row[f'{agg_type}_{metric}'] = np.nan
        if 'sinr_5th_percentile' in group.columns and pd.to_numeric(group['sinr_5th_percentile'], errors='coerce').notna().any():
            row['group_extreme_low_5th_sinr'] = (pd.to_numeric(group['sinr_5th_percentile'], errors='coerce') < LOW_SINR_PERCENTILE_THRESHOLD).any()
        if 'sinr_95th_percentile' in group.columns and pd.to_numeric(group['sinr_95th_percentile'], errors='coerce').notna().any():
            row['group_extreme_high_95th_sinr'] = (pd.to_numeric(group['sinr_95th_percentile'], errors='coerce') > HIGH_SINR_PERCENTILE_THRESHOLD).any()
        row['group_very_low_indiv_sinr'] = False; row['group_very_high_indiv_sinr'] = False
        if 'assigned_sinr_list_str' in group.columns:
            all_sinrs_group = [s for sinr_list in group['assigned_sinr_list_str'].dropna() if isinstance(sinr_list, list) for s in sinr_list if isinstance(s, (int, float)) and pd.notna(s)]
            if all_sinrs_group:
                if np.min(all_sinrs_group) < INDIVIDUAL_EXTREME_LOW_SINR_THRESHOLD: row['group_very_low_indiv_sinr'] = True
                if np.max(all_sinrs_group) > INDIVIDUAL_EXTREME_HIGH_SINR_THRESHOLD: row['group_very_high_indiv_sinr'] = True
        if 'handover_count_iter' in group.columns and pd.notna(group['handover_count_iter'].mean()) and group['handover_count_iter'].mean() > HIGH_HO_RATE_THRESHOLD:
            row['group_had_high_ho_rate'] = True
        summary_list.append(row)
    summary_df = pd.DataFrame(summary_list)
    try:
        summary_df.to_csv(output_filename, index=False, encoding='utf-8'); print(f"Summary CSV saved to: {output_filename}")
    except Exception as e: print(f"Error saving summary CSV: {e}")
    return summary_df

def print_diagnostic_flags_summary():
    # (Identical to version 5)
    print("\n--- Overall Diagnostic Flags Summary (from Analysis Script v5.2) ---")
    if FLAGS["data_loading_error"]: print("  CRITICAL: Data loading failed. Cannot proceed with analysis."); return
    if FLAGS["simulation_incomplete"]: print("  WARNING: Simulation iterations might be too low for conclusive results (less than 50 iterations detected).")
    print("\n  [Extreme Value Flags]")
    print(f"    Extreme low 5th percentile SINR (<{LOW_SINR_PERCENTILE_THRESHOLD}dB) found: {FLAGS['extreme_low_5th_perc_sinr_found']}")
    print(f"    Extreme high 95th percentile SINR (>{HIGH_SINR_PERCENTILE_THRESHOLD}dB) found: {FLAGS['extreme_high_95th_perc_sinr_found']}")
    if FLAGS["extreme_high_95th_perc_sinr_found"]: print("      SUGGESTION: High 95th percentile SINR is good, but review if values are unrealistically high (e.g., > 60dB consistently).")
    print(f"    Very low individual UE SINR (<{INDIVIDUAL_EXTREME_LOW_SINR_THRESHOLD}dB) found: {FLAGS['very_low_individual_ue_sinr_found']}")
    if FLAGS["very_low_individual_ue_sinr_found"]: print("      SUGGESTION: Investigate specific UE locations and fuzzer inputs for these cases.")
    print(f"    Very high individual UE SINR (>{INDIVIDUAL_EXTREME_HIGH_SINR_THRESHOLD}dB) found: {FLAGS['very_high_individual_ue_sinr_found']}")
    if FLAGS["very_high_individual_ue_sinr_found"]: print("      WARNING: Individual UEs experiencing >"+str(INDIVIDUAL_EXTREME_HIGH_SINR_THRESHOLD)+"dB SINR is highly unusual and might indicate issues in power/interference calculation or extreme ideal scenarios being fuzzed.")
    print("\n  [Vulnerability Pattern Flags]")
    print(f"    QoS violations are predominant vulnerability type: {FLAGS['predominant_qos_violations']}")
    print(f"    Few Ping-Pong events in High Mobility (relative to QoS/iterations): {FLAGS['few_ping_pongs_in_high_mobility']}")
    if FLAGS["few_ping_pongs_in_high_mobility"]: print("      SUGGESTION: Check if very low SINRs in High Mobility are preventing HOs, or if Ping-Pong oracle window is too large.")
    print("\n  [Overall Performance Flags]")
    print(f"    Overall average handover rate is high (> {HIGH_HO_RATE_THRESHOLD}/iter): {FLAGS['high_handover_rate_overall']}")
    print(f"    Overall average fairness index is very low (< {LOW_FAIRNESS_THRESHOLD_OVERALL}): {FLAGS['very_low_fairness_overall']}")
    print(f"    Suspiciously high 95th percentile SINR values present globally: {FLAGS['suspiciously_high_95th_percentile_sinr_values_present']}")
    print("\n  --- End of Flags Summary ---")

def main_analyzer(csv_filepath, output_plot_dir, summary_csv_name):
    global SIMULATION_ITERATIONS, FLAGS
    try:
        temp_df_for_iter = pd.read_csv(csv_filepath, usecols=['iteration'], nrows=2000)
        if 'iteration' in temp_df_for_iter.columns and not temp_df_for_iter['iteration'].empty:
            max_iter = temp_df_for_iter['iteration'].max()
            if pd.notna(max_iter): SIMULATION_ITERATIONS = int(max_iter) + 1
    except Exception as e: print(f"Could not infer SIMULATION_ITERATIONS, using default. Error: {e}")

    df = load_and_preprocess_data(csv_filepath)
    if df is None: FLAGS["data_loading_error"] = True; print_diagnostic_flags_summary(); return
    os.makedirs(output_plot_dir, exist_ok=True)

    overall_summary_stats(df)
    analyze_vulnerabilities(df, output_plot_dir)
    analyze_key_metrics(df, output_plot_dir)
    diagnose_extreme_sinr_values(df, output_plot_dir, 
                                 low_sinr_threshold=LOW_SINR_PERCENTILE_THRESHOLD, 
                                 high_sinr_threshold=HIGH_SINR_PERCENTILE_THRESHOLD,
                                 individual_low_thr=INDIVIDUAL_EXTREME_LOW_SINR_THRESHOLD,
                                 individual_high_thr=INDIVIDUAL_EXTREME_HIGH_SINR_THRESHOLD,
                                 num_samples_to_show=3) 
    plot_cdfs(df, output_plot_dir) # تابع اینجا فراخوانی می‌شود
    
    summary_csv_path = os.path.join(output_plot_dir, summary_csv_name)
    generate_summary_csv_with_flags(df, summary_csv_path)
    
    print_diagnostic_flags_summary() 
    print(f"\nAnalysis complete. Plots and summary CSV saved to: {output_plot_dir}")

if __name__ == "__main__":
    target_csv = CSV_FILEPATH 
    if not os.path.exists(target_csv):
        csv_from_last_good_run = "fuzzing_results_v20_OFDM_retracing_fix_attempt.csv"
        if os.path.exists(csv_from_last_good_run) and CSV_FILEPATH != csv_from_last_good_run :
             print(f"File '{target_csv}' (current default) not found, trying '{csv_from_last_good_run}' from a previous known run.")
             target_csv = csv_from_last_good_run
        else: 
            if not os.path.exists(CSV_FILEPATH):
                 print(f"ERROR: Neither '{target_csv}' nor the default CSV_FILEPATH '{CSV_FILEPATH}' found. Please specify the correct CSV_FILEPATH at the top.")
                 target_csv = None
    
    if target_csv:
        analysis_run_plot_dir = f"analysis_output_{os.path.splitext(os.path.basename(target_csv))[0]}_v5_2_run" # Updated version
        summary_csv_name_arg = f"diagnostic_summary_{os.path.splitext(os.path.basename(target_csv))[0]}.csv"
        main_analyzer(target_csv, analysis_run_plot_dir, summary_csv_name_arg)
    else:
        print("No CSV file to analyze. Exiting.")