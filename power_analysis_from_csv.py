import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.power import tt_ind_solve_power


# Load CSV
df = pd.read_csv('fuzzing_results_v28_strategic_fuzzing.csv')


# Print column names and data summary for debugging
print('CSV columns:', df.columns.tolist())
print('Data summary:', df['vulnerability_count'].describe())
print('Group counts:', df.groupby(['scenario', 'algorithm', 'fuzzer_type'])['vulnerability_count'].count())

# If needed, rename 'method' to 'fuzzer_type' for compatibility
df = df.rename(columns={'method': 'fuzzer_type'})




# Create synthetic 'run' index assuming 15 iterations per run (adjust if needed)
df['run'] = df.groupby(['scenario', 'algorithm', 'fuzzer_type']).cumcount() // 15

# Group by scenario, algorithm, fuzzer_type, run and sum vulnerability_count per run
per_run_df = df.groupby(['scenario', 'algorithm', 'fuzzer_type', 'run'])['vulnerability_count'].sum().reset_index()
per_run_df = per_run_df.rename(columns={'vulnerability_count': 'total_vulnerabilities_per_run'})

# Filter per method
ai_per_run = per_run_df[per_run_df['fuzzer_type'] == 'AI-Fuzzing']['total_vulnerabilities_per_run']
trad_per_run = per_run_df[per_run_df['fuzzer_type'] == 'Traditional-Testing']['total_vulnerabilities_per_run']

# Calculate means and std
mean_ai, std_ai, n_ai = ai_per_run.mean(), ai_per_run.std(), ai_per_run.count()
mean_trad, std_trad, n_trad = trad_per_run.mean(), trad_per_run.std(), trad_per_run.count()

# Pooled SD
pooled_sd = np.sqrt(((n_ai-1)*std_ai**2 + (n_trad-1)*std_trad**2) / (n_ai + n_trad - 2))

# Cohen's d
d = (mean_ai - mean_trad) / pooled_sd if pooled_sd > 0 else 0

# t-test (two-sided by default)
stat, pval = ttest_ind(ai_per_run, trad_per_run, equal_var=True)


# Power analysis (two-sided, to match paper's t-test) with fixed ratio
power = tt_ind_solve_power(effect_size=d, nobs1=n_ai, alpha=0.05, ratio=1.0, alternative='two-sided')

print(f"AI-Fuzzing per-run: mean={mean_ai:.2f}, SD={std_ai:.2f}, n={n_ai}")
print(f"Traditional per-run: mean={mean_trad:.2f}, SD={std_trad:.2f}, n={n_trad}")
print(f"Cohen's d: {d:.3f}")
print(f"t-test p-value: {pval:.5f}")
print(f"Estimated power (two-sided, alpha=0.05): {power:.2f}")
