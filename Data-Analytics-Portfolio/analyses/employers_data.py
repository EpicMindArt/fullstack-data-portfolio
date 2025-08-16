# -*- coding: utf-8 -*-
"""
HR Compensation & Budget Analysis
This script implements a practical analytics pipeline for HR pay-equity, compression,
bands, and budgeting questions using the provided dataset and framework style.

Key capabilities implemented (aligned with "Quick Wins" and "Deep Dives"):
- Pay Equity global and stratified audit (OLS on log salary, robust SE, FDR correction)
- Top segments by adjusted gender gap
- Cost-to-parity scenarios (corridor ±2%) and bring-to-median
- Compensation bands p25–p50–p75 by role×location×seniority with out-of-band shares
- Compression indices and progression slopes
- Mispricing pockets via residuals (model without Gender for diagnosis)
- Manager premium vs IC by department×location
- Location premium indices and basic scenario modeling helper
- Education ROI (Master/PhD vs Bachelor)
- Female representation and Wilson confidence intervals
- Outlier rates and cost to bring to nearest band boundary

Notes:
- No destructive cleaning beyond safe trimming/standardization; we do not drop rows on content grounds.
- Age is used only as a control in equity models, not as a lever for decisions.
- Minimum segment N and FDR correction are applied to avoid noisy claims.
"""

# --------------------------------------------------------------------------------------
# Pretty console setup (Rich + Loguru)
# --------------------------------------------------------------------------------------

import os
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters

from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.traceback import install as rich_traceback_install

from loguru import logger

# Statsmodels for econometrics
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from statsmodels.regression.quantile_regression import QuantReg

from utils.data_io import export_dataframe

from visualizers.employers_visualizer import generate_visualizations

# Custom Rich theme for consistent styling throughout the console output
_custom_theme = Theme(
    {
        "phase": "bold bright_cyan",
        "question": "bold cyan",
        "good": "bold green",
        "warn": "bold yellow",
        "error": "bold red",
        "info": "cyan",
        "muted": "dim",
    }
)

console = Console(theme=_custom_theme, highlight=False)
rich_traceback_install(show_locals=False, width=120, extra_lines=2, word_wrap=True)

# Configure Loguru to write via Rich console with a neat format
logger.remove()
logger.add(
    console.print,
    level="INFO",
    colorize=True,
    format="<dim>{time:YYYY-MM-DD HH:mm:ss}</dim> | <level>{level: <8}</level> | "
           "<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


# --------------------------------------------------------------------------------------
# Utility helpers for nice console output
# --------------------------------------------------------------------------------------

def _human_readable_bytes(num_bytes: int) -> str:
    """Return human-readable memory size string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def _show_dataset_snapshot(df: pd.DataFrame, title: str = "Dataset overview"):
    """Show a compact metadata snapshot of DataFrame."""
    mem = df.memory_usage(deep=True).sum()
    table = Table(title=title, show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Metric", style="muted")
    table.add_column("Value", style="bold")
    table.add_row("Rows", f"{df.shape[0]:,}")
    table.add_row("Columns", f"{df.shape[1]:,}")
    table.add_row("Memory", _human_readable_bytes(mem))
    console.print(table)


def _show_output_tree(output_root: str, csv_dir: str, excel_dir: str, charts_dir: str, title: str = "Saved outputs"):
    """Show a tree preview of the output directories and up to 10 files in each."""
    tree = Tree(f"[bold]Output[/] -> {output_root}", guide_style="bright_blue")
    for sub in [csv_dir, excel_dir, charts_dir]:
        node = tree.add(f"[bold]{os.path.basename(sub)}/[/] ({len(os.listdir(sub)) if os.path.exists(sub) else 0} files)")
        if os.path.exists(sub):
            files = sorted(os.listdir(sub))
            preview = files[:10]  # show up to 10 files for the preview
            for f in preview:
                node.add(f"{f}")
            if len(files) > len(preview):
                node.add(f"... and {len(files) - len(preview)} more")
        else:
            node.add("[muted]Directory not found[/]")
    console.print(Panel.fit(tree, title=title, border_style="bright_blue"))


# --------------------------------------------------------------------------------------
# Main analysis function (business logic)
# --------------------------------------------------------------------------------------

def run_analysis(df_raw_data: pd.DataFrame, config: dict):
    """
    Main execution function for the HR compensation analysis.

    Required columns (case-insensitive, will be normalized to snake_case):
        Employee_ID, Name, Age, Gender, Department, Job_Title,
        Experience_Years, Education_Level, Location, Salary

    Config keys:
        output_dir: str -> subfolder for outputs
        min_segment_n: int -> minimum segment headcount (default 30)
        fdr_alpha: float -> FDR alpha for multiple testing (default 0.05)
        equity_corridor: float -> acceptable parity corridor (default 0.02 for ±2%)
        increase_rate: float -> X% increase scenario for BUD-01 (default 0.05 for +5%)
        baseline_location: str -> GEO baseline location (default 'Austin')
        geo_scenario: dict -> optional scenario params, e.g. {"src": "San Francisco", "dst": "Austin", "fte_per_cluster": 1}
    """

    # ==============================================================================
    # CONFIGURATION & CONSTANTS
    # ==============================================================================
    PROJECT_NAME = config.get('output_dir', 'hr_compensation_analysis')

    # Governance defaults
    MIN_SEGMENT_N = int(config.get('min_segment_n', 30))
    FDR_ALPHA = float(config.get('fdr_alpha', 0.05))
    EQUITY_CORRIDOR = float(config.get('equity_corridor', 0.02))  # ±2%
    INCREASE_RATE = float(config.get('increase_rate', 0.05))      # +5% indexing scenario
    BASELINE_LOCATION = str(config.get('baseline_location', 'Austin'))
    GEO_SCENARIO = config.get('geo_scenario', {"src": "San Francisco", "dst": "Austin", "fte_per_cluster": 1})

    # --- Plotting Style Configuration (minimal plotting used here; still set defaults) ---
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    register_matplotlib_converters()
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)  # statsmodels/patsy occasionally warns on rank-deficiency

    # --- Output Directories ---
    OUTPUT_BASE_DIR = 'output'
    OUTPUT_CSV_DIR = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME, 'csv_reports')
    OUTPUT_EXCEL_DIR = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME, 'excel_reports')
    OUTPUT_CHARTS_DIR = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME, 'charts')
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_EXCEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_CHARTS_DIR, exist_ok=True)

    # --- Column Names for Exported Files: pretty names map (kept light and relevant) ---
    EXPORT_COLUMN_NAMES = {
        'employee_id': 'Employee ID',
        'name': 'Name',
        'age': 'Age',
        'gender': 'Gender',
        'department': 'Department',
        'job_title': 'Job Title',
        'experience_years': 'Experience (Years)',
        'education_level': 'Education Level',
        'location': 'Location',
        'salary': 'Salary',
        'log_salary': 'Log Salary',
        'experience_bucket': 'Experience Bucket',
        'age_cohort': 'Age Cohort',
        'cluster_cse02': 'Cluster (Role×Location×Seniority)',
        'cluster_heavy': 'Cluster (Role×Dept×Loc×Seniority×Edu)',

        # Analysis-specific columns
        'segment': 'Segment',
        'segment_value': 'Segment Value',
        'n': 'Headcount',
        'n_female': 'Female Count',
        'n_male': 'Male Count',
        'beta_female_log': 'Beta Female (log)',
        'se': 'Std. Error',
        'p_value': 'p-value',
        'p_value_fdr': 'p-value (FDR)',
        'adj_gap_pct': 'Adjusted Gap (%)',
        'ci_low_pct': 'CI Low (%)',
        'ci_high_pct': 'CI High (%)',
        'median_overall': 'Median Overall',
        'median_male': 'Median Male',
        'median_female': 'Median Female',
        'unadjusted_gap_pct': 'Unadjusted Median Gap (%)',
        'weighted_gap': 'Weighted Gap',
        'threshold_pct': 'Threshold (%)',
        'below_threshold_share': 'Share Below Threshold (%)',
        'above_threshold_share': 'Share Above Threshold (%)',
        'p10': 'P10',
        'p25': 'P25',
        'p50': 'P50',
        'p75': 'P75',
        'p90': 'P90',
        'spread_ratio': 'Spread Ratio (P75/P25)',
        'compression_index': 'Compression Index (Top25/Bottom25)',
        'slope_per_year': 'Slope per Year ($)',
        'p90_p10_ratio': 'P90/P10',
        'manager_premium_ratio': 'Manager Premium Ratio',
        'manager_premium_abs': 'Manager Premium ($)',
        'location_premium_index': 'Location Premium Index',

        'payroll_baseline': 'Payroll Baseline',
        'cost_increase_x': 'Cost of X% Increase',
        'cost_bring_to_median': 'Cost of Bring-to-Median',
        'impacted_share': 'Impacted Share (%)',
        'avg_adjustment': 'Average Adjustment',
        'total_adjustment': 'Total Adjustment',
        'cost_pct_of_payroll': 'Cost as % of Payroll',

        'residual_log': 'Residual (log)',
        'residual_pct': 'Residual (%)',
        'residual_abs_gt_10pct_share': 'Share |Residual|>10% (%)',
        'budget_impact_overpay': 'Budget Impact (Overpay)',
        'budget_impact_underpay': 'Budget Impact (Underpay)',

        'female_share': 'Female Share (%)',
        'female_share_ci_low': 'Female Share CI Low (%)',
        'female_share_ci_high': 'Female Share CI High (%)',
        'representation_index': 'Representation Index',

        'outlier_rate_pct': 'Outlier Rate (%)',
        'align_to_band_cost_total': 'Cost to Nearest Band (Total)',
    }

    # Helper functions to export
    def _export_df(df_to_export: pd.DataFrame, base_filename: str):
        export_dataframe(df_to_export, base_filename, OUTPUT_CSV_DIR, OUTPUT_EXCEL_DIR, EXPORT_COLUMN_NAMES)
        logger.debug(f"Exported DataFrame -> base='{base_filename}' into CSV/Excel directories.")
        
        # Add the dataframe to our dictionary for later use (save to image)
        if not df_to_export.empty:
            calculated_dfs[base_filename] = df_to_export
        
        
    # Dictionary to store all calculated DataFrames for later visualization
    calculated_dfs = {}

    # Warm welcome panel
    console.print(
        Panel.fit(
            f"[phase]HR Compensation Analysis[/phase]\n"
            f"[muted]Project folder:[/muted] [bold]{os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME)}[/bold]",
            border_style="bright_cyan",
            title="Initialization",
            subtitle="Ready to analyze",
        )
    )

    timings = {}
    t_start = time.perf_counter()

    # ==============================================================================
    # PHASE 1: DATA LOADING AND PREPARATION
    # ==============================================================================
    console.print(Rule(title="[phase]Phase 1: Data Preparation[/phase]", style="bright_cyan"))
    t_phase1 = time.perf_counter()

    EXPECTED_COLUMNS = {
        'employee_id', 'name', 'age', 'gender', 'department', 'job_title',
        'experience_years', 'education_level', 'location', 'salary'
    }

    with Progress(
        SpinnerColumn(spinner_name="simpleDots", style="bright_magenta"),
        TextColumn("[progress.description]{task.description}", style="bright_magenta"),
        BarColumn(bar_width=None, style="blue", complete_style="cyan", finished_style="green"),
        TextColumn("{task.completed}/{task.total} • "),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as prep_progress:
        prep_task = prep_progress.add_task("[bold magenta]Preparing dataset[/]", total=7)

        # Normalize column names
        df_raw = df_raw_data.copy()
        df_raw.columns = df_raw.columns.str.strip().str.lower().str.replace(' ', '_')
        prep_progress.advance(prep_task)

        # Validate expected schema
        missing_cols = EXPECTED_COLUMNS - set(df_raw.columns)
        if missing_cols:
            console.print(
                Panel.fit(
                    f"[error]Missing required columns:[/error] {sorted(missing_cols)}",
                    border_style="red",
                    title="Schema Error"
                )
            )
            raise ValueError(f"Input data missing required columns: {missing_cols}")
        prep_progress.advance(prep_task)

        # Clean string columns whitespace only (no destructive operations)
        str_cols = df_raw.select_dtypes(include=['object']).columns.tolist()
        for col in str_cols:
            df_raw[col] = df_raw[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
        prep_progress.advance(prep_task)

        # Ensure numeric types are numeric
        for col in ['age', 'experience_years', 'salary']:
            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')
        if df_raw[['age', 'experience_years', 'salary']].isna().any().any():
            console.print(
                Panel.fit(
                    "[error]Numeric columns ('age','experience_years','salary') contain invalid values.[/error]\n"
                    "Please fix the source data.",
                    border_style="red",
                    title="Validation Error",
                )
            )
            raise ValueError("Numeric columns contain invalid values after coercion.")

        # Harmonize Gender values
        df_raw['gender'] = df_raw['gender'].str.title().replace({'F': 'Female', 'M': 'Male'})
        if not set(df_raw['gender'].unique()).issubset({'Male', 'Female'}):
            console.print(
                Panel.fit(
                    "[warn]Gender contains unexpected values. Only 'Male'/'Female' are supported in equity models.[/warn]",
                    border_style="yellow",
                    title="Gender normalization",
                )
            )
        prep_progress.advance(prep_task)

        # Feature engineering: cohorts and buckets
        # Experience buckets: [0–4], [5–9], [10–19], [20+]
        exp_bins = [0, 5, 10, 20, np.inf]
        exp_labels = ['0–4', '5–9', '10–19', '20+']
        df_raw['experience_bucket'] = pd.cut(df_raw['experience_years'], bins=exp_bins, labels=exp_labels, right=False, include_lowest=True)

        # Age cohorts (audit only): [21–29], [30–39], [40–49], [50–60]
        age_bins = [21, 30, 40, 50, 60]
        age_labels = ['21–29', '30–39', '40–49', '50–60']
        df_raw['age_cohort'] = pd.cut(df_raw['age'], bins=age_bins, labels=age_labels, right=False, include_lowest=True)

        # Log salary for modeling
        df_raw['log_salary'] = np.log(df_raw['salary'].astype(float).clip(lower=1))
        prep_progress.advance(prep_task)

        # Categorical dtypes for modeling/grouping consistency
        for col in ['gender', 'department', 'job_title', 'education_level', 'location', 'experience_bucket', 'age_cohort']:
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].astype('category')

        # Lightweight cluster keys
        df_raw['cluster_cse02'] = (
            df_raw['job_title'].astype(str) + ' | ' +
            df_raw['location'].astype(str) + ' | ' +
            df_raw['experience_bucket'].astype(str)
        )
        df_raw['cluster_heavy'] = (
            df_raw['job_title'].astype(str) + ' | ' +
            df_raw['department'].astype(str) + ' | ' +
            df_raw['location'].astype(str) + ' | ' +
            df_raw['experience_bucket'].astype(str) + ' | ' +
            df_raw['education_level'].astype(str)
        )
        prep_progress.advance(prep_task)

        # Final tidy order (no row removal; duplicates assumed 0 from profile; we keep as-is)
        final_cols = [
            'employee_id', 'name', 'age', 'age_cohort', 'gender', 'department', 'job_title',
            'experience_years', 'experience_bucket', 'education_level', 'location', 'salary',
            'log_salary', 'cluster_cse02', 'cluster_heavy'
        ]
        df_clean = df_raw[final_cols].copy()
        prep_progress.advance(prep_task)

    console.print(
        Panel.fit(
            f"[good]Master DataFrame 'df_clean' is ready[/good]\n"
            f"[muted]Dimensions:[/muted] {df_clean.shape[0]} rows, {df_clean.shape[1]} columns\n"
            f"[muted]Output path:[/muted] {os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME)}",
            title="Data prepared",
            border_style="green",
        )
    )
    _show_dataset_snapshot(df_clean, title="Dataset snapshot (df_clean)")
    timings["phase_1_prep"] = time.perf_counter() - t_phase1
    logger.success(f"Phase 1 completed in {timings['phase_1_prep']:.2f}s")

    # ==============================================================================
    # Helper analytical functions (econometrics & aggregations)
    # ==============================================================================

    def _safe_ols_log_salary_gender(df_in: pd.DataFrame):
        """
        OLS on log(salary) with robust SE. Controls: Job_Title, Department, Location,
        Experience_Years, Education_Level, Age. Gender effect is Female vs Male.
        Returns stats for Female coefficient.
        """
        # Set explicit reference for Gender to ensure interpretability
        # Use Patsy treatment coding with 'Male' reference
        formula = (
            'log_salary ~ C(gender, Treatment(reference="Male")) '
            '+ C(job_title) + C(department) + C(location) '
            '+ experience_years + C(education_level) + age'
        )
        try:
            model = smf.ols(formula=formula, data=df_in).fit(cov_type='HC3')
            term = 'C(gender, Treatment(reference="Male"))[T.Female]'
            if term not in model.params.index:
                return None
            beta = model.params[term]
            se = model.bse[term]
            pval = model.pvalues[term]
            ci_low, ci_high = model.conf_int().loc[term].values
            # Transform from log points to percent effect: (exp(beta)-1)*100
            to_pct = lambda x: (np.exp(x) - 1) * 100
            return {
                'beta_female_log': beta,
                'se': se,
                'p_value': pval,
                'adj_gap_pct': to_pct(beta),
                'ci_low_pct': to_pct(ci_low),
                'ci_high_pct': to_pct(ci_high)
            }
        except Exception as e:
            logger.warning(f"OLS failed: {e}")
            return None

    def _quantile_regressions_global(df_in: pd.DataFrame, taus=(0.25, 0.5, 0.75)):
        """
        Quantile Regression for specified quantiles on log salary with the same controls.
        Returns a DataFrame with Female effects for each tau (no robust SE bootstrap here).
        """
        # Build design matrix manually to reuse in QuantReg
        # Use pandas.get_dummies with drop_first to mirror treatment coding
        X = pd.get_dummies(
            df_in[['gender', 'job_title', 'department', 'location', 'education_level']],
            drop_first=True
        )
        # Ensure 'gender_Female' exists; if not, there may be single-gender data
        if 'gender_Female' not in X.columns:
            return pd.DataFrame(columns=['tau', 'beta_female_log', 'adj_gap_pct'])
        X = pd.concat([X, df_in[['experience_years', 'age']].reset_index(drop=True)], axis=1)
        X = sm.add_constant(X, has_constant='add')
        y = df_in['log_salary'].values

        out_rows = []
        for tau in taus:
            try:
                qr_model = QuantReg(y, X).fit(q=tau)
                beta = qr_model.params.get('gender_Female', np.nan)
                out_rows.append({
                    'tau': tau,
                    'beta_female_log': beta,
                    'adj_gap_pct': (np.exp(beta) - 1) * 100 if pd.notna(beta) else np.nan
                })
            except Exception as e:
                logger.warning(f"Quantile regression failed for tau={tau}: {e}")
                out_rows.append({'tau': tau, 'beta_female_log': np.nan, 'adj_gap_pct': np.nan})
        return pd.DataFrame(out_rows)

    def _unadjusted_median_gap(df_in: pd.DataFrame):
        """Unadjusted median gap: 100 * (median(Male) − median(Female)) / median(Overall)."""
        med_overall = df_in['salary'].median()
        med_male = df_in.loc[df_in['gender'] == 'Male', 'salary'].median()
        med_female = df_in.loc[df_in['gender'] == 'Female', 'salary'].median()
        gap_pct = 100.0 * (med_male - med_female) / med_overall if med_overall > 0 else np.nan
        return {
            'median_overall': med_overall,
            'median_male': med_male,
            'median_female': med_female,
            'unadjusted_gap_pct': gap_pct
        }

    def _segment_adjusted_gender_gaps(df_in: pd.DataFrame, segment: str, min_n=30):
        """
        For each level of the given segment variable, fit local OLS (same controls except segment itself)
        and extract Female effect. Apply FDR correction within this family of tests.
        """
        results = []
        for seg_val, g in df_in.groupby(segment, observed=True):
            g = g.dropna(subset=['salary', 'log_salary', 'gender'])
            n = len(g)
            n_f = (g['gender'] == 'Female').sum()
            n_m = (g['gender'] == 'Male').sum()
            if n < min_n or n_f == 0 or n_m == 0:
                continue
            # Local OLS excludes the segment var to avoid perfect multicollinearity
            try:
                local_dict = _safe_ols_log_salary_gender(g)
            except Exception as e:
                logger.warning(f"Local OLS failed for {segment}={seg_val}: {e}")
                local_dict = None
            if local_dict:
                results.append({
                    'segment': segment,
                    'segment_value': seg_val,
                    'n': n,
                    'n_female': n_f,
                    'n_male': n_m,
                    **local_dict
                })
        df_res = pd.DataFrame(results)
        if df_res.empty:
            return df_res
        # FDR correction on p-values within this segment family
        try:
            reject, p_adj, _, _ = multipletests(df_res['p_value'], alpha=FDR_ALPHA, method='fdr_bh')
            df_res['p_value_fdr'] = p_adj
        except Exception as e:
            logger.warning(f"FDR correction failed for segment '{segment}': {e}")
            df_res['p_value_fdr'] = df_res['p_value']
        return df_res

    def _compute_bands(df_in: pd.DataFrame, cluster_col: str):
        """Compute p10/p25/p50/p75/p90 and out-of-band shares per cluster."""
        quants = df_in.groupby(cluster_col, observed=True)['salary'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
        quants.columns = ['p10', 'p25', 'p50', 'p75', 'p90']
        quants = quants.reset_index()
        # Spread ratio
        quants['spread_ratio'] = quants['p75'] / quants['p25']
        # Add headcount and out-of-band shares
        gsize = df_in.groupby(cluster_col, observed=True).size().rename('n').reset_index()
        merged = quants.merge(gsize, on=cluster_col, how='left')
        df_in = df_in.merge(quants[[cluster_col, 'p25', 'p75']], on=cluster_col, how='left')
        oob = df_in.assign(
            below=lambda d: (d['salary'] < d['p25']).astype(int),
            above=lambda d: (d['salary'] > d['p75']).astype(int),
        ).groupby(cluster_col, observed=True).agg(
            below=('below', 'mean'),
            above=('above', 'mean')
        ).reset_index()
        oob['below'] = oob['below'] * 100.0
        oob['above'] = oob['above'] * 100.0
        out = merged.merge(oob, on=cluster_col, how='left').rename(
            columns={'below': 'below_threshold_share', 'above': 'above_threshold_share'}
        )
        return out

    def _compression_metrics(df_in: pd.DataFrame, cluster_cols: list):
        """Compute compression metrics per cluster: ratio of medians Top25/Bottom25, slope per year, p90/p10."""
        rows = []
        for keys, g in df_in.groupby(cluster_cols, observed=True):
            g = g.dropna(subset=['salary', 'experience_years'])
            n = len(g)
            if n < MIN_SEGMENT_N:
                continue
            q25 = g['experience_years'].quantile(0.25)
            q75 = g['experience_years'].quantile(0.75)
            low = g[g['experience_years'] <= q25]
            high = g[g['experience_years'] >= q75]
            if low.empty or high.empty or low['salary'].median() == 0:
                continue
            comp_index = high['salary'].median() / low['salary'].median()
            # OLS slope salary ~ experience_years
            try:
                m = smf.ols('salary ~ experience_years', data=g).fit()
                slope = m.params.get('experience_years', np.nan)
            except Exception:
                slope = np.nan
            p90_p10 = g['salary'].quantile(0.90) / max(g['salary'].quantile(0.10), 1)
            key_dict = dict(zip(cluster_cols, keys if isinstance(keys, tuple) else (keys,)))
            rows.append({**key_dict, 'n': n, 'compression_index': comp_index, 'slope_per_year': slope, 'p90_p10_ratio': p90_p10})
        return pd.DataFrame(rows)

    def _manager_premium(df_in: pd.DataFrame):
        """Manager premium vs IC in Department×Location; exclude interns from IC by default."""
        df = df_in.copy()
        df['role_class'] = np.where(df['job_title'].isin(['Manager', 'Executive']), 'Manager', 'IC')
        df = df[df['job_title'].isin(['Manager', 'Executive', 'Analyst', 'Engineer'])].copy()
        rows = []
        for (dep, loc), g in df.groupby(['department', 'location'], observed=True):
            n = len(g)
            g_m = g[g['role_class'] == 'Manager']
            g_ic = g[g['role_class'] == 'IC']
            if len(g_m) < 10 or len(g_ic) < 10:
                continue
            med_m = g_m['salary'].median()
            med_ic = g_ic['salary'].median()
            if med_ic <= 0:
                continue
            ratio = med_m / med_ic
            rows.append({
                'department': dep, 'location': loc, 'n': n,
                'manager_premium_ratio': ratio, 'manager_premium_abs': med_m - med_ic
            })
        return pd.DataFrame(rows)

    def _mispricing_map(df_in: pd.DataFrame):
    
        df_local = df_in.copy()
    
        # === SOLUTION 1A: Robust Column Validation ===
        required_base_cols = ['employee_id', 'log_salary', 'salary']
        missing_base = set(required_base_cols) - set(df_local.columns)
        if missing_base:
            logger.error(f"Missing critical columns for mispricing analysis: {missing_base}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
        # === SOLUTION 1B: Smart Experience Bucket Creation ===
        if 'experience_bucket' not in df_local.columns:
            if 'experience_years' in df_local.columns:                
                exp_years = pd.to_numeric(df_local['experience_years'], errors='coerce')
                exp_bins = [0, 5, 10, 20, np.inf]
                exp_labels = ['0-4', '5-9', '10-19', '20+']
            
                try:
                    df_local['experience_bucket'] = pd.cut(
                        exp_years,
                        bins=exp_bins,
                        labels=exp_labels,
                        right=False,
                        include_lowest=True
                    )
                    # NaN
                    df_local['experience_bucket'] = df_local['experience_bucket'].cat.add_categories('Unknown')
                    df_local['experience_bucket'].fillna('Unknown', inplace=True)
                    #logger.info("Successfully created 'experience_bucket' from 'experience_years'")
                except Exception as e:
                    logger.warning(f"Failed to create experience_bucket from experience_years: {e}")
                    df_local['experience_bucket'] = 'Unknown'
            else:
                logger.warning("Neither 'experience_bucket' nor 'experience_years' found. Using placeholder.")
                df_local['experience_bucket'] = 'Unknown'
        
        if not pd.api.types.is_categorical_dtype(df_local['experience_bucket']):
            df_local['experience_bucket'] = df_local['experience_bucket'].astype('category')

        # === SOLUTION 1C: Dynamic Formula Building ===        
        potential_predictors = {
            'job_title': 'C(job_title)',
            'department': 'C(department)', 
            'location': 'C(location)',
            'experience_years': 'experience_years',
            'education_level': 'C(education_level)',
            'age': 'age',
            'experience_bucket': 'C(experience_bucket)'
        }

        available_predictors = []
        for col, formula_part in potential_predictors.items():
            if col in df_local.columns:                
                if col in ['job_title', 'department', 'location', 'education_level', 'experience_bucket']:
                    unique_vals = df_local[col].dropna().nunique()
                    if unique_vals > 1:
                        available_predictors.append(formula_part)
                        #logger.debug(f"Added {col} to model (unique values: {unique_vals})")
                    else:
                        logger.warning(f"Skipped {col}: insufficient variation (unique: {unique_vals})")
                else:
                    # Numeric
                    if not df_local[col].isna().all():
                        available_predictors.append(formula_part)
                        #logger.debug(f"Added numeric {col} to model")
                    else:
                        logger.warning(f"Skipped {col}: all values are NaN")

        if not available_predictors:
            logger.error("No valid predictors available for mispricing model")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        formula = f"log_salary ~ {' + '.join(available_predictors)}"
        #logger.info(f"Using model formula: {formula}")

        # === SOLUTION 1D: Robust Model Fitting ===
        try:            
            model_data = df_local.dropna(subset=['log_salary'] + [col for col in potential_predictors.keys() if col in df_local.columns])

            if len(model_data) < 50:
                logger.error(f"Insufficient data for modeling: {len(model_data)} rows")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            model = smf.ols(formula=formula, data=model_data).fit(cov_type='HC3')
            #logger.info(f"Model fitted successfully on {len(model_data)} observations")

        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # === SOLUTION 1E: Safe Residuals Computation ===
        try:            
            y_log = model_data['log_salary'].values
            yhat_log = model.predict(model_data)
            resid_log = y_log - yhat_log
            resid_pct = (np.exp(resid_log) - 1) * 100.0
           
            df_res = model_data[['employee_id', 'salary']].copy()
            
            seg_cols = ['department', 'job_title', 'location', 'experience_bucket', 'education_level', 'gender']
            available_seg_cols = [col for col in seg_cols if col in model_data.columns]
            for col in available_seg_cols:
                df_res[col] = model_data[col]

            df_res['residual_log'] = resid_log
            df_res['residual_pct'] = resid_pct

            # === SOLUTION 1F: Safe Aggregations ===
            agg_rows = []
            for col in available_seg_cols:
                if df_res[col].nunique() > 1:
                    try:
                        tmp = df_res.groupby(col, observed=True).agg(
                            n=('employee_id', 'count'),
                            mean_residual_pct=('residual_pct', 'mean'),
                            residual_abs_gt_10pct_share=('residual_pct', lambda x: (np.abs(x) > 10).mean() * 100.0)
                        ).reset_index().rename(columns={col: 'segment_value'})
                        tmp['segment'] = col
                        agg_rows.append(tmp)
                    except Exception as e:
                        logger.warning(f"Aggregation failed for segment {col}: {e}")

            df_seg = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()
            
            df_res['residual_$'] = df_res['salary'] * df_res['residual_pct'] / 100.0
            budget_overpay = df_res.loc[df_res['residual_$'] > 0, 'residual_$'].sum()
            budget_underpay = df_res.loc[df_res['residual_$'] < 0, 'residual_$'].sum()

            df_budget = pd.DataFrame([{
                'budget_impact_overpay': float(budget_overpay),
                'budget_impact_underpay': float(abs(budget_underpay)),
                'model_observations': len(model_data),
                'model_r_squared': model.rsquared
            }])

            #logger.success(f"Mispricing analysis completed successfully")
            return df_res, df_seg, df_budget

        except Exception as e:
            logger.error(f"Residuals computation failed: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _location_premiums(df_in: pd.DataFrame, baseline_location: str):
        """Compute location premium index by cluster Role×Seniority×Department relative to baseline location."""
        cl_cols = ['job_title', 'experience_bucket', 'department']
        # Median salary by cluster×location
        med = df_in.groupby(cl_cols + ['location'], observed=True)['salary'].median().reset_index(name='median_salary')
        # Split baseline medians
        base = med[med['location'] == baseline_location].rename(columns={'median_salary': 'median_baseline'})
        merged = med.merge(base[cl_cols + ['median_baseline']], on=cl_cols, how='left')
        merged['location_premium_index'] = merged['median_salary'] / merged['median_baseline']
        # Add headcount
        ncount = df_in.groupby(cl_cols + ['location'], observed=True).size().rename('n').reset_index()
        out = merged.merge(ncount, on=cl_cols + ['location'], how='left')
        return out

    def _equity_cost_to_parity(df_in: pd.DataFrame, corridor=0.02):
        """
        Compute cost to close gender gap to within ±corridor by bringing underpaid group's median
        to the other group's median within cluster (role×dept×loc×seniority×edu).
        We only raise pay (no reductions).
        """
        cluster = 'cluster_heavy'
        # Median by cluster×gender
        med = df_in.groupby([cluster, 'gender'], observed=True)['salary'].median().unstack()
        med = med.rename(columns={'Female': 'median_female', 'Male': 'median_male'})
        med = med.reset_index()

        # Determine target per cluster and target gender
        med['gap_flag_female_under'] = med['median_female'] * (1 + corridor) < med['median_male']
        med['gap_flag_male_under'] = med['median_male'] * (1 + corridor) < med['median_female']

        # For simplicity and compliance, we prioritize raising the underpaid group only
        adj_rows = []
        baseline_payroll = df_in['salary'].sum()

        # Index employees for quick join
        base = df_in[['employee_id', 'salary', 'gender', cluster] + ['department', 'job_title', 'location',
                                                                     'experience_bucket', 'education_level']].copy()

        # Female underpaid case
        female_clusters = set(med.loc[med['gap_flag_female_under'], cluster].astype(str).tolist())
        male_clusters = set(med.loc[med['gap_flag_male_under'], cluster].astype(str).tolist())

        med_index = med.set_index(cluster)
        # Female raise to male median
        for cl in female_clusters:
            target = med_index.loc[cl, 'median_male']
            sub = base[(base[cluster].astype(str) == cl) & (base['gender'] == 'Female')].copy()
            sub['target_salary'] = target
            sub['delta'] = (sub['target_salary'] - sub['salary']).clip(lower=0)
            adj_rows.append(sub)
        # Male raise to female median (very rare; still compute symmetrically)
        for cl in male_clusters:
            target = med_index.loc[cl, 'median_female']
            sub = base[(base[cluster].astype(str) == cl) & (base['gender'] == 'Male')].copy()
            sub['target_salary'] = target
            sub['delta'] = (sub['target_salary'] - sub['salary']).clip(lower=0)
            adj_rows.append(sub)

        if len(adj_rows) == 0:
            df_adj = pd.DataFrame(columns=base.columns.tolist() + ['target_salary', 'delta'])
        else:
            df_adj = pd.concat(adj_rows, ignore_index=True)

        # Summaries
        total_adjustment = df_adj['delta'].sum() if not df_adj.empty else 0.0
        impacted_share = 100.0 * (df_adj['employee_id'].nunique() / df_in['employee_id'].nunique()) if not df_adj.empty else 0.0
        avg_adjustment = df_adj['delta'].mean() if not df_adj.empty else 0.0
        cost_pct_of_payroll = 100.0 * total_adjustment / baseline_payroll if baseline_payroll > 0 else 0.0

        summary = pd.DataFrame([{
            'total_adjustment': total_adjustment,
            'impacted_share': impacted_share,
            'avg_adjustment': avg_adjustment,
            'cost_pct_of_payroll': cost_pct_of_payroll
        }])
        # Also roll-up by cluster
        by_cluster = df_adj.groupby(cluster, as_index=False)['delta'].sum().rename(columns={'delta': 'total_adjustment'})

        return df_adj, by_cluster, summary

    def _bring_to_median(df_in: pd.DataFrame, cluster_col: str):
        """Scenario: bring everyone below cluster median up to p50 (neutral reference)."""
        med = df_in.groupby(cluster_col, observed=True)['salary'].median().reset_index(name='p50')
        tmp = df_in[['employee_id', 'salary', cluster_col]].merge(med, on=cluster_col, how='left')
        tmp['delta'] = (tmp['p50'] - tmp['salary']).clip(lower=0)
        total_adjustment = tmp['delta'].sum()
        impacted_share = 100.0 * (tmp['delta'] > 0).mean()
        avg_adjustment = tmp.loc[tmp['delta'] > 0, 'delta'].mean() if (tmp['delta'] > 0).any() else 0.0
        baseline_payroll = df_in['salary'].sum()
        cost_pct_of_payroll = 100.0 * total_adjustment / baseline_payroll if baseline_payroll > 0 else 0.0

        summary = pd.DataFrame([{
            'total_adjustment': total_adjustment,
            'impacted_share': impacted_share,
            'avg_adjustment': avg_adjustment,
            'cost_pct_of_payroll': cost_pct_of_payroll
        }])
        by_cluster = tmp.groupby(cluster_col, as_index=False)['delta'].sum().rename(columns={'delta': 'total_adjustment'})
        return tmp, by_cluster, summary

    def _female_representation(df_in: pd.DataFrame, seg_vars: list):
        """Female share by segment with Wilson (95%) CI and representation index vs company average."""
        company_female_share = (df_in['gender'] == 'Female').mean()
        frames = []
        for s in seg_vars:
            g = df_in.groupby(s, observed=True)['gender'].apply(lambda x: (x == 'Female').sum()).reset_index(name='n_female')
            g_tot = df_in.groupby(s, observed=True)['gender'].size().reset_index(name='n_total')
            merged = g.merge(g_tot, on=s)
            merged['female_share'] = merged['n_female'] / merged['n_total']
            # Wilson CI
            ci_low, ci_high = proportion_confint(merged['n_female'], merged['n_total'], alpha=0.05, method='wilson')
            merged['female_share_ci_low'] = ci_low
            merged['female_share_ci_high'] = ci_high
            merged['representation_index'] = merged['female_share'] / company_female_share if company_female_share > 0 else np.nan
            merged['female_share'] *= 100.0
            merged['female_share_ci_low'] *= 100.0
            merged['female_share_ci_high'] *= 100.0
            merged = merged.rename(columns={s: 'segment_value'})
            merged['segment'] = s
            frames.append(merged[['segment', 'segment_value', 'n_total', 'n_female',
                                  'female_share', 'female_share_ci_low', 'female_share_ci_high', 'representation_index']])
        return pd.concat(frames, ignore_index=True)

    def _outliers_and_alignment_cost(df_in: pd.DataFrame, cluster_col: str):
        """
        Outlier rate using z-score within cluster; cost to align to nearest band boundary (p10/p90).
        """
        # Compute per-cluster stats
        stats = df_in.groupby(cluster_col, observed=True)['salary'].agg(['mean', 'std']).reset_index().rename(columns={'mean': 'mu', 'std': 'sigma'})
        bands = df_in.groupby(cluster_col, observed=True)['salary'].quantile([0.1, 0.9]).unstack().reset_index().rename(columns={0.1: 'p10', 0.9: 'p90'})
        st = stats.merge(bands, on=cluster_col, how='left')
        df_tmp = df_in[['employee_id', 'salary', cluster_col]].merge(st, on=cluster_col, how='left')
        df_tmp['z'] = (df_tmp['salary'] - df_tmp['mu']) / df_tmp['sigma'].replace(0, np.nan)
        df_tmp['is_outlier'] = (df_tmp['z'].abs() > 2).fillna(False)
        # Rate by cluster
        rate = df_tmp.groupby(cluster_col, observed=True)['is_outlier'].mean().reset_index(name='outlier_rate_pct')
        rate['outlier_rate_pct'] = rate['outlier_rate_pct'] * 100.0
        # Cost to align to nearest boundary (only for outliers outside p10/p90):
        df_out = df_tmp[df_tmp['is_outlier']].copy()
        df_out['target'] = np.where(df_out['salary'] < df_out['p10'], df_out['p10'], df_out['p90'])
        df_out['delta'] = (df_out['target'] - df_out['salary']).clip(lower=0)
        # Cost per cluster:
        cost = df_out.groupby(cluster_col, observed=True)['delta'].sum().reset_index(name='align_to_band_cost_total')
        out = rate.merge(cost, on=cluster_col, how='left').fillna({'align_to_band_cost_total': 0.0})
        return out, df_out

    # ==============================================================================
    # PHASE 2: BUSINESS QUESTIONS
    # ==============================================================================
    console.print(Rule(title="[phase]Phase 2: Answering Business Questions[/phase]", style="bright_cyan"))
    t_phase2 = time.perf_counter()

    questions = [
        "PEQ-01 Global Pay Equity (OLS+QR) and Unadjusted Gap",
        "PEQ-01 Stratified Adjusted Gaps with FDR",
        "PEQ-02 Top segments by adjusted gender gap and coverage",
        "PEQ-03 Cost-to-Parity (±corridor) on cluster_heavy",
        "BUD-01 Payroll baseline and scenarios (X% and bring-to-median)",
        "CSE-02 Offer bands p25–p50–p75 by role×location×seniority (out-of-band shares)",
        "CPR-01 Compression indices and progression slope by role×location×department",
        "CPR-02 Manager premium vs IC by department×location",
        "CSE-01 Mispricing pockets via residuals (no Gender)",
        "GEO-01 Location premiums vs baseline",
        "EDU-01 Education ROI (Master/PhD vs Bachelor)",
        "DEI-01 Female representation by segments",
        "RISK-01 Outlier rates and cost to align to band"
    ]
    total_questions = len(questions)
    q_timings = {}

    with Progress(
        SpinnerColumn(spinner_name="simpleDots", style="yellow"),
        TextColumn("[progress.description]{task.description}", style="bright_magenta"),
        BarColumn(bar_width=None, style="blue", complete_style="cyan", finished_style="green"),
        TextColumn("{task.completed}/{task.total} • "),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as q_progress:
        q_task = q_progress.add_task("[bold yellow]Business Questions[/]", total=total_questions)

        # 1) PEQ-01: Global OLS + QR and unadjusted gap
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> PEQ-01: Global Pay Equity (OLS + Quantile Regression)[/question]", border_style="cyan"))
        peq01_ols = _safe_ols_log_salary_gender(df_clean)
        df_peq01_ols = pd.DataFrame([peq01_ols]) if peq01_ols else pd.DataFrame(columns=[
            'beta_female_log', 'se', 'p_value', 'adj_gap_pct', 'ci_low_pct', 'ci_high_pct'
        ])
        df_peq01_ols.insert(0, 'model', 'OLS')
        _export_df(df_peq01_ols, 'peq01_global_ols')

        df_peq01_qr = _quantile_regressions_global(df_clean, taus=(0.25, 0.5, 0.75))
        _export_df(df_peq01_qr, 'peq01_global_quantile_regression')

        unadj = _unadjusted_median_gap(df_clean)
        df_peq01_unadjusted = pd.DataFrame([unadj])
        _export_df(df_peq01_unadjusted, 'peq01_unadjusted_median_gap')       


        q_progress.advance(q_task)
        q_timings["PEQ-01 Global"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 2) PEQ-01: Stratified adjusted gaps with FDR by key families
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> PEQ-01: Stratified Adjusted Gaps (with FDR)[/question]", border_style="cyan"))
        segment_vars = ['department', 'job_title', 'location', 'experience_bucket', 'education_level']
        frames = []
        for seg in segment_vars:
            frames.append(_segment_adjusted_gender_gaps(df_clean, seg, min_n=MIN_SEGMENT_N))
        df_peq01_segments = pd.concat(frames, ignore_index=True) if len(frames) else pd.DataFrame()
        _export_df(df_peq01_segments, 'peq01_segment_adjusted_gaps_fdr')

        # Share of segments with |gap| > 5% and p_fdr < 0.05
        if not df_peq01_segments.empty:
            flags = df_peq01_segments.assign(
                is_sig=lambda d: (d['p_value_fdr'] < FDR_ALPHA) & (d['adj_gap_pct'].abs() > 5)
            ).groupby('segment', observed=True)['is_sig'].mean().reset_index(name='share_sig_segments')
            _export_df(flags, 'peq01_segments_share_sig_gaps')
        q_progress.advance(q_task)
        q_timings["PEQ-01 Stratified"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 3) PEQ-02: Top segments by adjusted gap and coverage
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> PEQ-02: Top segments by adjusted gender gap[/question]", border_style="cyan"))
        if not df_peq01_segments.empty:
            df_peq02_top = df_peq01_segments.copy()
            df_peq02_top['weighted_gap'] = df_peq02_top['adj_gap_pct'] * df_peq02_top['n']
            # Focus N≥30 and FDR-significant
            df_peq02_top = df_peq02_top[(df_peq02_top['n'] >= MIN_SEGMENT_N)]
            df_peq02_top_sig = df_peq02_top[df_peq02_top['p_value_fdr'] < FDR_ALPHA].copy()
            df_peq02_top_abs = df_peq02_top_sig.reindex(df_peq02_top_sig['adj_gap_pct'].abs().sort_values(ascending=False).index).head(50)
            _export_df(df_peq02_top_abs, 'peq02_top_segments_by_gap')

            # Share of employees in segments with significant gap > 5%
            if not df_peq02_top_sig.empty:
                impacted_n = df_peq02_top_sig.loc[df_peq02_top_sig['adj_gap_pct'].abs() > 5, 'n'].sum()
                share_employees_impacted = 100.0 * impacted_n / df_clean.shape[0]
            else:
                share_employees_impacted = 0.0
            df_peq02_summary = pd.DataFrame([{'share_employees_in_sig_gap_segments': share_employees_impacted}])
            _export_df(df_peq02_summary, 'peq02_summary')
        q_progress.advance(q_task)
        q_timings["PEQ-02"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 4) PEQ-03: Cost-to-Parity (±corridor) on cluster_heavy
        q_start = time.perf_counter()
        console.print(Panel.fit(f"[question]-> PEQ-03: Cost-to-Parity (corridor ±{int(EQUITY_CORRIDOR*100)}%)[/question]", border_style="cyan"))
        df_peq03_adj, df_peq03_by_cluster, df_peq03_summary = _equity_cost_to_parity(df_clean, corridor=EQUITY_CORRIDOR)
        _export_df(df_peq03_adj, 'peq03_adjustments_detailed')
        _export_df(df_peq03_by_cluster, 'peq03_cost_by_cluster')
        _export_df(df_peq03_summary, 'peq03_summary')
        q_progress.advance(q_task)
        q_timings["PEQ-03"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 5) BUD-01: Payroll baseline and scenarios
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> BUD-01: Payroll scenarios (X% and bring-to-median)[/question]", border_style="cyan"))
        payroll_baseline = df_clean['salary'].sum()
        cost_increase_x = payroll_baseline * INCREASE_RATE

        df_bring_emp, df_bring_by_cluster, df_bring_summary = _bring_to_median(df_clean, cluster_col='cluster_cse02')

        df_bud01_summary = pd.DataFrame([{
            'payroll_baseline': payroll_baseline,
            'cost_increase_x': cost_increase_x,
            'cost_bring_to_median': float(df_bring_summary['total_adjustment'].iloc[0]),
        }])
        _export_df(df_bud01_summary, 'bud01_summary')
        # Add cluster bands context
        df_bands = _compute_bands(df_clean, 'cluster_cse02')
        _export_df(df_bands, 'cse02_bands_by_cluster')
        _export_df(df_bring_by_cluster, 'bud01_bring_to_median_cost_by_cluster')
        _export_df(df_bring_emp, 'bud01_bring_to_median_employee_deltas')
        q_progress.advance(q_task)
        q_timings["BUD-01"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 6) CSE-02: Offer bands p25–p50–p75 and OOB shares
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> CSE-02: Offer bands p25–p50–p75 and OOB shares[/question]", border_style="cyan"))
        # df_bands already computed in previous step (re-use)
        # Flag clusters where spread ratio is too high/low or OOB is large
        df_cse02_flags = df_bands.assign(
            spread_flag=lambda d: np.where(d['spread_ratio'] > 1.6, 'Too wide',
                                  np.where(d['spread_ratio'] < 1.2, 'Too narrow', 'OK'))
        )
        _export_df(df_cse02_flags, 'cse02_bands_flags')
        q_progress.advance(q_task)
        q_timings["CSE-02"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 7) CPR-01: Compression metrics by role×location×department
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> CPR-01: Compression & progression by role×location×department[/question]", border_style="cyan"))
        df_cpr01 = _compression_metrics(df_clean, ['job_title', 'location', 'department'])
        _export_df(df_cpr01, 'cpr01_compression_by_cluster')
        # Flagging problem clusters: compression index < 1.15 or slope < 1500
        if not df_cpr01.empty:
            df_cpr01_flags = df_cpr01.assign(
                compression_flag=lambda d: np.where((d['compression_index'] < 1.15) | (d['slope_per_year'] < 1500), 'Risk', 'OK')
            )
            _export_df(df_cpr01_flags, 'cpr01_compression_flags')
        q_progress.advance(q_task)
        q_timings["CPR-01"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 8) CPR-02: Manager premium vs IC
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> CPR-02: Manager premium vs IC[/question]", border_style="cyan"))
        df_cpr02 = _manager_premium(df_clean)
        _export_df(df_cpr02, 'cpr02_manager_premium_by_dep_loc')
        q_progress.advance(q_task)
        q_timings["CPR-02"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 9) CSE-01: Mispricing pockets via residuals
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> CSE-01: Mispricing map via residuals (no Gender)[/question]", border_style="cyan"))
        df_cse01_emp, df_cse01_seg, df_cse01_budget = _mispricing_map(df_clean)
        _export_df(df_cse01_emp, 'cse01_residuals_employee')
        _export_df(df_cse01_seg, 'cse01_residuals_by_segment')
        _export_df(df_cse01_budget, 'cse01_budget_impact_over_under')
        q_progress.advance(q_task)
        q_timings["CSE-01"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 10) GEO-01: Location premiums vs baseline
        q_start = time.perf_counter()
        console.print(Panel.fit(f"[question]-> GEO-01: Location premiums vs baseline ({BASELINE_LOCATION})[/question]", border_style="cyan"))
        df_geo01 = _location_premiums(df_clean, baseline_location=BASELINE_LOCATION)
        _export_df(df_geo01, 'geo01_location_premium_by_cluster')
        # Basic scenario helper: move X FTE from src to dst per cluster at median delta
        src, dst, x = GEO_SCENARIO.get('src'), GEO_SCENARIO.get('dst'), GEO_SCENARIO.get('fte_per_cluster', 1)
        if src in df_geo01['location'].unique() and dst in df_geo01['location'].unique():
            # For clusters with both src and dst, compute delta per FTE as median(dst) - median(src)
            med = df_clean.groupby(['job_title', 'experience_bucket', 'department', 'location'], observed=True)['salary'].median().reset_index(name='median')
            pivot = med.pivot_table(index=['job_title', 'experience_bucket', 'department'], columns='location', values='median')
            if src in pivot.columns and dst in pivot.columns:
                tmp = pivot[[src, dst]].dropna().reset_index()
                tmp['delta_per_fte'] = tmp[dst] - tmp[src]
                tmp['scenario_delta'] = tmp['delta_per_fte'] * x
                scenario_total = tmp['scenario_delta'].sum()
                df_geo01_scenario = tmp.rename(columns={
                    'job_title': 'job_title', 'experience_bucket': 'experience_bucket', 'department': 'department'
                })
                _export_df(df_geo01_scenario, 'geo01_scenario_delta_per_cluster')
                _export_df(pd.DataFrame([{'scenario_total_delta': scenario_total, 'src': src, 'dst': dst, 'fte_per_cluster': x}]),
                           'geo01_scenario_summary')
        q_progress.advance(q_task)
        q_timings["GEO-01"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 11) EDU-01: Education ROI
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> EDU-01: Education ROI (Master/PhD vs Bachelor)[/question]", border_style="cyan"))
        # Global model with education effects
        try:
            edu_model = smf.ols(
                'log_salary ~ C(education_level, Treatment(reference="Bachelor")) + '
                'C(job_title) + C(department) + C(location) + experience_years + age',
                data=df_clean
            ).fit(cov_type='HC3')
            # Extract education effects
            rows = []
            for term in [
                'C(education_level, Treatment(reference="Bachelor"))[T.Master]',
                'C(education_level, Treatment(reference="Bachelor"))[T.PhD]'
            ]:
                if term in edu_model.params.index:
                    beta = edu_model.params[term]
                    se = edu_model.bse[term]
                    p = edu_model.pvalues[term]
                    ci_low, ci_high = edu_model.conf_int().loc[term].values
                    rows.append({
                        'term': term,
                        'beta_log': beta,
                        'adj_gap_pct': (np.exp(beta) - 1) * 100,
                        'ci_low_pct': (np.exp(ci_low) - 1) * 100,
                        'ci_high_pct': (np.exp(ci_high) - 1) * 100,
                        'p_value': p
                    })
            df_edu01_global = pd.DataFrame(rows)
        except Exception as e:
            logger.warning(f"EDU-01 global model failed: {e}")
            df_edu01_global = pd.DataFrame()
        _export_df(df_edu01_global, 'edu01_global_education_premia')

        # By Department
        dep_rows = []
        for dep, g in df_clean.groupby('department', observed=True):
            if len(g) < MIN_SEGMENT_N:
                continue
            try:
                em = smf.ols(
                    'log_salary ~ C(education_level, Treatment(reference="Bachelor")) + '
                    'C(job_title) + C(location) + experience_years + age',
                    data=g
                ).fit(cov_type='HC3')
                for term in [
                    'C(education_level, Treatment(reference="Bachelor"))[T.Master]',
                    'C(education_level, Treatment(reference="Bachelor"))[T.PhD]'
                ]:
                    if term in em.params.index:
                        beta = em.params[term]
                        p = em.pvalues[term]
                        dep_rows.append({
                            'department': dep, 'term': term,
                            'adj_gap_pct': (np.exp(beta) - 1) * 100,
                            'p_value': p, 'n': len(g)
                        })
            except Exception:
                continue
        df_edu01_by_dep = pd.DataFrame(dep_rows)
        _export_df(df_edu01_by_dep, 'edu01_by_department')
        q_progress.advance(q_task)
        q_timings["EDU-01"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 12) DEI-01: Female representation
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> DEI-01: Female representation by segments[/question]", border_style="cyan"))
        df_dei01 = _female_representation(df_clean, ['department', 'job_title', 'location'])
        _export_df(df_dei01, 'dei01_female_representation')
        q_progress.advance(q_task)
        q_timings["DEI-01"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

        # 13) RISK-01: Outliers and cost to align to band (p10/p90)
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> RISK-01: Outlier rates and alignment cost[/question]", border_style="cyan"))
        df_risk01_rate, df_risk01_outliers = _outliers_and_alignment_cost(df_clean, 'cluster_cse02')
        _export_df(df_risk01_rate, 'risk01_outlier_rates_by_cluster')
        _export_df(df_risk01_outliers, 'risk01_outliers_detailed')
        q_progress.advance(q_task)
        q_timings["RISK-01"] = time.perf_counter() - q_start
        console.print(Rule(style="bright_black"))

    timings["phase_2_questions"] = time.perf_counter() - t_phase2
    logger.success(f"Phase 2 completed in {timings['phase_2_questions']:.2f}s")
    console.print(Rule(style="bright_black"))


    # ==============================================================================
    # PHASE 3: VISUALIZATION
    # ==============================================================================
    console.print(Rule(title="[phase]Phase 3: Generating Visualizations[/phase]", style="bright_cyan"))
        
    # Loop through all generated reports and create a chart for each if a plotter exists
    for filename, df_data in calculated_dfs.items():
        # We pass the entire dictionary `calculated_dfs` in case a future plot
        # needs to combine data from multiple reports.
        generate_visualizations(filename, df_data, calculated_dfs, OUTPUT_CHARTS_DIR, logger)
        
    logger.success("All visualizations have been generated.")

    # ==============================================================================
    # WRAP-UP: EXECUTION STATS AND OUTPUT OVERVIEW
    # ==============================================================================
    total_elapsed = time.perf_counter() - t_start
    console.print(Rule(title="[phase]Execution summary[/phase]", style="bright_cyan"))

    table = Table(title="Execution timings", show_header=True, header_style="bold")
    table.add_column("Step", style="muted")
    table.add_column("Elapsed", style="bold")
    table.add_row("Phase 1: Data Preparation", f"{timings['phase_1_prep']:.2f}s")
    table.add_row("Phase 2: Business Questions", f"{timings['phase_2_questions']:.2f}s")
    for q, tval in q_timings.items():
        table.add_row(f"  {q}", f"{tval:.2f}s")
    table.add_row("Total runtime", f"{total_elapsed:.2f}s")
    console.print(table)

    _show_dataset_snapshot(df_clean, title="Final master DataFrame (df_clean)")
    _show_output_tree(
        output_root=os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME),
        csv_dir=OUTPUT_CSV_DIR,
        excel_dir=OUTPUT_EXCEL_DIR,
        charts_dir=OUTPUT_CHARTS_DIR,
        title="Saved reports"
    )

    console.print(
        Panel.fit(
            "Analysis complete. All tabular outputs are saved to CSV/Excel.\n"
            "Notes:\n"
            "- Age used only as a control for fairness audits.\n"
            "- Segment tests apply FDR correction and minimum N for reliability.\n"
            "- Residuals model excludes Gender (diagnostic only).",
            title="Notes",
            border_style="cyan",
        )
    )

    console.print(
        Panel.fit(
            f"[good]Completed[/good] • Results in: [bold]{os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME)}[/bold]",
            border_style="green",
        )
    )
    logger.success("All done.")
