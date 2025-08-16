# -*- coding: utf-8 -*-
"""
Generates and saves visualizations from HR analysis DataFrames.

This module provides a suite of functions to create plots for various
HR metrics, such as budget scenarios, compensation structures, pay equity,
and diversity representation. It is designed to be called after the main
analysis script has generated the necessary data files.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import save_matplotlib_figure

# --- Global Plotting Style Configuration ---
# Applied to all generated charts for a consistent look and feel.
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12


def _get_plot_data(df_input: pd.DataFrame, sort_by_col: str, max_categories: int = 20) -> tuple[pd.DataFrame, bool]:
    """
    Sorts and trims a DataFrame to make it suitable for plotting, especially
    when dealing with a large number of categories.

    It returns a truncated DataFrame and a boolean indicating if truncation occurred.
    """
    if len(df_input) > max_categories:
        try:
            # For readability, show the most extreme values (top and bottom).
            df_sorted = df_input.sort_values(by=sort_by_col, ascending=False)
            df_top = df_sorted.head(max_categories // 2)
            df_bottom = df_sorted.tail(max_categories // 2)
            return pd.concat([df_top, df_bottom]), True
        except KeyError:
            # Fallback for cases where the sort column might be missing.
            return df_input.head(max_categories), True
    return df_input, False


def plot_budget_scenarios_summary(df_summary: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes the main budget scenarios: baseline vs. potential increases."""
    df_melted = df_summary.melt(var_name='Scenario', value_name='Cost')
    
    fig, ax = plt.subplots()
    sns.barplot(data=df_melted, x='Scenario', y='Cost', ax=ax, palette="viridis")
    
    ax.set_title('Personnel Cost Scenarios')
    ax.set_ylabel('Amount ($)')
    ax.set_xlabel('Cost Type')
    ax.ticklabel_format(style='plain', axis='y')
    plt.xticks(rotation=15, ha='right')
    
    save_matplotlib_figure(fig, filename, output_dir)


def plot_cost_to_median_by_cluster(df_costs: pd.DataFrame, output_dir: str, filename: str):
    """Plots the total cost required to bring salaries to the median for each cluster."""
    df_plot, was_trimmed = _get_plot_data(df_costs, sort_by_col='total_adjustment', max_categories=30)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(data=df_plot, y='cluster_cse02', x='total_adjustment', ax=ax, palette="plasma")
    
    title = 'Cost to Bring Salaries to Median by Cluster'
    if was_trimmed:
        title += '\n(Top & Bottom 15 by Amount)'
        
    ax.set_title(title)
    ax.set_xlabel('Total Adjustment Cost ($)')
    ax.set_ylabel('Cluster (Role | Location | Experience)')
    
    save_matplotlib_figure(fig, filename, output_dir)


def plot_adjustment_deltas_distribution(df_deltas: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes the distribution of individual salary adjustment amounts."""
    s_deltas = df_deltas[df_deltas['delta'] > 0]['delta']
    
    fig, ax = plt.subplots()
    sns.histplot(s_deltas, bins=30, kde=True, ax=ax)
    
    ax.set_title('Distribution of Salary Adjustment Amounts (Bring-to-Median)')
    ax.set_xlabel('Adjustment Amount ($)')
    ax.set_ylabel('Number of Employees')
    
    save_matplotlib_figure(fig, filename, output_dir)


def plot_compression_metrics(df_compression: pd.DataFrame, output_dir: str, filename: str):
    """Creates two bar plots for key compression and progression metrics by cluster."""
    df_compression['cluster_label'] = df_compression['job_title'] + ' | ' + df_compression['location'] + ' | ' + df_compression['department']

    # Plot 1: Compression Index
    df_plot_index, _ = _get_plot_data(df_compression, 'compression_index', 20)
    fig1, ax1 = plt.subplots(figsize=(12, 9))
    sns.barplot(data=df_plot_index, y='cluster_label', x='compression_index', ax=ax1, palette="rocket")
    ax1.set_title('Salary Compression Index by Cluster (Top & Bottom 10)')
    ax1.set_xlabel('Compression Index (Salary Ratio of Top 25% vs. Bottom 25% by Experience)')
    ax1.set_ylabel('Cluster')
    ax1.axvline(1.15, color='r', linestyle='--', label='Risk Threshold (< 1.15)')
    ax1.legend()
    save_matplotlib_figure(fig1, f"{filename}_index", output_dir)

    # Plot 2: Salary Progression Slope
    df_plot_slope, _ = _get_plot_data(df_compression, 'slope_per_year', 20)
    fig2, ax2 = plt.subplots(figsize=(12, 9))
    sns.barplot(data=df_plot_slope, y='cluster_label', x='slope_per_year', ax=ax2, palette="mako")
    ax2.set_title('Salary Progression Slope by Cluster (Top & Bottom 10)')
    ax2.set_xlabel('Salary Growth per Year of Experience ($)')
    ax2.set_ylabel('Cluster')
    ax2.axvline(1500, color='r', linestyle='--', label='Risk Threshold (< $1500/yr)')
    ax2.legend()
    save_matplotlib_figure(fig2, f"{filename}_slope", output_dir)


def plot_manager_premium_heatmap(df_premium: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes the manager salary premium as a heatmap across departments and locations."""
    df_pivot = df_premium.pivot_table(index='department', columns='location', values='manager_premium_ratio')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=.5, ax=ax)
    
    ax.set_title('Manager Premium (Salary Ratio) vs. Individual Contributor')
    ax.set_xlabel('Location')
    ax.set_ylabel('Department')
    plt.xticks(rotation=45, ha='right')
    
    save_matplotlib_figure(fig, filename, output_dir)


def plot_model_residuals_by_segment(df_residuals: pd.DataFrame, output_dir: str, filename: str):
    """Creates a plot for mean model residuals for each segment type (e.g., department, location)."""
    for seg_type in df_residuals['segment'].unique():
        df_segment_data = df_residuals[df_residuals['segment'] == seg_type]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_segment_data, x='mean_residual_pct', y='segment_value', ax=ax, orient='h')
        
        ax.set_title(f'Mean Model Residual by Segment: {seg_type.title()}')
        ax.set_xlabel('Mean Residual (%)')
        ax.set_ylabel('Segment Value')
        ax.axvline(0, color='k', linestyle='--')
        
        save_matplotlib_figure(fig, f"{filename}_{seg_type}", output_dir)


def plot_salary_bands(df_bands: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes salary bands (P25-P75) and medians (P50) for top clusters."""
    df_plot, _ = _get_plot_data(df_bands, 'p50', 25)

    fig, ax = plt.subplots(figsize=(12, 10))

    # This layered approach creates a floating bar effect for the P25-P75 range.
    sns.barplot(y='cluster_cse02', x='p90', data=df_plot, color='lightgrey', ax=ax, label='P10-P90 Range')
    sns.barplot(y='cluster_cse02', x='p75', data=df_plot, color='skyblue', ax=ax, label='P25-P75 Range')
    sns.barplot(y='cluster_cse02', x='p25', data=df_plot, color='white', ax=ax, label='_nolegend_') # This erases the bar up to P25.

    # Overlay median points on each bar.
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        # The 'i' from enumerate corresponds to the bar's y-position (0, 1, 2...).
        ax.plot(row['p50'], i, 'ko', markersize=5, label='P50 (Median)' if i == 0 else "")

    ax.legend()
    ax.set_title('Salary Bands (P25-P75) and Medians (P50) by Cluster (Top 25)')
    ax.set_xlabel('Salary ($)')
    ax.set_ylabel('Cluster')
    
    save_matplotlib_figure(fig, f"{filename}_ranges", output_dir)


def plot_female_representation(df_representation: pd.DataFrame, output_dir: str, filename: str):
    """Plots female representation percentage with confidence intervals for each segment."""
    for seg_type in df_representation['segment'].unique():
        df_segment_data = df_representation[df_representation['segment'] == seg_type].sort_values('female_share', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = sns.barplot(data=df_segment_data, y='segment_value', x='female_share', color='skyblue', ax=ax)
        
        # Add error bars to represent the 95% Wilson confidence interval.
        error_bars = [
            df_segment_data['female_share'] - df_segment_data['female_share_ci_low'],
            df_segment_data['female_share_ci_high'] - df_segment_data['female_share']
        ]
        ax.errorbar(x=df_segment_data['female_share'], y=bars.get_yticks(), xerr=error_bars, fmt='none', c='black', capsize=3)
        
        ax.axvline(50, color='r', linestyle='--', label='Parity (50%)')
        ax.set_title(f'Female Representation by Segment: {seg_type.title()}')
        ax.set_xlabel('Female Share (%) with 95% Confidence Interval')
        ax.set_ylabel('Segment Value')
        ax.legend()
        
        save_matplotlib_figure(fig, f"{filename}_{seg_type}", output_dir)


def plot_unadjusted_gender_gap(df_gap: pd.DataFrame, output_dir: str, filename: str):
    """Displays the unadjusted (raw) median salary gap between genders."""
    data = {
        'Group': ['Median Male', 'Median Female'],
        'Salary': [df_gap['median_male'].iloc[0], df_gap['median_female'].iloc[0]]
    }
    df_plot = pd.DataFrame(data)
    
    fig, ax = plt.subplots()
    sns.barplot(data=df_plot, x='Group', y='Salary', ax=ax)
    
    gap_pct = df_gap['unadjusted_gap_pct'].iloc[0]
    ax.set_title(f'Unadjusted Median Salaries\nGap: {gap_pct:.2f}%')
    ax.set_ylabel('Median Salary ($)')
    ax.set_xlabel('Gender Group')
    
    save_matplotlib_figure(fig, filename, output_dir)


def plot_adjusted_gender_gap(df_gaps: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes the adjusted gender pay gap with confidence intervals across segments."""
    # Filter for segments with some statistical significance or the largest gaps for clarity.
    df_plot = df_gaps[df_gaps['p_value_fdr'] < 0.1].copy()
    if df_plot.empty:
        df_plot, _ = _get_plot_data(df_gaps, 'adj_gap_pct', 30)

    df_plot['segment_full_label'] = df_plot['segment'] + ' | ' + df_plot['segment_value'].astype(str)
    df_plot = df_plot.sort_values('adj_gap_pct', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['red' if x > 0 else 'blue' for x in df_plot['adj_gap_pct']]
    sns.barplot(data=df_plot, y='segment_full_label', x='adj_gap_pct', palette=colors, ax=ax)
    
    # Add error bars for the confidence interval of the gap.
    error_bars = [
        df_plot['adj_gap_pct'] - df_plot['ci_low_pct'],
        df_plot['ci_high_pct'] - df_plot['adj_gap_pct']
    ]
    ax.errorbar(x=df_plot['adj_gap_pct'], y=ax.get_yticks(), xerr=error_bars, fmt='none', c='black', capsize=3)
    
    ax.axvline(0, color='k', linestyle='--')
    ax.set_title('Adjusted Gender Pay Gap by Segment (with 95% CI)')
    ax.set_xlabel('Adjusted Pay Gap (%), negative values favor women')
    ax.set_ylabel('Segment')
    
    save_matplotlib_figure(fig, filename, output_dir)


def generate_visualizations(base_filename: str, df_current_report: pd.DataFrame, dfs_all_reports: dict, output_charts_dir: str, logger):
    """
    Acts as a router, calling the appropriate plotting function for the given report.
    
    This version receives the complete dictionary of all reports (`dfs_all_reports`),
    making it possible to create complex visualizations that combine multiple data sources
    in the future, even if none are implemented yet.
    """
    PLOT_ROUTER = {
        'bud01_summary': plot_budget_scenarios_summary,
        'bud01_bring_to_median_cost_by_cluster': plot_cost_to_median_by_cluster,
        'bud01_bring_to_median_employee_deltas': plot_adjustment_deltas_distribution,
        'cpr01_compression_by_cluster': plot_compression_metrics,
        'cpr01_compression_flags': plot_compression_metrics,
        'cpr02_manager_premium_by_dep_loc': plot_manager_premium_heatmap,
        'cse01_residuals_by_segment': plot_model_residuals_by_segment,
        'cse02_bands_by_cluster': plot_salary_bands,
        'cse02_bands_flags': plot_salary_bands,
        'dei01_female_representation': plot_female_representation,
        'peq01_unadjusted_median_gap': plot_unadjusted_gender_gap,
        'peq01_segment_adjusted_gaps_fdr': plot_adjusted_gender_gap,
    }

    if base_filename in PLOT_ROUTER:
        try:
            plot_function = PLOT_ROUTER[base_filename]
            # Even though we receive all reports, we pass only the current one
            # to the simple plotting functions.
            plot_function(df_current_report.copy(), output_charts_dir, base_filename)
            #logger.success(f"Successfully generated chart for '{base_filename}'")
        except Exception as e:
            logger.warning(f"Failed to generate chart for '{base_filename}'. Error: {e}")