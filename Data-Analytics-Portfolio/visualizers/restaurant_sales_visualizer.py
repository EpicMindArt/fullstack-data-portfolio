# -*- coding: utf-8 -*-
"""
Generates and saves visualizations for the Restaurant Sales analysis.

This module encapsulates all plotting logic related to restaurant sales data.
It provides functions to visualize payment preferences, product performance,
revenue trends, and other key business metrics. It is designed to be
interchangeable with other visualizers in the project framework.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from utils.plotting import (
    save_matplotlib_figure,
    export_plotly_figure,
    format_matplotlib_date_axis,
    create_barplot_with_optional_hue,
)

# --- Global Plotting Style Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12


def plot_payment_method_by_orders(df_payment_methods: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes the preferred payment methods based on the number of orders."""
    fig, ax = plt.subplots(figsize=(10, 6))
    create_barplot_with_optional_hue(
        ax, data=df_payment_methods, x='payment_method', y='order_count'
    )
    ax.set_title('Preferred Payment Method by Order Count')
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Number of Orders')
    save_matplotlib_figure(fig, filename, output_dir)


def plot_payment_method_by_revenue(df_payment_methods: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes the most profitable payment methods based on total revenue."""
    fig, ax = plt.subplots(figsize=(10, 6))
    create_barplot_with_optional_hue(
        ax, data=df_payment_methods, x='payment_method', y='revenue', color_when_no_hue='tab:green'
    )
    ax.set_title('Payment Method by Total Revenue')
    ax.set_xlabel('Payment Method')
    ax.set_ylabel('Total Revenue ($)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'${val:,.0f}'))
    save_matplotlib_figure(fig, filename, output_dir)


def plot_top_products_comparison(df_top_revenue: pd.DataFrame, df_top_quantity: pd.DataFrame, output_dir: str, filename: str):
    """Creates a side-by-side comparison of top products by revenue and quantity sold."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Top Products by Revenue
    sns.barplot(data=df_top_revenue.head(10), x='revenue', y='product', ax=axes[0], color='tab:purple', orient='h')
    axes[0].set_title('Top 10 Products by Revenue')
    axes[0].set_xlabel('Total Revenue ($)')
    axes[0].set_ylabel('Product')

    # Plot 2: Top Products by Quantity
    sns.barplot(data=df_top_quantity.head(10), x='quantity', y='product', ax=axes[1], color='tab:orange', orient='h')
    axes[1].set_title('Top 10 Products by Quantity Sold')
    axes[1].set_xlabel('Total Quantity Sold')
    axes[1].set_ylabel('')  # Avoid redundant y-label

    fig.suptitle('Product Performance: Revenue vs. Quantity', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_matplotlib_figure(fig, filename, output_dir)


def plot_revenue_by_city_and_manager(df_city_revenue: pd.DataFrame, df_manager_revenue: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes revenue performance by city and by manager in a single figure."""
    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    
    # Plot 1: Revenue by City
    sns.barplot(data=df_city_revenue, x='revenue', y='city', ax=axes[0], color='tab:green', orient='h')
    axes[0].set_title('Revenue by City')
    axes[0].set_xlabel('Total Revenue ($)')
    axes[0].set_ylabel('City')

    # Plot 2: Revenue by Manager
    sns.barplot(data=df_manager_revenue, x='revenue', y='manager', ax=axes[1], color='tab:red', orient='h')
    axes[1].set_title('Revenue by Manager')
    axes[1].set_xlabel('Total Revenue ($)')
    axes[1].set_ylabel('Manager')

    plt.tight_layout()
    save_matplotlib_figure(fig, filename, output_dir)


def plot_daily_revenue_trend(df_daily_revenue: pd.DataFrame, output_dir: str, filename: str):
    """Creates two plots for the daily revenue trend: one in Matplotlib and one in Plotly."""
    # --- Matplotlib Version ---
    fig_mpl, ax_mpl = plt.subplots(figsize=(12, 6))
    ax_mpl.plot(df_daily_revenue['date'], df_daily_revenue['revenue'], marker='o', linestyle='-', markersize=4)
    ax_mpl.set_title('Daily Revenue Trend (Matplotlib)')
    ax_mpl.set_xlabel('Date')
    ax_mpl.set_ylabel('Revenue ($)')
    format_matplotlib_date_axis(ax_mpl, fmt='%Y-%m-%d', rotation=45)
    save_matplotlib_figure(fig_mpl, f"{filename}_matplotlib", output_dir)

    # --- Plotly Version ---
    df_plotly = df_daily_revenue.copy()
    df_plotly['date_str'] = df_plotly['date'].dt.strftime('%Y-%m-%d') # Use string for robust display
    fig_plotly = px.line(
        df_plotly, x='date_str', y='revenue',
        title='Daily Revenue Trend (Plotly)',
        labels={'date_str': 'Date', 'revenue': 'Revenue ($)'}
    )
    fig_plotly.update_layout(yaxis_tickprefix='$', yaxis_tickformat=',.0f', title_x=0.5)
    export_plotly_figure(fig_plotly, f"{filename}_plotly", output_dir)


def plot_long_term_revenue_trend(df_trend_data: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes the long-term revenue trend using a moving average and an OLS trendline."""
    # --- Matplotlib Version with 7-Day Moving Average ---
    fig_mpl, ax_mpl = plt.subplots(figsize=(12, 6))
    ax_mpl.scatter(df_trend_data['date'], df_trend_data['revenue'], color='tab:blue', label='Daily Revenue', alpha=0.6, s=15)
    ax_mpl.plot(df_trend_data['date'], df_trend_data['revenue'].rolling(7, min_periods=1).mean(), color='tab:red', label='7-Day Moving Average')
    ax_mpl.set_title('Revenue Trend with 7-Day Moving Average (Matplotlib)')
    ax_mpl.set_xlabel('Date')
    ax_mpl.set_ylabel('Revenue ($)')
    ax_mpl.legend()
    format_matplotlib_date_axis(ax_mpl, fmt='%Y-%m-%d', rotation=45)
    save_matplotlib_figure(fig_mpl, f"{filename}_matplotlib", output_dir)

    # --- Plotly Version with OLS Trendline ---
    # OLS requires a numeric x-axis, so we create one and then replace the tick labels.
    df_plotly = df_trend_data.copy()
    df_plotly['date_numeric'] = np.arange(len(df_plotly))
    df_plotly['date_str'] = df_plotly['date'].dt.strftime('%Y-%m-%d')
    
    fig_plotly = px.scatter(
        df_plotly, x='date_numeric', y='revenue', trendline="ols",
        title='Revenue Trend with OLS Trendline (Plotly)',
        labels={'date_numeric': 'Date'},
        hover_name='date_str', hover_data={'date_numeric': False, 'revenue': ':.2f'}
    )
    
    # Replace numeric tick labels with readable date strings.
    tick_spacing = max(1, len(df_plotly) // 15) # Show about 15 ticks
    tick_vals = df_plotly['date_numeric'][::tick_spacing]
    tick_texts = df_plotly['date_str'][::tick_spacing]
    fig_plotly.update_xaxes(tickvals=tick_vals, ticktext=tick_texts, tickangle=-45)
    
    fig_plotly.update_layout(yaxis_title='Revenue ($)', yaxis_tickprefix='$', yaxis_tickformat=',.0f', title_x=0.5)
    export_plotly_figure(fig_plotly, f"{filename}_plotly", output_dir)


def plot_product_averages(df_averages: pd.DataFrame, output_dir: str, filename: str):
    """Creates a grouped bar chart for average quantity and revenue per product."""
    df_melted = df_averages.melt(id_vars='product', var_name='metric', value_name='value')
    df_melted['metric'] = df_melted['metric'].map({
        'average_quantity': 'Average Quantity Sold',
        'average_revenue': 'Average Revenue ($)'
    })

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=df_melted, x='product', y='value', hue='metric', palette='tab10', ax=ax)
    
    ax.set_title('Average Sale Quantity and Revenue per Product')
    ax.set_xlabel('Product')
    ax.set_ylabel('Average Value')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Metric')
    
    plt.tight_layout()
    save_matplotlib_figure(fig, filename, output_dir)


def plot_revenue_by_weekday(df_weekday_revenue: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes total revenue for each day of the week."""
    # Ensure days are sorted correctly, not alphabetically.
    days_of_week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_weekday_revenue['day_of_week'] = pd.Categorical(df_weekday_revenue['day_of_week'], categories=days_of_week_order, ordered=True)
    df_plot = df_weekday_revenue.sort_values('day_of_week')

    fig, ax = plt.subplots(figsize=(12, 7))
    create_barplot_with_optional_hue(ax, df_plot, x='day_of_week', y='revenue', color_when_no_hue='tab:cyan')
    
    ax.set_title('Total Revenue by Day of the Week')
    ax.set_xlabel('Day of the Week')
    ax.set_ylabel('Total Revenue ($)')
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: f'${val:,.0f}'))
    
    plt.tight_layout()
    save_matplotlib_figure(fig, filename, output_dir)


def plot_correlation_heatmap(df_correlation: pd.DataFrame, output_dir: str, filename: str):
    """Generates a heatmap of the correlation matrix for numeric features."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Correlation Matrix of Numeric Features')
    plt.tight_layout()
    save_matplotlib_figure(fig, filename, output_dir)


def generate_visualizations(base_filename: str, dfs: dict, output_charts_dir: str, logger):
    """
    Acts as a router, calling the appropriate plotting function based on the
    filename of the report. This version handles plots that require multiple
    DataFrames.
    """
    # Note: Some plots require multiple dataframes. The 'dfs' dictionary
    # gives access to all calculated dataframes from the main script.
    
    # Simple plots that map one filename to one function
    PLOT_ROUTER_SIMPLE = {
        'q1_payment_method_by_orders': plot_payment_method_by_orders,
        'q1_payment_method_by_revenue': plot_payment_method_by_revenue,
        'q4_daily_revenue': plot_daily_revenue_trend,
        'q9_revenue_trend_data': plot_long_term_revenue_trend,
        'q10_product_averages': plot_product_averages,
        'bonus1_revenue_by_weekday': plot_revenue_by_weekday,
        'bonus2_correlation_matrix': plot_correlation_heatmap,
    }
    
    try:
        if base_filename in PLOT_ROUTER_SIMPLE:
            plot_function = PLOT_ROUTER_SIMPLE[base_filename]
            plot_function(dfs[base_filename].copy(), output_charts_dir, base_filename)
            logger.success(f"Successfully generated chart for '{base_filename}'")
        
        # --- Special Handlers for multi-DataFrame plots ---
        # These are triggered by one of their input files to avoid duplicate plots.
        elif base_filename == 'q2_top_product_by_revenue':
            plot_top_products_comparison(
                dfs['q2_top_product_by_revenue'].copy(),
                dfs['q2_top_product_by_quantity'].copy(),
                output_charts_dir,
                'q2_top_products_comparison_chart'
            )
            logger.success("Successfully generated combined chart for 'q2_top_products'")
            
        elif base_filename == 'q3_top_revenue_by_city':
            plot_revenue_by_city_and_manager(
                dfs['q3_top_revenue_by_city'].copy(),
                dfs['q3_top_revenue_by_manager'].copy(),
                output_charts_dir,
                'q3_revenue_by_city_and_manager_chart'
            )
            logger.success("Successfully generated combined chart for 'q3_revenue_sources'")

    except Exception as e:
        logger.warning(f"Failed to generate chart for '{base_filename}'. Error: {e}")