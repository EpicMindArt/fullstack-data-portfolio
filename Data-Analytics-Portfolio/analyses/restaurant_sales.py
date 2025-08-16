# -*- coding: utf-8 -*-
"""
Complete sales analysis script, refactored for clarity, self-documentation,
and professional presentation. This script is intended as a portfolio example.
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

from utils.data_io import export_dataframe

from visualizers.restaurant_sales_visualizer import generate_visualizations

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
    format="<dim>{time:YYYY-MM-DD HH:mm:ss}</dim> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
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


def _show_output_tree(output_root: str, csv_dir: str, excel_dir: str, title: str = "Saved outputs"):
    """Show a tree preview of the output directories and up to 10 files in each."""
    tree = Tree(f"[bold]Output[/] -> {output_root}", guide_style="bright_blue")
    for sub in [csv_dir, excel_dir]:
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
# Main analysis function (business logic intact, output beautified)
# --------------------------------------------------------------------------------------

def run_analysis(df_raw_data, config):
    """
    Main execution function for the restaurant sales analysis.
    This function contains the entire pipeline from data prep to visualization.
    """    
    
    # ==============================================================================
    # CONFIGURATION & CONSTANTS (now dynamically set)
    # ==============================================================================
    PROJECT_NAME = config['output_dir']

    # Option to round quantities to the nearest whole number.
    ROUND_QUANTITY = False

    # --- Plotting Style Configuration ---
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    # Register converters for proper date display in Matplotlib plots
    register_matplotlib_converters()

    # Suppress FutureWarning for cleaner output
    warnings.filterwarnings("ignore", category=FutureWarning)

    # --- Output Directories ---
    OUTPUT_BASE_DIR = 'output'
    OUTPUT_CSV_DIR = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME, 'csv_reports')
    OUTPUT_EXCEL_DIR = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME, 'excel_reports')
    OUTPUT_CHARTS_DIR = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME, 'charts')
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_EXCEL_DIR, exist_ok=True)

    # --- Column Names for Exported Files ---
    EXPORT_COLUMN_NAMES = {
        'payment_method': 'Payment Method', 'transaction_count': 'Transaction Count',
        'product': 'Product', 'revenue': 'Revenue', 'quantity': 'Quantity Sold',
        'city': 'City', 'manager': 'Manager', 'date': 'Date', 'Metric': 'Metric',
        'Value': 'Value', 'average_quantity': 'Average Quantity Sold', 'average_revenue': 'Average Revenue',
        'day_of_week': 'Day of Week', 'price': 'Price', 'order_count': 'Order Count',
        'avg_daily_revenue': 'Average Daily Revenue', 'monthly_revenue': 'Monthly Revenue'
    }        

    # Helper functions now use the dynamically created output directories
    def _export_df(df_to_export, base_filename):        
        export_dataframe(df_to_export, base_filename, OUTPUT_CSV_DIR, OUTPUT_EXCEL_DIR, EXPORT_COLUMN_NAMES)
        # logger.debug(f"Exported DataFrame -> base='{base_filename}'")        
        calculated_dfs[base_filename] = df_to_export


    # Warm welcome panel
    console.print(
        Panel.fit(
            f"[phase]Sales Analysis[/phase]\n"
            f"[muted]Project folder:[/muted] [bold]{os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME)}[/bold]",
            border_style="bright_cyan",
            title="Initialization",
            subtitle="Ready to analyze",
        )
    )

    timings = {}
    t_start = time.perf_counter()
    
    calculated_dfs = {}

    # ==============================================================================
    # PHASE 1: DATA LOADING AND PREPARATION
    # ==============================================================================
    console.print(Rule(title="[phase]Phase 1: Data Preparation[/phase]", style="bright_cyan"))
    t_phase1 = time.perf_counter()

    # --- Data Cleaning ---
    with Progress(
        SpinnerColumn(spinner_name="simpleDots", style="bright_magenta"),
        TextColumn("[progress.description]{task.description}", style="bright_magenta"),
        BarColumn(bar_width=None, style="blue", complete_style="cyan", finished_style="green"),
        TextColumn("{task.completed}/{task.total} • "),
        TimeElapsedColumn(),
        console=console,
        transient=True,  # hide after completion
    ) as prep_progress:
        prep_task = prep_progress.add_task("[bold magenta]Preparing dataset[/]", total=8)

        # Standardize column names and remove duplicates
        # --- Data Cleaning ---
        df_raw_data.columns = df_raw_data.columns.str.strip().str.lower().str.replace(' ', '_')
        df_raw_data = df_raw_data.drop_duplicates()
        prep_progress.advance(prep_task)

        # Clean string columns whitespace
        string_columns = df_raw_data.select_dtypes(include=['object']).columns
        for col in string_columns:
            df_raw_data[col] = df_raw_data[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
        prep_progress.advance(prep_task)

        # --- Date and Numeric Type Conversion ---
        try:
            df_raw_data['date'] = pd.to_datetime(df_raw_data['date'], format='%d-%m-%Y', errors='raise')
        except Exception:
            # Show a clear warning panel and fallback to dayfirst parsing
            console.print(
                Panel.fit(
                    f"[warn]Failed to parse dates with format '%d-%m-%Y'.[/warn]\n"
                    f"[muted]Retrying with dayfirst=True[/muted]",
                    border_style="yellow",
                    title="Date parsing warning",
                )
            )
            df_raw_data['date'] = pd.to_datetime(df_raw_data['date'], dayfirst=True, errors='coerce')
        prep_progress.advance(prep_task)

        if df_raw_data['date'].isna().sum() > 0:
            console.print(
                Panel.fit(
                    "[error]Date column contains invalid values. Please fix the source data.[/error]",
                    border_style="red", title="Validation Error"
                )
            )
            raise ValueError("Date column contains invalid values. Please fix the source data.")
        prep_progress.advance(prep_task)

        df_raw_data['price'] = pd.to_numeric(df_raw_data['price'], errors='coerce')
        df_raw_data['quantity'] = pd.to_numeric(df_raw_data['quantity'], errors='coerce')
        if df_raw_data['price'].isna().any() or df_raw_data['quantity'].isna().any():
            console.print(
                Panel.fit(
                    "[error]Columns 'price' or 'quantity' contain non-numeric values.[/error]",
                    border_style="red",
                    title="Validation Error",
                )
            )
            raise ValueError("Columns 'price' or 'quantity' contain non-numeric values.")
        prep_progress.advance(prep_task)

        # --- Feature Engineering ---
        if ROUND_QUANTITY:
            df_raw_data['quantity_rounded'] = df_raw_data['quantity'].round().astype(int)
            df_raw_data['revenue'] = df_raw_data['price'] * df_raw_data['quantity_rounded']
        else:
            df_raw_data['revenue'] = df_raw_data['price'] * df_raw_data['quantity']
        prep_progress.advance(prep_task)

        for col in ['product', 'purchase_type', 'payment_method', 'manager', 'city']:
            if col in df_raw_data.columns and df_raw_data[col].nunique() / len(df_raw_data) < 0.5:
                df_raw_data[col] = df_raw_data[col].astype('category')

        df_raw_data['month'] = df_raw_data['date'].dt.month_name()
        df_raw_data['day_of_week'] = df_raw_data['date'].dt.day_name()
        df_raw_data['year'] = df_raw_data['date'].dt.year
        prep_progress.advance(prep_task)

        final_column_order = [c for c in [
            'order_id', 'date', 'city', 'manager', 'product',
            'price', 'quantity', 'revenue', 'purchase_type',
            'payment_method', 'month', 'day_of_week', 'year'
        ] if c in df_raw_data.columns]
        df_clean = df_raw_data[final_column_order].copy()
        prep_progress.advance(prep_task)

    # Post-preparation reporting
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
    # PHASE 2: BUSINESS QUESTION ANALYSIS
    # ==============================================================================
    console.print(Rule(title="[phase]Phase 2: Answering Business Questions[/phase]", style="bright_cyan"))
    t_phase2 = time.perf_counter()

    # Prepare tqdm progress bar for ten business questions
    total_questions = 9
    q_timings = {}
    
    console.print(Rule(title="[phase]Phase: Business Questions Analysis[/phase]", style="bright_cyan"))

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

        # Q1. What is the most preferred payment method?
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> Q1: Analyzing Payment Methods[/question]", border_style="cyan"))

        # To answer this, we can measure by two different metrics:
        # 1. By count of unique orders: This is often more reliable as it's not skewed by large, infrequent purchases.
        df_q1_payment_by_orders = (
            df_clean.drop_duplicates(subset=['order_id'])
                    .groupby('payment_method', observed=True)['order_id']
                    .nunique()
                    .sort_values(ascending=False)
                    .reset_index(name='order_count')
        )
        _export_df(df_q1_payment_by_orders, 'q1_payment_method_by_orders')

        # 2. By total revenue: This shows which method brings in the most money.
        df_q1_payment_by_revenue = (
            df_clean.groupby('payment_method', observed=True)['revenue']
                    .sum()
                    .sort_values(ascending=False)
                    .reset_index()
        )
        _export_df(df_q1_payment_by_revenue, 'q1_payment_method_by_revenue')
        
        logger.success("Q1 completed.")
        
        q_progress.advance(q_task)
        console.print(Rule(style="bright_black"))
        
        q_timings["Q1"] = time.perf_counter() - q_start
        

        # Q2. What are the top-selling products by quantity and revenue?
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> Q2: Analyzing Top Products by Revenue and Quantity[/question]", border_style="cyan"))
        df_q2_top_products_by_revenue = df_clean.groupby('product', observed=True)['revenue'].sum().sort_values(ascending=False).reset_index()
        df_q2_top_products_by_quantity = df_clean.groupby('product', observed=True)['quantity'].sum().sort_values(ascending=False).reset_index()

        _export_df(df_q2_top_products_by_revenue, 'q2_top_product_by_revenue')
        _export_df(df_q2_top_products_by_quantity, 'q2_top_product_by_quantity')
        
        logger.success("Q2 completed.")
        
        q_progress.advance(q_task)
        console.print(Rule(style="bright_black"))
        
        q_timings["Q2"] = time.perf_counter() - q_start

        # Q3. Which city and manager generated the most revenue?
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> Q3: Analyzing Revenue by City and Manager[/question]", border_style="cyan"))
        df_q3_revenue_by_city = df_clean.groupby('city', observed=True)['revenue'].sum().sort_values(ascending=False).reset_index()
        df_q3_revenue_by_manager = df_clean.groupby('manager', observed=True)['revenue'].sum().sort_values(ascending=False).reset_index()

        _export_df(df_q3_revenue_by_city, 'q3_top_revenue_by_city')
        _export_df(df_q3_revenue_by_manager, 'q3_top_revenue_by_manager')
        
        logger.success("Q3 completed.")
        
        q_progress.advance(q_task)
        console.print(Rule(style="bright_black"))
        
        q_timings["Q3"] = time.perf_counter() - q_start

        # Q4. What is the daily revenue trend?
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> Q4: Analyzing Daily Revenue[/question]", border_style="cyan"))
        df_q4_daily_revenue = df_clean.groupby('date', as_index=False)['revenue'].sum()
        _export_df(df_q4_daily_revenue, 'q4_daily_revenue')
        
        logger.success("Q4 completed.")
        
        q_progress.advance(q_task)
        console.print(Rule(style="bright_black"))
        
        q_timings["Q4"] = time.perf_counter() - q_start

        # Q5–Q8. What are the key statistical metrics?
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> Q5–Q8: Calculating Key Statistical Metrics[/question]", border_style="cyan"))
        # Average Order Value (AOV) across all transactions.
        avg_revenue_per_transaction = df_clean['revenue'].mean()

        # Average daily revenue (calculated over days with sales).
        df_daily_revenue = df_clean.groupby('date')['revenue'].sum()
        avg_daily_revenue_total = df_daily_revenue.mean()

        # Average daily revenue for November and December.
        is_november = (df_clean['month'] == 'November')
        is_december = (df_clean['month'] == 'December')
        avg_daily_revenue_nov = df_clean[is_november].groupby('date')['revenue'].sum().mean() if is_november.any() else np.nan
        avg_daily_revenue_dec = df_clean[is_december].groupby('date')['revenue'].sum().mean() if is_december.any() else np.nan

        # Standard deviation and variance for revenue and quantity per transaction.
        std_dev_revenue = df_clean['revenue'].std()
        std_dev_quantity = df_clean['quantity'].std()
        variance_revenue = df_clean['revenue'].var()
        variance_quantity = df_clean['quantity'].var()

        df_q5_to_q8_summary_stats = pd.DataFrame({
            'Metric': [
                'Q5: Avg Revenue per Transaction (AOV)', 'Q5: Avg Daily Revenue (All Time)',
                'Q6: Avg Daily Revenue (November)', 'Q6: Avg Daily Revenue (December)',
                'Q7: Std. Deviation of Revenue (per transaction)', 'Q7: Std. Deviation of Quantity (per transaction)',
                'Q8: Variance of Revenue (per transaction)', 'Q8: Variance of Quantity (per transaction)'
            ],
            'Value': [
                f"${avg_revenue_per_transaction:,.2f}", f"${avg_daily_revenue_total:,.2f}",
                f"${avg_daily_revenue_nov:,.2f}", f"${avg_daily_revenue_dec:,.2f}",
                f"${std_dev_revenue:,.2f}", f"{std_dev_quantity:,.2f} units",
                f"${variance_revenue:,.2f}", f"{variance_quantity:,.2f} units"
            ]
        })
        _export_df(df_q5_to_q8_summary_stats, 'q5_to_q8_statistical_summary')
        logger.success("Q5–Q8 completed.")
        
        q_progress.advance(q_task)# counting as a single logical step in the progress
        q_progress.advance(q_task)
        q_progress.advance(q_task)
        q_progress.advance(q_task)
        console.print(Rule(style="bright_black"))
        
        q_timings["Q5–Q8"] = time.perf_counter() - q_start

        # Q9. What is the overall revenue trend?
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> Q9: Analyzing Long-Term Revenue Trend[/question]", border_style="cyan"))
        df_q9_trend_data = df_daily_revenue.reset_index()

        _export_df(df_q9_trend_data, 'q9_revenue_trend_data')
        logger.success("Q9 completed.")
        
        q_progress.advance(q_task)
        console.print(Rule(style="bright_black"))
        
        q_timings["Q9"] = time.perf_counter() - q_start

        # Q10. What is the average quantity and revenue for each product?
        q_start = time.perf_counter()
        console.print(Panel.fit("[question]-> Q10: Calculating Average Metrics per Product[/question]", border_style="cyan"))
        df_q10_product_averages = (
            df_clean.groupby('product', observed=True)
                    .agg(average_quantity=('quantity', 'mean'),
                         average_revenue=('revenue', 'mean'))
                    .reset_index()
                    .round(2)
        )
        _export_df(df_q10_product_averages, 'q10_product_averages')
        
        logger.success("Q10 completed.")      
                
        q_timings["Q10"] = time.perf_counter() - q_start

    timings["phase_2_questions"] = time.perf_counter() - t_phase2
    logger.success(f"Phase 2 completed in {timings['phase_2_questions']:.2f}s")
    
    console.print(Rule(style="bright_black"))

    # ==============================================================================
    # PHASE 3: BONUS ANALYSIS
    # ==============================================================================
    console.print(Rule(title="[phase]Phase 3: Bonus Analysis[/phase]", style="bright_cyan"))
    t_bonus = time.perf_counter()

    # Bonus 1: Revenue by Day of the Week
    console.print(Panel.fit("[question]-> Bonus 1: Analyzing Revenue by Day of the Week[/question]", border_style="cyan"))
    df_bonus1_revenue_by_weekday = df_clean.groupby('day_of_week', observed=True)['revenue'].sum().reset_index()
    days_of_week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df_bonus1_revenue_by_weekday['day_of_week'] = pd.Categorical(
        df_bonus1_revenue_by_weekday['day_of_week'], categories=days_of_week_order, ordered=True
    )
    df_bonus1_revenue_by_weekday = df_bonus1_revenue_by_weekday.sort_values('day_of_week')
    _export_df(df_bonus1_revenue_by_weekday, 'bonus1_revenue_by_weekday')

    
    logger.success("Bonus 1 completed.")

    # Bonus 2: Correlation Matrix of Numeric Features
    console.print(Panel.fit("[question]-> Bonus 2: Generating a Correlation Matrix[/question]", border_style="cyan"))
    df_bonus2_correlation_matrix = df_clean[['price', 'quantity', 'revenue']].corr()
    # Use the export names for a more presentable heatmap

    _export_df(df_bonus2_correlation_matrix, 'bonus2_correlation_matrix')
    
    logger.success("Bonus 2 completed.")

    timings["phase_3_bonus"] = time.perf_counter() - t_bonus

    # ==============================================================================
    # WRAP-UP: EXECUTION STATS AND OUTPUT OVERVIEW
    # ==============================================================================
    total_elapsed = time.perf_counter() - t_start
    console.print(Rule(style="bright_cyan"))

    # Build timings table
    table = Table(title="Execution timings", show_header=True, header_style="bold")
    table.add_column("Step", style="muted")
    table.add_column("Elapsed", style="bold")
    table.add_row("Phase 1: Data Preparation", f"{timings['phase_1_prep']:.2f}s")
    table.add_row("Phase 2: Business Questions", f"{timings['phase_2_questions']:.2f}s")
    for q, tval in q_timings.items():
        table.add_row(f"  {q}", f"{tval:.2f}s")
    table.add_row("Phase 3: Bonus", f"{timings['phase_3_bonus']:.2f}s")
    table.add_row("Total runtime", f"{total_elapsed:.2f}s")
    console.print(table)
    
    console.print(Rule(title="[phase]Phase 4: Generating Visualizations[/phase]", style="bright_cyan"))
    from visualizers.restaurant_sales_visualizer import generate_visualizations
    
    for filename, df_data in calculated_dfs.items():
        generate_visualizations(filename, calculated_dfs, OUTPUT_CHARTS_DIR, logger)
        
    logger.success("All visualizations generated.")

    # Show output directories and files
    _show_output_tree(
        output_root=os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME),
        csv_dir=OUTPUT_CSV_DIR,
        excel_dir=OUTPUT_EXCEL_DIR,
        title="Saved reports and charts",
    )

    # Final note and reminder panel
    console.print(
        Panel.fit(
            "Analysis complete. All reports and charts have been saved successfully.\n"
            "Reminder: If a date axis ever displays as numbers, review the\n"
            "'format_matplotlib_date_axis' and Plotly 'date_str' methods used in this script.",
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