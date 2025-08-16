# -*- coding: utf-8 -*-

"""
Airlines flights analysis script.
Answers 9 business questions over the Indian airlines flights dataset.

This script follows the same framework patterns as the provided example:
- Uses utils.data_io.export_dataframe for CSV/Excel exports.
- Uses utils.plotting.save_matplotlib_figure and export_plotly_figure for charts.
- Uses utils.plotting.create_barplot_with_optional_hue for Matplotlib barplots.

Important data hygiene notes (based on provided dataset profile):
- No missing values and no duplicate rows were detected. We therefore do NOT drop any rows
  except removing a redundant 'index' column if present (does not discard domain data).
- We only standardize column names and trim whitespaces in string columns.
- Numeric conversions are validated but expected to succeed based on the profile.
"""

import os
import time
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from loguru import logger

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.traceback import install as rich_traceback_install
from rich.theme import Theme

from utils.data_io import export_dataframe

from visualizers.airlines_flights_visualizer import generate_visualizations


# --------------------------------------------------------------------------------------
# Pretty console setup (Rich + Loguru)
# --------------------------------------------------------------------------------------
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

logger.remove()
logger.add(
    console.print,
    level="INFO",
    colorize=True,
    format="<dim>{time:YYYY-MM-DD HH:mm:ss}</dim> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
)


def _human_readable_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def _show_dataset_snapshot(df: pd.DataFrame, title: str = "Dataset overview"):
    mem = df.memory_usage(deep=True).sum()
    table = Table(title=title, show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Metric", style="muted")
    table.add_column("Value", style="bold")
    table.add_row("Rows", f"{df.shape[0]:,}")
    table.add_row("Columns", f"{df.shape[1]:,}")
    table.add_row("Memory", _human_readable_bytes(mem))
    console.print(table)


def _show_output_tree(output_root: str, csv_dir: str, excel_dir: str, title: str = "Saved outputs"):
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


def run_analysis(df_raw_data: pd.DataFrame, config: dict):
    """
    Main execution function for the airlines flights analysis.

    Parameters
    ----------
    df_raw_data : pd.DataFrame
        Raw flights dataset already loaded into memory by the framework.
    config : dict
        Configuration dictionary. Must contain 'output_dir'
        which will be used as project name under 'output/'.
    """

    # ==============================================================================
    # CONFIGURATION & CONSTANTS
    # ==============================================================================
    PROJECT_NAME = config['output_dir']  # e.g., 'airlines_flights'

    # Pretty banner
    console.print(
        Panel.fit(
            f"[bold bright_white]Airlines flights analysis[/]\n[muted]Project:[/] [bold cyan]{PROJECT_NAME}[/]",
            title="Airline Analytics",
            border_style="bright_magenta",
            padding=(1, 4),
        )
    )

    # Plotting style
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    warnings.filterwarnings("ignore", category=FutureWarning)

    # Output directories
    OUTPUT_BASE_DIR = 'output'
    OUTPUT_CSV_DIR = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME, 'csv_reports')
    OUTPUT_EXCEL_DIR = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME, 'excel_reports')
    OUTPUT_CHARTS_DIR = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME, 'charts')
    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_EXCEL_DIR, exist_ok=True)

    # Column rename mapping for exports (human-friendly labels)
    EXPORT_COLUMN_NAMES = {
        # dataset columns
        'airline': 'Airline',
        'flight': 'Flight Code',
        'source_city': 'Source City',
        'departure_time': 'Departure Time',
        'stops': 'Stops',
        'arrival_time': 'Arrival Time',
        'destination_city': 'Destination City',
        'class': 'Class',
        'duration': 'Duration (hours)',
        'days_left': 'Days Left',
        'price': 'Price',
        'route': 'Route',
        # common metrics
        'frequency': 'Frequency',
        'count': 'Count',
        'share': 'Share',
        'n_flights': 'Flight Count',
        'avg_price': 'Average Price',
        'mean_price': 'Average Price',
        'median_price': 'Median Price',
        'std_price': 'Price StdDev',
        'min_price': 'Min Price',
        'q25_price': '25th Percentile Price',
        'q75_price': '75th Percentile Price',
        'max_price': 'Max Price',
        'booking_window': 'Booking Window',
        'last_minute': 'Booking Window',
        'days_left_bucket': 'Days Left Bucket'
    }  
    
    # Local helpers for export and save
    def _export_df(df_to_export, base_filename):
        export_dataframe(df_to_export, base_filename, OUTPUT_CSV_DIR, OUTPUT_EXCEL_DIR, EXPORT_COLUMN_NAMES)
        logger.debug(f"DataFrame exported as: {base_filename}.csv / {base_filename}.xlsx")
        # logger.debug(f"Exported DataFrame -> base='{base_filename}'")        
        calculated_dfs[base_filename] = df_to_export
        
    # Categorical orders
    TIME_OF_DAY_ORDER = ['Early_Morning', 'Morning', 'Afternoon', 'Evening', 'Night', 'Late_Night']
    STOPS_ORDER = ['zero', 'one', 'two_or_more']
    CLASS_ORDER = ['Economy', 'Business']

    # Utility for stats over price grouped by one or more columns
    def _price_stats(df, group_cols):
        def _agg(g):
            return pd.Series({
                'count': int(g.shape[0]),
                'mean_price': g['price'].mean(),
                'median_price': g['price'].median(),
                'std_price': g['price'].std(),
                'min_price': g['price'].min(),
                'q25_price': g['price'].quantile(0.25),
                'q75_price': g['price'].quantile(0.75),
                'max_price': g['price'].max()
            })
        res = df.groupby(group_cols, observed=True).apply(_agg).reset_index()
        # Round numeric stats for readability
        for c in ['mean_price', 'median_price', 'std_price', 'min_price', 'q25_price', 'q75_price', 'max_price']:
            if c in res.columns:
                res[c] = res[c].round(2)
        return res
    
    calculated_dfs = {}

    # ==============================================================================
    # PHASE 1: DATA PREPARATION
    # ==============================================================================
    console.print(Rule(title="[phase]Phase 1: Data Preparation[/phase]", style="bright_cyan"))
    t_phase1 = time.perf_counter()

    with Progress(
        SpinnerColumn(spinner_name="simpleDots", style="yellow"),
        TextColumn("[progress.description]{task.description}", style="bright_magenta"),
        BarColumn(bar_width=None, style="blue", complete_style="cyan", finished_style="green"),
        TextColumn("{task.completed}/{task.total} • "),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as q_progress:
        q_task = q_progress.add_task("[bold magenta]Business Questions[/]", total=9)

        # Standardize column names
        df = df_raw_data.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        q_progress.advance(q_task)

        # Validate required columns presence
        required_cols = [
            'airline', 'flight', 'source_city', 'departure_time', 'stops',
            'arrival_time', 'destination_city', 'class', 'duration', 'days_left', 'price'
        ]
        missing_required = [c for c in required_cols if c not in df.columns]
        if missing_required:
            console.print(Panel.fit(
                f"Dataset is missing required columns: {missing_required}",
                title="Validation Error",
                border_style="red"
            ))
            raise ValueError(f"Dataset is missing required columns: {missing_required}")
        q_progress.advance(q_task)

        # Remove redundant 'index' column if it exists (does not remove domain rows)
        if 'index' in df.columns:
            df = df.drop(columns=['index'])
        q_progress.advance(q_task)

        # Do NOT drop duplicates arbitrarily; profile indicates 0 duplicate rows.
        # Standardize whitespace in string columns
        string_columns = df.select_dtypes(include=['object']).columns.tolist()
        for col in string_columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
        q_progress.advance(q_task)

        # Numeric type conversions (validated but non-destructive; dataset has valid types)
        df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
        df['days_left'] = pd.to_numeric(df['days_left'], downcast='integer', errors='coerce')
        df['price'] = pd.to_numeric(df['price'], downcast='integer', errors='coerce')

        if df[['duration', 'days_left', 'price']].isna().any().any():
            console.print(Panel.fit(
                "Columns 'duration', 'days_left' or 'price' contain invalid (non-numeric) values after conversion.",
                title="Validation Error",
                border_style="red"
            ))
            raise ValueError("Columns 'duration', 'days_left' or 'price' contain invalid (non-numeric) values after conversion.")
        q_progress.advance(q_task)

        # Categorical typing with logical orders where applicable
        if set(TIME_OF_DAY_ORDER).issuperset(set(df['departure_time'].unique())):
            df['departure_time'] = pd.Categorical(df['departure_time'], categories=TIME_OF_DAY_ORDER, ordered=True)
        else:
            df['departure_time'] = df['departure_time'].astype('category')

        if set(TIME_OF_DAY_ORDER).issuperset(set(df['arrival_time'].unique())):
            df['arrival_time'] = pd.Categorical(df['arrival_time'], categories=TIME_OF_DAY_ORDER, ordered=True)
        else:
            df['arrival_time'] = df['arrival_time'].astype('category')

        if set(STOPS_ORDER).issuperset(set(df['stops'].unique())):
            df['stops'] = pd.Categorical(df['stops'], categories=STOPS_ORDER, ordered=True)
        else:
            df['stops'] = df['stops'].astype('category')

        if set(CLASS_ORDER).issuperset(set(df['class'].unique())):
            df['class'] = pd.Categorical(df['class'], categories=CLASS_ORDER, ordered=True)
        else:
            df['class'] = df['class'].astype('category')
        q_progress.advance(q_task)

        # Convert other categoricals for memory/performance (non-destructive)
        for col in ['airline', 'flight', 'source_city', 'destination_city']:
            df[col] = df[col].astype('category')
        q_progress.advance(q_task)

        # Route helper (for Q6)
        df['route'] = df['source_city'].astype(str) + ' \u2192 ' + df['destination_city'].astype(str)
        q_progress.advance(q_task)

        # Final clean dataframe
        df_clean = df[['airline', 'flight', 'source_city', 'departure_time', 'stops',
                       'arrival_time', 'destination_city', 'class', 'duration',
                       'days_left', 'price', 'route']].copy()
        q_progress.advance(q_task)

    logger.success("Master DataFrame 'df_clean' is ready.")
    _show_dataset_snapshot(df_clean, title="Master DataFrame (df_clean)")

    console.print(
        Panel.fit(
            f"All analysis results will be saved to: [bold cyan]{os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME)}[/]",
            title="Output Directory",
            border_style="bright_blue",
        )
    )

    # Save master clean dataset (optional, useful for auditing)
    _export_df(df_clean, 'master_clean_dataset')

    # ==============================================================================
    # PHASE 2: ANSWERING BUSINESS QUESTIONS
    # ==============================================================================
    console.print(Rule(title="[phase]Phase 2: Answering Business Questions[/phase]", style="bright_cyan"))
    t_phase2 = time.perf_counter()

    timings = {}
    total_questions = 9
    
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
               

        # Q1. What are the airlines in the dataset, accompanied by their frequencies?
        console.print(Panel.fit("[question]-> Q1: Airlines frequencies[/question]", border_style="cyan"))
        t_q = time.perf_counter()
        df_q1_airlines_freq = (
            df_clean.groupby('airline', observed=True)
                    .size()
                    .sort_values(ascending=False)
                    .reset_index(name='frequency')
        )
        df_q1_airlines_freq['share'] = (df_q1_airlines_freq['frequency'] / len(df_clean)).round(4)
        _export_df(df_q1_airlines_freq, 'q1_airlines_frequency')

        timings['Q1'] = time.perf_counter() - t_q
        
        console.print(Rule(style="bright_black"))
        q_progress.advance(q_task)


        # Q2. Show Bar Graphs representing the Departure Time & Arrival Time.
        console.print(Panel.fit("[question]-> Q2: Departure and Arrival time distributions[/question]", border_style="cyan"))
        t_q = time.perf_counter()
        df_q2_departure_counts = (
            df_clean.groupby('departure_time', observed=True)
                    .size()
                    .reset_index(name='count')
                    .sort_values('departure_time')
        )
        df_q2_arrival_counts = (
            df_clean.groupby('arrival_time', observed=True)
                    .size()
                    .reset_index(name='count')
                    .sort_values('arrival_time')
        )
        _export_df(df_q2_departure_counts, 'q2_departure_time_counts')
        _export_df(df_q2_arrival_counts, 'q2_arrival_time_counts')

        timings['Q2'] = time.perf_counter() - t_q
        
        console.print(Rule(style="bright_black"))
        q_progress.advance(q_task)

        # Q3. Show Bar Graphs representing the Source City & Destination City.
        console.print(Panel.fit("[question]-> Q3: Source/Destination city distributions[/question]", border_style="cyan"))
        t_q = time.perf_counter()
        df_q3_source_counts = (
            df_clean.groupby('source_city', observed=True)
                    .size().reset_index(name='count')
                    .sort_values('count', ascending=False)
        )
        df_q3_destination_counts = (
            df_clean.groupby('destination_city', observed=True)
                    .size().reset_index(name='count')
                    .sort_values('count', ascending=False)
        )
        _export_df(df_q3_source_counts, 'q3_source_city_counts')
        _export_df(df_q3_destination_counts, 'q3_destination_city_counts')

        timings['Q3'] = time.perf_counter() - t_q
        
        console.print(Rule(style="bright_black"))

        # Q4. Does price vary with airlines?
        console.print(Panel.fit("[question]-> Q4: Price vs Airline[/question]", border_style="cyan"))
        t_q = time.perf_counter()
        df_q4_price_by_airline = _price_stats(df_clean, ['airline']).sort_values('mean_price', ascending=False)
        _export_df(df_q4_price_by_airline, 'q4_price_by_airline_stats')

        timings['Q4'] = time.perf_counter() - t_q
        
        console.print(Rule(style="bright_black"))
        q_progress.advance(q_task)

        # Q5. Does ticket price change based on the departure time and arrival time?
        console.print(Panel.fit("[question]-> Q5: Price vs Departure/Arrival time[/question]", border_style="cyan"))
        t_q = time.perf_counter()
        df_q5_price_by_departure = _price_stats(df_clean, ['departure_time']).sort_values('departure_time')
        df_q5_price_by_arrival = _price_stats(df_clean, ['arrival_time']).sort_values('arrival_time')
        _export_df(df_q5_price_by_departure, 'q5_price_by_departure_time_stats')
        _export_df(df_q5_price_by_arrival, 'q5_price_by_arrival_time_stats')

        timings['Q5'] = time.perf_counter() - t_q
        
        console.print(Rule(style="bright_black"))
        q_progress.advance(q_task)

        # Q6. How the price changes with change in Source and Destination?
        console.print(Panel.fit("[question]-> Q6: Price vs Source/Destination (cities and routes)[/question]", border_style="cyan"))
        t_q = time.perf_counter()
        df_q6_price_by_source = _price_stats(df_clean, ['source_city']).sort_values('mean_price', ascending=False)
        df_q6_price_by_destination = _price_stats(df_clean, ['destination_city']).sort_values('mean_price', ascending=False)
        _export_df(df_q6_price_by_source, 'q6_price_by_source_city')
        _export_df(df_q6_price_by_destination, 'q6_price_by_destination_city')

        # Route-level stats and heatmap matrix
        df_q6_price_by_route = _price_stats(df_clean, ['source_city', 'destination_city']).sort_values('mean_price', ascending=False)
        df_q6_price_by_route['route'] = df_q6_price_by_route['source_city'].astype(str) + ' \u2192 ' + df_q6_price_by_route['destination_city'].astype(str)
        _export_df(df_q6_price_by_route, 'q6_price_by_route_stats')

        # Heatmap matrix (mean price)
        route_matrix = df_clean.groupby(['source_city', 'destination_city'], observed=True)['price'].mean().unstack().round(2)
        _export_df(route_matrix.reset_index(), 'q6_price_heatmap_matrix')

        timings['Q6'] = time.perf_counter() - t_q
        
        console.print(Rule(style="bright_black"))
        q_progress.advance(q_task)

        # Q7. How is the price affected when tickets are bought in just 1 or 2 days before departure?
        console.print(Panel.fit("[question]-> Q7: Price effect for last-minute bookings (1-2 days)[/question]", border_style="cyan"))
        t_q = time.perf_counter()
        df_q7 = df_clean.copy()
        df_q7['last_minute'] = np.where(df_q7['days_left'].isin([1, 2]), '1-2 days', '3+ days')
        df_q7_last_minute_stats = _price_stats(df_q7, ['last_minute']).sort_values('last_minute')
        _export_df(df_q7_last_minute_stats, 'q7_last_minute_vs_rest_stats')

        # Bar chart: Average price by last-minute flag + class breakdown
        df_q7_lm_class = (
            df_q7.groupby(['last_minute', 'class'], observed=True)['price']
                 .mean().round(2).reset_index(name='mean_price')
        )
        _export_df(df_q7_lm_class, 'q7_last_minute_vs_rest_by_class_mean_price')

        # Trend by exact days_left
        df_q7_days_left_trend = df_clean.groupby('days_left', observed=True)['price'].mean().round(2).reset_index(name='mean_price')
        _export_df(df_q7_days_left_trend, 'q7_average_price_by_days_left')

        timings['Q7'] = time.perf_counter() - t_q
        
        console.print(Rule(style="bright_black"))
        q_progress.advance(q_task)

        # Q8. How does the ticket price vary between Economy and Business class?
        console.print(Panel.fit("[question]-> Q8: Price vs Class[/question]", border_style="cyan"))
        t_q = time.perf_counter()
        df_q8_price_by_class = _price_stats(df_clean, ['class']).sort_values('class')
        _export_df(df_q8_price_by_class, 'q8_price_by_class_stats')

        timings['Q8'] = time.perf_counter() - t_q
        
        console.print(Rule(style="bright_black"))
        q_progress.advance(q_task)

        # Q9. Average Price of Vistara from Delhi to Hyderabad in Business Class
        console.print(Panel.fit("[question]-> Q9: Vistara Delhi → Hyderabad (Business) average price[/question]", border_style="cyan"))
        t_q = time.perf_counter()
        mask_q9 = (
            (df_clean['airline'] == 'Vistara') &
            (df_clean['source_city'] == 'Delhi') &
            (df_clean['destination_city'] == 'Hyderabad') &
            (df_clean['class'] == 'Business')
        )
        df_q9_subset = df_clean.loc[mask_q9].copy()

        if df_q9_subset.empty:
            # Export an informative empty result
            df_q9_result = pd.DataFrame([{
                'airline': 'Vistara',
                'source_city': 'Delhi',
                'destination_city': 'Hyderabad',
                'class': 'Business',
                'n_flights': 0,
                'avg_price': np.nan,
                'median_price': np.nan,
                'min_price': np.nan,
                'max_price': np.nan
            }])
        else:
            df_q9_result = pd.DataFrame([{
                'airline': 'Vistara',
                'source_city': 'Delhi',
                'destination_city': 'Hyderabad',
                'class': 'Business',
                'n_flights': int(df_q9_subset.shape[0]),
                'avg_price': round(df_q9_subset['price'].mean(), 2),
                'median_price': round(df_q9_subset['price'].median(), 2),
                'min_price': int(df_q9_subset['price'].min()),
                'max_price': int(df_q9_subset['price'].max())
            }])

        _export_df(df_q9_result, 'q9_vistara_delhi_hyderabad_business_avg_price')

        # Chart: single-bar representation of the computed average (if available)
        if not df_q9_subset.empty:
            # Optional: price vs. days_left for this subset
            df_q9_trend = df_q9_subset.groupby('days_left', observed=True)['price'].mean().round(2).reset_index(name='mean_price')
            _export_df(df_q9_trend, 'q9_vistara_dh_price_vs_days_left_trend')


        timings['Q9'] = time.perf_counter() - t_q
        
        console.print(Rule(style="bright_black"))
        q_progress.advance(q_task)   
        q_progress.advance(q_task)     


        # ==============================================================================
        # PHASE 3: VISUALIZATION
        # ==============================================================================
        console.print(Rule(title="[phase]Phase 3: Generating Visualizations[/phase]", style="bright_cyan"))    

        # We also need a sample of the full dataset for some plots
        sample_n = min(50000, len(df_clean))
        calculated_dfs['master_clean_dataset_sample'] = df_clean.sample(n=sample_n, random_state=42) if sample_n < len(df_clean) else df_clean

        # Loop through all generated reports and create charts
        for filename in list(calculated_dfs.keys()): # Use list() to avoid issues with dict size changes
           generate_visualizations(filename, calculated_dfs, OUTPUT_CHARTS_DIR, logger)

        logger.success("All visualizations have been generated.")

        # ==============================================================================
        # COMPLETION
        # ==============================================================================
        total_phase1 = time.perf_counter() - t_phase1
        total_phase2 = time.perf_counter() - t_phase2

        console.print(Rule(style="bright_cyan"))
        logger.success("Analysis complete. All dataframes (CSV/Excel) and charts have been saved.")

        # Timings table
        tt = Table(title="Execution timings", header_style="bold", box=None)
        tt.add_column("Stage/Question", style="muted")
        tt.add_column("Duration (s)", justify="right", style="bold")
        tt.add_row("Phase 1: Data Preparation", f"{total_phase1:.2f}")
        for k in [f"Q{i}" for i in range(1, 10)]:
            if k in timings:
                tt.add_row(k, f"{timings[k]:.2f}")
        tt.add_row("Phase 2: Business Questions (total)", f"{total_phase2:.2f}")
        console.print(tt)

        out_dir = os.path.join(OUTPUT_BASE_DIR, PROJECT_NAME)
        _show_output_tree(out_dir, OUTPUT_CSV_DIR, OUTPUT_EXCEL_DIR, title="Artifacts Overview")

        console.print(
            Panel.fit(
                f"Output directory: [bold cyan]{out_dir}[/]",
                title="Completed",
                border_style="bold green",
            )
        )