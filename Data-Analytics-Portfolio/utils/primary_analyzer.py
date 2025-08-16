import sys
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import yaml
import pandas as pd
import json
import platform
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
from ydata_profiling import ProfileReport
from loguru import logger
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

# --- Configuration ---
# Use 'Agg' backend for matplotlib to prevent GUI windows from appearing on servers.
matplotlib.use('Agg')
# Ignore common UserWarnings from libraries like ydata-profiling.
warnings.filterwarnings("ignore", category=UserWarning)

# Configure Loguru for clear and informative console output.
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
)

def set_console_dimensions(width: int, height: int):
    """
    Attempts to set the console window size.
    This functionality is platform-dependent and may not work in all terminals.
    
    Args:
        width (int): The desired width of the console in columns.
        height (int): The desired height of the console in lines.
    """
    system_name = platform.system()
    try:
        if system_name == "Windows":
            os.system(f'mode con: cols={width} lines={height}')
        elif system_name in ["Linux", "Darwin"]:  # Darwin is macOS
            sys.stdout.write(f"\x1b[8;{height};{width}t")
        logger.info(f"Attempted to set console size to {width}x{height}.")
    except Exception as e:
        logger.warning(f"Could not set console size: {e}")

def display_dataframe_as_rich_table(df_to_display: pd.DataFrame, title: str):
    """
    Displays a Pandas DataFrame as a formatted table in the console using Rich.

    Args:
        df_to_display (pd.DataFrame): The DataFrame to display.
        title (str): The title for the table.
    """
    console = Console()
    table = Table(
        show_header=True,
        header_style="bold magenta",
        title=title,
        title_style="bold green"
    )

    for column in df_to_display.columns:
        table.add_column(str(column))

    for _, row in df_to_display.iterrows():
        table.add_row(*[str(item) for item in row])

    console.print(table)

def create_json_summary_report(df_source: pd.DataFrame, output_directory: str, project_key: str):
    """
    Creates and saves a lightweight JSON report with key data metrics.

    Args:
        df_source (pd.DataFrame): The source DataFrame for analysis.
        output_directory (str): The directory to save the report in.
        project_key (str): The unique identifier for the project.

    Returns:
        dict: The summary report as a Python dictionary.
    """
    logger.info("Creating enhanced lightweight JSON summary report...")

    report_summary = {
        "project_info": {
            "project_key": project_key,
            "source_file": df_source.attrs.get('name', 'N/A'),
            "report_date": pd.Timestamp.now().isoformat()
        },
        "table_summary": {
            "rows": len(df_source),
            "columns": len(df_source.columns),
            "total_cells": int(df_source.size),
            "missing_cells": int(df_source.isnull().sum().sum()),
            "duplicate_rows": int(df_source.duplicated().sum()),
        },
        "column_analysis": {}
    }

    for col_name in df_source.columns:
        column_details = {}
        column_series = df_source[col_name]
        
        column_details['dtype'] = str(column_series.dtype)
        column_details['missing_values'] = int(column_series.isnull().sum())
        column_details['missing_percentage'] = round(column_series.isnull().mean() * 100, 2)
        column_details['unique_values_count'] = column_series.nunique()

        if pd.api.types.is_numeric_dtype(column_series.dtype):
            column_details['detected_type'] = 'numeric'
            stats = column_series.describe()
            column_details['stats'] = {
                'mean': round(stats.get('mean', 0), 2),
                'std': round(stats.get('std', 0), 2),
                'min': round(stats.get('min', 0), 2),
                'q25': round(stats.get('25%', 0), 2),
                'median': round(stats.get('50%', 0), 2),
                'q75': round(stats.get('75%', 0), 2),
                'max': round(stats.get('max', 0), 2)
            }
        else:
            # Heuristic for datetime detection: attempt conversion and check success rate.
            # If over 80% of non-null values parse as dates, classify as datetime.
            temp_datetime_series = pd.to_datetime(column_series, errors='coerce')
            if column_series.notna().any() and (temp_datetime_series.notna().sum() / column_series.notna().sum() > 0.8):
                column_details['detected_type'] = 'datetime'
                column_details['date_range'] = {
                    'min_date': temp_datetime_series.min().isoformat(),
                    'max_date': temp_datetime_series.max().isoformat()
                }
            else:
                column_details['detected_type'] = 'categorical'
                str_lengths = column_series.dropna().astype(str).str.len()
                if not str_lengths.empty:
                    column_details['string_stats'] = {
                        'mean_length': round(str_lengths.mean(), 2),
                        'min_length': int(str_lengths.min()),
                        'max_length': int(str_lengths.max())
                    }
            
            top_values = column_series.value_counts(dropna=False).head(10)
            column_details['top_10_values'] = {str(k): int(v) for k, v in top_values.items()}

        report_summary["column_analysis"][col_name] = column_details
    
    report_path = os.path.join(output_directory, f"{project_key}_summary_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_summary, f, ensure_ascii=False, indent=4)
        
    logger.success(f"Enhanced JSON report saved successfully: {report_path}")
    return report_summary

def generate_ydata_profile(df_source: pd.DataFrame, output_directory: str, project_key: str):
    """
    Generates and saves a detailed HTML report using ydata-profiling.

    Args:
        df_source (pd.DataFrame): The source DataFrame for analysis.
        output_directory (str): The directory to save the report in.
        project_key (str): The unique identifier for the project.
    """
    logger.info("Generating detailed HTML report with ydata-profiling... This may take a moment.")
    
    profile = ProfileReport(df_source, title=f"Data Analysis Report for: {project_key}")
    report_path = os.path.join(output_directory, f"{project_key}_ydata_report.html")
    profile.to_file(report_path)
    
    logger.success(f"ydata-profiling report generated successfully! File saved at: {report_path}")

def perform_primary_analysis(project_config: dict, project_key: str):
    """
    Executes the full analysis pipeline for a specified project.
    
    Args:
        project_config (dict): The configuration dictionary for the project.
        project_key (str): The unique identifier for the project.
    """
    logger.info(f"Starting analysis for project: '{project_key}'")
    logger.info(f"Description: {project_config['description']}")

    input_file = project_config['input_file']
    base_output_folder = "output"
    project_specific_folder = project_config['output_dir']
    analysis_type_folder = "primary_analysis"
    output_directory = os.path.join(base_output_folder, project_specific_folder, analysis_type_folder)

    os.makedirs(output_directory, exist_ok=True)
    logger.info(f"All artifacts will be saved to: {output_directory}")

    try:
        logger.info(f"Loading data from file: {input_file}")
        df_source = pd.read_csv(input_file)
        df_source.attrs['name'] = input_file # Store filename for reporting
    except FileNotFoundError:
        logger.error(f"File not found: {input_file}. Please check the path in config.yaml.")
        return
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV file: {e}")
        return

    logger.success("Data loaded successfully.")

    rows, columns = df_source.shape
    logger.info(f"DataFrame dimensions: {rows} rows, {columns} columns.")
    
    logger.info("Displaying first 5 rows of the dataset:")
    display_dataframe_as_rich_table(df_source.head(), "First 5 Rows (df.head())")
    
    with open(os.path.join(output_directory, '01_summary.txt'), 'w', encoding='utf-8') as f:
        f.write("--- DataFrame General Information ---\n")
        df_source.info(buf=f, verbose=True, show_counts=True)
        
        f.write("\n\n--- Descriptive Statistics (All Columns) ---\n")
        
        f.write(df_source.describe(include='all').to_string())
    logger.success("Basic info and statistics saved to '01_summary_info.txt'")

    logger.info("Performing deeper data quality analysis...")
    
    missing_values = df_source.isnull().sum()
    missing_percent = (missing_values / len(df_source)) * 100
    df_missing_report = pd.DataFrame({'missing_count': missing_values, 'missing_percent': missing_percent})
    df_missing_report = df_missing_report[df_missing_report['missing_count'] > 0]
    
    if not df_missing_report.empty:
        logger.warning("Missing values found. See details below:")
        print(df_missing_report.sort_values(by='missing_percent', ascending=False))
    else:
        logger.success("No missing values found.")
        
    num_duplicates = df_source.duplicated().sum()
    if num_duplicates > 0:
        logger.warning(f"Found {num_duplicates} fully duplicated rows.")
    else:
        logger.success("No duplicate rows found.")
        
    cardinality = df_source.nunique()
    constant_columns = cardinality[cardinality == 1].index.tolist()
    if constant_columns:
        logger.warning(f"Found constant-value columns (unhelpful for analysis): {constant_columns}")

    plots_directory = os.path.join(output_directory, 'plots')
    os.makedirs(plots_directory, exist_ok=True)
    numeric_cols = df_source.select_dtypes(include=['number']).columns
    
    logger.info("Creating and saving histograms for numerical columns...")
    for col in tqdm(numeric_cols, desc="Generating Histograms"):
        plt.figure(figsize=(10, 6))
        sns.histplot(df_source[col], kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(os.path.join(plots_directory, f'hist_{col}.png'))
        plt.close()
    logger.success(f"Histograms saved to: {plots_directory}")

    if len(numeric_cols) > 1:
        logger.info("Creating correlation heatmap...")
        plt.figure(figsize=(14, 10))
        sns.heatmap(df_source[numeric_cols].corr(), annot=True, cmap='viridis', fmt='.2f')
        plt.title('Correlation Heatmap of Numerical Columns')
        plt.savefig(os.path.join(plots_directory, 'correlation_heatmap.png'))
        plt.close()
        logger.success(f"Heatmap saved to: {plots_directory}")
        
    create_json_summary_report(df_source, output_directory, project_key)

    logger.info("Generating detailed HTML report with Sweetviz... This may take some time.")
    sweetviz_report = sv.analyze(df_source)
    report_path_sv = os.path.join(output_directory, f"{project_key}_sweetviz_report.html")
    sweetviz_report.show_html(report_path_sv, open_browser=False)
    logger.success(f"Sweetviz report generated successfully! File saved at: {report_path_sv}")
            
    generate_ydata_profile(df_source, output_directory, project_key)
    
    logger.info(f"Analysis for project '{project_key}' has completed successfully!")
