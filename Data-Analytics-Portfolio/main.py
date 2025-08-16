# main.py
# -*- coding: utf-8 -*-
"""
Main entry point for the data analysis portfolio.

This script orchestrates the execution of analysis pipelines.

Usage:
    # Run a deep-dive analysis (e.g., answering business questions)
    python main.py restaurant_sales

    # Run a primary, automated EDA (generates ydata, sweetviz reports)
    python main.py restaurant_sales --primary
"""
import sys
import yaml
import pandas as pd
import importlib
from pathlib import Path
import argparse

from utils.primary_analyzer import perform_primary_analysis

sys.stdout.reconfigure(encoding='utf-8')

def main():
    """
    Orchestrates the execution of a specific analysis pipeline based on CLI arguments.
    """    
    parser = argparse.ArgumentParser(
        description="Data Analysis Portfolio Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "project_key",
        type=str,
        help="The key of the project to run (e.g., 'restaurant_sales')."
    )
    parser.add_argument(
        "-p", "--primary",
        action="store_true",
        help="If set, runs the primary automated EDA instead of the deep-dive analysis."
    )
    args = parser.parse_args()    
    
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("FATAL: config.yaml not found. Run from the project root.")
        sys.exit(1)        
    
    project_key = args.project_key
    if project_key not in config:
        print(f"FATAL: Project key '{project_key}' not found in config.yaml.")
        print(f"Available projects: {list(config.keys())}")
        sys.exit(1)
        
    project_config = config[project_key]
    
    
    if args.primary:        
        print(f"--- Initializing PRIMARY EDA for: {project_key} ---")        
        perform_primary_analysis(project_config, project_key)
        print(f"--- Primary EDA for '{project_key}' completed successfully! ---")
        
    else:        
        print(f"--- Initializing DEEP-DIVE analysis for: {project_key} ---")
        print(f"Description: {project_config['description']}")
        
        input_path = Path(project_config['input_file'])
        if not input_path.exists():
            print(f"FATAL: Input file not found at '{input_path}'. Check config.yaml.")
            sys.exit(1)
        
        try:
            df_raw = pd.read_csv(input_path)
            print(f"Successfully loaded data from: {input_path}")
        except Exception as e:
            print(f"FATAL: Failed to read the CSV file. Error: {e}")
            sys.exit(1)
        
        try:
            analysis_module = importlib.import_module(project_config['analysis_module'])
            print(f"Successfully imported module: {project_config['analysis_module']}")
        except ImportError as e:
            print(f"FATAL: Could not import analysis module '{project_config['analysis_module']}'. Error: {e}")
            sys.exit(1)
            
        try:
            analysis_module.run_analysis(df_raw, project_config)
            print(f"--- Deep-dive analysis for '{project_key}' completed successfully! ---")
        except Exception as e:
            print(f"FATAL: An error occurred during the analysis execution. Error: {e}")

if __name__ == '__main__':
    main()