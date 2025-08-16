# -*- coding: utf-8 -*-
"""
Utility functions for data input/output operations.
"""
import os
import pandas as pd

def export_dataframe(df_to_export, base_filename, output_csv_dir, output_excel_dir, column_map):
    """Saves a DataFrame to both CSV and Excel formats with user-friendly column names."""
    df_export_copy = df_to_export.copy()
    df_export_copy = df_export_copy.rename(columns=column_map)
    
    os.makedirs(output_csv_dir, exist_ok=True)
    os.makedirs(output_excel_dir, exist_ok=True)

    csv_path = os.path.join(output_csv_dir, f"{base_filename}.csv")
    excel_path = os.path.join(output_excel_dir, f"{base_filename}.xlsx")
    
    df_export_copy.to_csv(csv_path, index=False, encoding='utf-8')
    
    # Saving to Excel requires 'openpyxl' or 'xlsxwriter' package
    try:
        df_export_copy.to_excel(excel_path, index=False)
    except Exception as e:
        print(f"[WARNING] Could not save Excel file for '{base_filename}'. Make sure 'openpyxl' is installed. Error: {e}")