# -*- coding: utf-8 -*-
"""
Generates and saves visualizations for the Airlines Flights analysis.

This module contains all plotting functions related to the Indian airlines
dataset. It handles visualizations for flight frequencies, price distributions,
route analyses, and booking behaviors, adhering to the project's
standard visualization framework.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from utils.plotting import (
    save_matplotlib_figure,
    export_plotly_figure,
    create_barplot_with_optional_hue,
)

# --- Global Plotting Style Configuration ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12


def plot_airline_frequency(df_frequency: pd.DataFrame, output_dir: str, filename: str):
    """Visualizes the frequency of flights for each airline."""
    fig, ax = plt.subplots(figsize=(12, 6))
    create_barplot_with_optional_hue(ax, data=df_frequency, x='airline', y='frequency')
    ax.set_title('Flight Frequency by Airline')
    ax.set_xlabel('Airline')
    ax.set_ylabel('Number of Flights')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    save_matplotlib_figure(fig, filename, output_dir)


def plot_time_distribution(df_counts: pd.DataFrame, time_col: str, output_dir: str, filename: str):
    """Creates a bar plot for flight counts by a given time category (departure or arrival)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    create_barplot_with_optional_hue(ax, data=df_counts, x=time_col, y='count', color_when_no_hue='tab:orange' if 'departure' in time_col else 'tab:green')
    
    title_prefix = 'Departure' if 'departure' in time_col else 'Arrival'
    ax.set_title(f'{title_prefix} Time Distribution')
    ax.set_xlabel(f'{title_prefix} Time of Day')
    ax.set_ylabel('Number of Flights')
    ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    save_matplotlib_figure(fig, filename, output_dir)


def plot_city_distribution(df_counts: pd.DataFrame, city_col: str, output_dir: str, filename: str):
    """Creates a bar plot for flight counts by city (source or destination)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    color = 'tab:purple' if 'source' in city_col else 'tab:red'
    create_barplot_with_optional_hue(ax, data=df_counts, x=city_col, y='count', color_when_no_hue=color)

    title_prefix = 'Source' if 'source' in city_col else 'Destination'
    ax.set_title(f'{title_prefix} City Distribution')
    ax.set_xlabel(f'{title_prefix} City')
    ax.set_ylabel('Number of Flights')
    ax.tick_params(axis='x', rotation=30)
    
    plt.tight_layout()
    save_matplotlib_figure(fig, filename, output_dir)


def plot_price_by_airline(df_stats: pd.DataFrame, df_full_data_sample: pd.DataFrame, output_dir: str):
    """Creates a bar chart of average prices and a boxplot of price distributions by airline."""
    # Bar chart for mean price
    fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
    create_barplot_with_optional_hue(ax_bar, data=df_stats, x='airline', y='mean_price')
    ax_bar.set_title('Average Ticket Price by Airline')
    ax_bar.set_xlabel('Airline')
    ax_bar.set_ylabel('Average Price')
    ax_bar.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    save_matplotlib_figure(fig_bar, 'q4_mean_price_by_airline_chart', output_dir)

    # Plotly boxplot for full distribution
    fig_box = px.box(
        df_full_data_sample, x='airline', y='price',
        title='Ticket Price Distribution by Airline (Sampled)',
        points='outliers'
    )
    fig_box.update_layout(title_x=0.5, xaxis_title='Airline', yaxis_title='Price')
    export_plotly_figure(fig_box, 'q4_price_by_airline_boxplot_plotly', output_dir)


def plot_price_by_time(df_departure_stats: pd.DataFrame, df_arrival_stats: pd.DataFrame, output_dir: str):
    """Visualizes average prices by departure and arrival times."""
    # Departure time plot
    fig_dep, ax_dep = plt.subplots(figsize=(10, 5))
    create_barplot_with_optional_hue(ax_dep, data=df_departure_stats, x='departure_time', y='mean_price', color_when_no_hue='tab:orange')
    ax_dep.set_title('Average Price by Departure Time')
    ax_dep.set_xlabel('Departure Time')
    ax_dep.set_ylabel('Average Price')
    ax_dep.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    save_matplotlib_figure(fig_dep, 'q5_mean_price_by_departure_time_chart', output_dir)
    
    # Arrival time plot
    fig_arr, ax_arr = plt.subplots(figsize=(10, 5))
    create_barplot_with_optional_hue(ax_arr, data=df_arrival_stats, x='arrival_time', y='mean_price', color_when_no_hue='tab:green')
    ax_arr.set_title('Average Price by Arrival Time')
    ax_arr.set_xlabel('Arrival Time')
    ax_arr.set_ylabel('Average Price')
    ax_arr.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    save_matplotlib_figure(fig_arr, 'q5_mean_price_by_arrival_time_chart', output_dir)


def plot_price_by_route_heatmap(df_route_matrix: pd.DataFrame, output_dir: str, filename: str):
    """Generates a heatmap of mean prices between source and destination cities."""
    # The matrix is already pivoted, just needs the index set for heatmap
    df_plot = df_route_matrix.set_index('source_city')
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_plot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=.5, ax=ax)
    ax.set_title('Mean Price Heatmap by Source → Destination')
    ax.set_xlabel('Destination City')
    ax.set_ylabel('Source City')
    plt.tight_layout()
    save_matplotlib_figure(fig, filename, output_dir)


def plot_last_minute_booking_effect(df_lm_class: pd.DataFrame, df_trend: pd.DataFrame, output_dir: str):
    """Visualizes the price impact of last-minute bookings."""
    # Bar chart comparing 1-2 days vs 3+ days
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df_lm_class, x='last_minute', y='mean_price', hue='class', palette='tab10', ax=ax_bar)
    ax_bar.set_title('Average Price: Last-Minute (1–2 days) vs. 3+ days, by Class')
    ax_bar.set_xlabel('Booking Window')
    ax_bar.set_ylabel('Average Price')
    plt.tight_layout()
    save_matplotlib_figure(fig_bar, 'q7_last_minute_vs_rest_chart', output_dir)

    # Line chart showing price trend by exact days left
    fig_line, ax_line = plt.subplots(figsize=(10, 5))
    ax_line.plot(df_trend['days_left'], df_trend['mean_price'], marker='o', linestyle='-')
    ax_line.set_title('Average Price vs. Days Left to Departure')
    ax_line.set_xlabel('Days Left')
    ax_line.set_ylabel('Average Price')
    ax_line.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    save_matplotlib_figure(fig_line, 'q7_price_trend_by_days_left_chart', output_dir)


def plot_price_by_class(df_stats: pd.DataFrame, df_full_data_sample: pd.DataFrame, output_dir: str):
    """Visualizes price differences between Economy and Business class."""
    # Bar chart for mean price
    fig_bar, ax_bar = plt.subplots(figsize=(8, 6))
    create_barplot_with_optional_hue(ax_bar, data=df_stats, x='class', y='mean_price')
    ax_bar.set_title('Average Price by Class')
    ax_bar.set_xlabel('Class')
    ax_bar.set_ylabel('Average Price')
    plt.tight_layout()
    save_matplotlib_figure(fig_bar, 'q8_mean_price_by_class_chart', output_dir)

    # Plotly boxplot for full distribution
    fig_box = px.box(
        df_full_data_sample, x='class', y='price',
        title='Ticket Price Distribution by Class (Sampled)',
        points='outliers'
    )
    fig_box.update_layout(title_x=0.5, xaxis_title='Class', yaxis_title='Price')
    export_plotly_figure(fig_box, 'q8_price_by_class_boxplot_plotly', output_dir)


def plot_specific_route_analysis(df_result: pd.DataFrame, df_trend: pd.DataFrame, output_dir: str):
    """Visualizes the analysis for a specific flight route (e.g., Vistara DEL-HYD)."""
    # Simple bar for the overall average price
    avg_price = df_result.loc[0, 'avg_price']
    fig_bar, ax_bar = plt.subplots(figsize=(6, 5))
    ax_bar.bar(['Vistara (DEL→HYD, Business)'], [avg_price], color='tab:blue')
    ax_bar.set_title('Average Price for Vistara DEL→HYD (Business)')
    ax_bar.set_ylabel('Average Price')
    plt.tight_layout()
    save_matplotlib_figure(fig_bar, 'q9_vistara_del_hyd_business_avg_price_chart', output_dir)

    # Line chart for price vs days_left trend
    fig_line, ax_line = plt.subplots(figsize=(8, 5))
    ax_line.plot(df_trend['days_left'], df_trend['mean_price'], marker='o', linestyle='-')
    ax_line.set_title('Price vs. Days Left for Vistara DEL→HYD (Business)')
    ax_line.set_xlabel('Days Left')
    ax_line.set_ylabel('Average Price')
    ax_line.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    save_matplotlib_figure(fig_line, 'q9_vistara_del_hyd_price_vs_days_left_chart', output_dir)


def generate_visualizations(base_filename: str, dfs: dict, output_charts_dir: str, logger):
    """
    Acts as a router, calling the appropriate plotting function based on the
    filename of the report. This version handles plots that require multiple
    DataFrames.
    """
    try:
        # --- Simple, 1-to-1 plots ---
        if base_filename == 'q1_airlines_frequency':
            plot_airline_frequency(dfs['q1_airlines_frequency'], output_charts_dir, 'q1_airlines_frequency_chart')
        
        elif base_filename == 'q2_departure_time_counts':
            plot_time_distribution(dfs['q2_departure_time_counts'], 'departure_time', output_charts_dir, 'q2_departure_time_counts_chart')

        elif base_filename == 'q2_arrival_time_counts':
            plot_time_distribution(dfs['q2_arrival_time_counts'], 'arrival_time', output_charts_dir, 'q2_arrival_time_counts_chart')
            
        elif base_filename == 'q3_source_city_counts':
            plot_city_distribution(dfs['q3_source_city_counts'], 'source_city', output_charts_dir, 'q3_source_city_counts_chart')
            
        elif base_filename == 'q3_destination_city_counts':
            plot_city_distribution(dfs['q3_destination_city_counts'], 'destination_city', output_charts_dir, 'q3_destination_city_counts_chart')
        
        elif base_filename == 'q6_price_heatmap_matrix':
             plot_price_by_route_heatmap(dfs['q6_price_heatmap_matrix'], output_charts_dir, 'q6_price_heatmap_chart')
        
        # --- Plots combining multiple DataFrames ---
        # These are triggered by one of their input files to avoid duplicate calls.
        elif base_filename == 'q4_price_by_airline_stats':
            plot_price_by_airline(
                dfs['q4_price_by_airline_stats'],
                dfs['master_clean_dataset'], # Needs full dataset for sampling
                output_charts_dir
            )
            
        elif base_filename == 'q5_price_by_departure_time_stats':
            plot_price_by_time(
                dfs['q5_price_by_departure_time_stats'],
                dfs['q5_price_by_arrival_time_stats'],
                output_charts_dir
            )

        elif base_filename == 'q7_last_minute_vs_rest_by_class_mean_price':
            plot_last_minute_booking_effect(
                dfs['q7_last_minute_vs_rest_by_class_mean_price'],
                dfs['q7_average_price_by_days_left'],
                output_charts_dir
            )

        elif base_filename == 'q8_price_by_class_stats':
            plot_price_by_class(
                dfs['q8_price_by_class_stats'],
                dfs['master_clean_dataset'], # Needs full dataset for sampling
                output_charts_dir
            )
            
        elif base_filename == 'q9_vistara_delhi_hyderabad_business_avg_price':
            # Check if there's data for the specific route trend before plotting it
            if 'q9_vistara_dh_price_vs_days_left_trend' in dfs:
                plot_specific_route_analysis(
                    dfs['q9_vistara_delhi_hyderabad_business_avg_price'],
                    dfs['q9_vistara_dh_price_vs_days_left_trend'],
                    output_charts_dir
                )

        # Log success for any plot that was generated.
        #if base_filename in dfs: # A simple check to see if we attempted a plot
        #     logger.success(f"Visualizations generated for report group: '{base_filename}'")

    except Exception as e:
        logger.warning(f"Failed to generate chart for report group '{base_filename}'. Error: {e}")