# -*- coding: utf-8 -*-
"""
Utility functions for plotting and saving figures.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


def save_matplotlib_figure(fig, base_filename, output_charts_dir):
    """Saves a Matplotlib figure to the designated charts directory."""
    os.makedirs(output_charts_dir, exist_ok=True)
    chart_path = os.path.join(output_charts_dir, f"{base_filename}.png")
    
    fig.savefig(chart_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def export_plotly_figure(fig, base_filename, output_charts_dir):
    """Saves a Plotly figure as a static PNG and a fallback HTML file to the charts directory."""
    # Saving to PNG requires the 'kaleido' package: pip install -U kaleido
    os.makedirs(output_charts_dir, exist_ok=True)
    chart_path_png = os.path.join(output_charts_dir, f"{base_filename}.png")
    
    try:
        fig.write_image(chart_path_png, scale=2)
    except Exception as e:
        print(f"[WARNING] Failed to save Plotly figure as PNG. Saving as HTML instead. Error: {e}")
        chart_path_html = os.path.join(output_charts_dir, f"{base_filename}.html")
        fig.write_html(chart_path_html)

def format_matplotlib_date_axis(ax, fmt='%Y-%m-%d', rotation=45):
    """
    Properly formats the date axis on a Matplotlib chart to prevent scientific notation
    and improve readability.
    """
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_horizontalalignment('right')
    plt.tight_layout()

def create_barplot_with_optional_hue(ax, data, x=None, y=None, hue=None, palette=None, legend=False, color_when_no_hue='tab:blue'):
    """
    Draws a barplot, carefully handling the 'hue' parameter to avoid warnings.
    If 'hue' is not provided, a single color is used.
    If 'hue' is provided, a palette is used.
    """
    if hue is None:
        sns.barplot(data=data, x=x, y=y, color=color_when_no_hue, ax=ax)
        if legend and ax.get_legend() is not None:
            ax.get_legend().remove()
    else:
        sns.barplot(data=data, x=x, y=y, hue=hue, palette=(palette or 'tab10'), ax=ax, legend=legend)