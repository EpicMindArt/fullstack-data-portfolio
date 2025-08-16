## From Raw Data to Business Insights

This project is a universal tool for conducting a full cycle of data analysis. It is driven from the command line and through a single configuration file, ensuring 100% reproducibility of results.

### Key Features

*   **Dual Analysis Modes:**
    1.  **Primary EDA:** Automatically generates interactive HTML reports (`ydata-profiling`, `sweetviz`) for a quick overview of any dataset.
    2.  **Deep-Dive Analysis:** Executes custom Python scripts to answer specific, complex business questions.
*   **Modular Architecture:** Logic is cleanly separated into modules for data I/O, analysis, visualization, and utilities. Adding a new analysis involves creating just two Python files.
*   **Rich Outputs:** The pipeline generates multiple artifacts, including CSV/Excel reports, charts (PNG, Plotly), and interactive HTML profiles.

### Case Study: HR Compensation & Pay Equity Analysis

One of the project's modules tackles a real-world business problem: **analyzing pay equity**. This case study addressed the following:
*   **Gender Pay Gap Assessment:** Using OLS regression (`statsmodels`), the adjusted pay gap was calculated, controlling for role, experience, seniority, and location.
*   **Cost-to-Parity Calculation:** The budget required to bring all salaries to a "parity corridor" (e.g., ±2%) was estimated.
*   **Salary Compression Analysis:** Identified segments where employee experience is poorly monetized.
*   **Compensation Band Definition:** Calculated market-rate salary bands (p25-p75) for key roles to prevent internal inconsistencies.

This case study demonstrates the ability not just to process data, but to apply statistical methods to solve complex and sensitive business problems.