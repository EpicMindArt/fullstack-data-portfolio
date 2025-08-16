# Configurable Data Analysis Pipeline

This project is a command-line framework for executing reproducible data analysis pipelines. It is designed to be modular and easily configurable via a central YAML file, allowing users to run either a high-level automated Exploratory Data Analysis (EDA) or a deep-dive analysis that answers specific business questions.

The entire system is containerized with Docker, ensuring a consistent and portable analysis environment.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Analysis Modes Explained](#analysis-modes-explained)
- [Adding a New Analysis](#adding-a-new-analysis)
- [Deployment & Portability (Docker)](#deployment--portability-docker)
- [License](#license)

## Features

-   **CLI-Driven**: All analyses are executed from the command line, making it ideal for automation and scripting.
-   **YAML Configuration**: Easily add new datasets and analysis modules by editing a single `config.yaml` file.
-   **Dual Analysis Modes**:
    1.  **Primary EDA**: Automatically generates comprehensive HTML reports (`ydata-profiling`, `sweetviz`) and a JSON summary for a quick dataset overview.
    2.  **Deep-Dive Analysis**: Executes custom Python scripts to answer specific, predefined business questions.
-   **Rich Outputs**: Generates multiple artifacts, including CSV/Excel reports, charts (PNG), and interactive HTML profiles.
-   **Modular Architecture**: Logic is cleanly separated into data I/O, analysis, plotting, and utilities.
-   **Enhanced Console Experience**: Uses `rich` and `loguru` for beautiful, informative terminal output and progress bars.
-   **Dockerized Environment**: Fully containerized with Docker and Docker Compose for a reproducible and dependency-free setup.

## Tech Stack

-   **Core Analysis**: Pandas, NumPy, SciPy
-   **Automated EDA**: `ydata-profiling`, `sweetviz`
-   **Visualization**: Matplotlib, Seaborn, Plotly
-   **CLI & Configuration**: `argparse`, `PyYAML`
-   **Console UI**: `rich`, `loguru`, `tqdm`

## Project Structure

```
Data-Analytics-Portfolio/
├── main.py                     # Main application entrypoint
├── config.yaml                 # Central configuration for all projects
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Instructions to build the Docker image
├── docker-compose.yml          # Configuration for Docker Compose
├── analyses/                   # Modules for "deep-dive" analysis
│   └── airlines_flights.py
├── data/                       # Raw input datasets (.csv)
├── output/                     # All generated reports and charts appear here
├── utils/                      # Helper modules for I/O, plotting, etc.
│   └── primary_analyzer.py     # Logic for the automated EDA
└── visualizers/                # Modules for generating charts from analysis results
    └── airlines_flights_visualizer.py
```

## Quick Start

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Data-Analytics-Portfolio.git
    cd Data-Analytics-Portfolio
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Requires Python 3.12 or earlier.
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run an analysis (see Usage section for more details):**
    ```bash
    python main.py airlines_flights
    ```

## Configuration

All analysis projects are defined in `config.yaml`. To add a new one, simply add a new key with the required parameters.

**Example `config.yaml`:**
```yaml
restaurant_sales:
  description: "Analyzes sales data from a restaurant chain to answer 10+ business questions."
  input_file: "data/[01] Restaurant Sales Data.csv"
  output_dir: "restaurant_sales"
  analysis_module: "analyses.restaurant_sales"
  visualizer_module: "visualizers.restaurant_sales_visualizer"

airlines_flights:
  description: "Analyzes Indian airline flight data to understand pricing and operational patterns."
  input_file: "data/[02] Airlines Flights Data.csv"
  output_dir: "airlines_flights"
  analysis_module: "analyses.airlines_flights"
  visualizer_module: "visualizers.airlines_flights_visualizer"
```

## Usage

The script is run from the terminal using `main.py`. It requires one argument (`project_key`) and has one optional flag (`--primary`).

`python main.py <project_key> [--primary]`

-   **`<project_key>`**: The key from `config.yaml` for the project you want to run (e.g., `airlines_flights`).
-   **`--primary` or `-p`**: (Optional) If this flag is present, the script runs the automated **Primary EDA**. If omitted, it runs the **Deep-Dive Analysis**.

#### Example 1: Run a Deep-Dive Analysis
This command executes the custom script defined in `analyses/airlines_flights.py` to answer specific business questions and generate detailed reports and charts.
```bash
python main.py airlines_flights
```

#### Example 2: Run a Primary Automated EDA
This command runs the automated analysis on the same dataset, generating `ydata-profiling` and `sweetviz` reports.
```bash
python main.py airlines_flights --primary
```
All results will be placed in the `output/` directory, organized by project name.

## Analysis Modes Explained

1.  **Deep-Dive Analysis (Default)**
    -   **Purpose**: To answer specific, complex business questions.
    -   **Process**: Loads data, executes a custom Python script from the `analyses/` directory, and uses a corresponding `visualizers/` script to generate targeted charts.
    -   **Output**: CSV/Excel files and PNG charts tailored to the questions being answered.

2.  **Primary EDA (`--primary` flag)**
    -   **Purpose**: To get a quick, comprehensive, and automated overview of any dataset.
    -   **Process**: Uses the powerful `ydata-profiling` and `sweetviz` libraries to automatically generate detailed reports.
    -   **Output**: Interactive HTML reports detailing variable distributions, correlations, missing values, and more. Also creates a lightweight `summary.json`.

## Adding a New Analysis

1.  **Add Data**: Place your new `.csv` file in the `data/` directory.
2.  **Configure**: Add a new entry to `config.yaml` (e.g., `my_new_analysis`).
3.  **Create Analysis Module**: Create a new file `analyses/my_new_analysis.py`. It must contain a `run_analysis(df_raw_data, config)` function.
4.  **Create Visualizer Module**: Create `visualizers/my_new_analysis_visualizer.py` with a `generate_visualizations(...)` function to handle plotting.
5.  **Run it!**: `python main.py my_new_analysis`

## Deployment & Portability (Docker)

To ensure the analysis environment is consistent and reproducible, you can use Docker. This avoids issues with Python versions or dependency conflicts.

First, create the `Dockerfile` and `docker-compose.yml` files in your project root.

---

#### Method 1: Docker Compose (Recommended)

This is the easiest way to manage the container and its volumes.

1.  **Build the Docker image:**
    This command needs to be run only once, or whenever you change `requirements.txt`.
    ```bash
    docker-compose build
    ```

2.  **Run an analysis:**
    Use `docker-compose run` to execute commands inside the container. The `--rm` flag automatically cleans up the container after it exits.


	Run a primary EDA (works always and automatically):
    ```bash    
    docker-compose run --rm portfolio python main.py restaurant_sales --primary
     ```

	Run a deep-dive analysis (you need to create a files from scratch in the "analyses" and "visualizers" folders):
	```bash	
	docker-compose run --rm portfolio python main.py restaurant_sales
	```



Generated files will appear in your local `output/` folder.

#### Method 2: Docker Run (Manual)

If you prefer not to use Docker Compose, you can manage the container manually.

1.  **Build the Docker image:**
    ```bash
    docker build -t data-analytics-portfolio .
    ```

2.  **Run an analysis:**
    This command is more complex as you must manually specify the volume mounts (`-v`).
    ```bash
    docker run --rm -it \
      -v ./output:/app/output \
      -v ./config.yaml:/app/config.yaml \
      -v ./data:/app/data \
      data-analytics-portfolio python main.py airlines_flights
    ```

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/license/mit) file for details.