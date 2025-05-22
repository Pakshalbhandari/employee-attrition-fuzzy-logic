# Employee Attrition Prediction using Fuzzy Logic

This project predicts employee attrition likelihood using a fuzzy logic-based system. It provides a web interface to input employee data and view the predicted attrition score.

## Project Structure

The project is organized as follows:

-   `app.py`: The main Flask web application that serves the user interface and handles prediction requests.
-   `model.py`: Contains the core fuzzy logic model for attrition prediction. This includes data loading, preprocessing, fuzzy variable definition, rule setup, and prediction logic.
-   `employee_attrition.py`: A command-line script for testing or demonstrating the model independently. It imports and utilizes the logic from `model.py`.
-   `HR-Employee-Attrition.csv`: The dataset used for training and running the model. It contains various employee attributes.
-   `templates/index.html`: The HTML template for the web interface, allowing users to input data and see prediction results.
-   `static/style.css`: The CSS file that styles the web interface.
-   `requirements.txt`: Lists the Python dependencies required to run the project.
-   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
-   `README.md`: This file, providing an overview and instructions for the project.

## User Interface Features

-   **Dynamic Gauge Chart**: The web interface now includes a dynamic gauge chart to visually represent the predicted employee attrition likelihood, providing an immediate and clear understanding of the model's output. This visualization is implemented using Chart.js, loaded via a CDN.

## Model Overview

The model uses a fuzzy logic approach to predict the likelihood of employee attrition. Key factors considered in the model include:

-   Overtime
-   Years at Company
-   Age
-   Work-Life Balance
-   Environment Satisfaction
-   Job Involvement
-   Gender

These factors are defined as fuzzy variables with appropriate membership functions. A set of fuzzy rules, derived from insights or decision tree logic, is then used to determine the attrition likelihood, which is a score typically ranging from 0 to 100.

## Configuration

The path to the data file (`.csv`) used by the model can be configured via an environment variable:

-   **`DATA_FILE_PATH`**: Set this environment variable to the full path of your data file.
    If this variable is not set, the application defaults to using `HR-Employee-Attrition.csv` in the project's root directory.

    Example (Linux/macOS):
    ```bash
    export DATA_FILE_PATH="/path/to/your/datafile.csv"
    ```
    Example (Windows CMD):
    ```cmd
    set DATA_FILE_PATH="C:\path\to\your\datafile.csv"
    ```
    Example (Windows PowerShell):
    ```powershell
    $env:DATA_FILE_PATH="C:\path\to\your\datafile.csv"
    ```

## To Run

1.  **Install Dependencies:**
    Ensure you have Python 3 installed. Then, install the required packages using pip:
    ```bash
    pip3 install -r requirements.txt
    ```

2.  **Run the Web Application:**
    To start the Flask web server:
    ```bash
    python3 app.py
    ```
    The application will typically be available at `http://127.0.0.1:5000/` in your web browser.

3.  **Run the Command-Line Test Script (Optional):**
    To test the model directly from the command line using `employee_attrition.py`:
    ```bash
    python3 employee_attrition.py
    ```
    This script will output sample predictions to the console. You can modify this script to test different input scenarios.

## Error Handling
The web application will display an error message on the page if the model encounters an issue (e.g., data file not found, issues with input data).
The `predict_attrition_likelihood` function in `model.py` returns a string with an error message in case of failure.
The `employee_attrition.py` script will print ValueErrors to the console if issues arise during its execution.
