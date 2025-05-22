# -*- coding: utf-8 -*-
"""employee_attrition_with_gender.ipynb"""

import os
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from model import (
    load_data, preprocess_data, check_numeric_features, normalize_features,
    define_fuzzy_variables, define_fuzzy_rules, predict_attrition, normalize_input
)

def main():
   # Load and preprocess data
   data_file_path = os.environ.get('DATA_FILE_PATH', 'HR-Employee-Attrition.csv')
   data = load_data(data_file_path)
   processed_data = preprocess_data(data)
   # Select features and normalize them
   features_to_use = [
       'OverTime_Yes', 'YearsAtCompany', 'Age',
       'WorkLifeBalance', 'EnvironmentSatisfaction',
       'JobInvolvement', 'Gender_Male'
   ]

   try:
        # Ensure that 'Gender_Male' is created during preprocessing if not already present
        if 'Gender_Male' not in processed_data.columns and 'Gender_Female' in processed_data.columns:
            processed_data['Gender_Male'] = (~processed_data['Gender_Female'].astype(bool)).astype(int)
        elif 'Gender_Male' not in processed_data.columns:
            # Handle cases where gender information might be missing or encoded differently
            # This might involve more complex logic depending on the dataset
            # For now, if 'Gender_Male' is missing, we can't proceed with it.
            # Or, we can create a default column if appropriate (e.g., all zeros or based on other data)
            # For this example, let's assume it must be present after preprocessing or we raise error.
            if 'Gender_Male' not in processed_data.columns:
                 raise ValueError("Gender_Male column is missing and could not be derived.")


        normalized_features = normalize_features(processed_data, features_to_use)

        # Define fuzzy variables and rules
        over_time_var, years_var, age_var, work_var, env_var, job_var, gender_var = define_fuzzy_variables()
        simulation_control_system = define_fuzzy_rules(over_time_var, years_var,
                                                    age_var, work_var,
                                                    env_var, job_var, gender_var)

        simulation_instance = ctrl.ControlSystemSimulation(simulation_control_system)

        input_data_example_1={
            'OverTime_Yes': 1,
            'YearsAtCompany': 0,
            'Age': 18,
            'WorkLifeBalance': 1,
            'EnvironmentSatisfaction': 1,
            'JobInvolvement': 1,
            'Gender_Male': 1 # Example: Male
        }
        normalized_input_1 = normalize_input(input_data_example_1, processed_data, features_to_use)
        likelihood_1 = predict_attrition(simulation_instance, normalized_input_1)
        print(f'Predicted Attrition Likelihood (Example 1): {likelihood_1:.2f}')

        input_data_example_2={
            'OverTime_Yes': 0,
            'YearsAtCompany': 40,
            'Age': 60,
            'WorkLifeBalance': 4,
            'EnvironmentSatisfaction': 4,
            'JobInvolvement': 4,
            'Gender_Male': 0 # Example: Female
        }
        normalized_input_2 = normalize_input(input_data_example_2, processed_data, features_to_use)
        likelihood_2 = predict_attrition(simulation_instance, normalized_input_2)
        print(f'Predicted Attrition Likelihood (Example 2): {likelihood_2:.2f}')

   except ValueError as ve:
       print(f"ValueError: {ve}")

if __name__ == "__main__":
   main()