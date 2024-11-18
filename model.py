import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Include all the functions from the original script here
# (load_data, preprocess_data, check_numeric_features, normalize_features,
# define_fuzzy_variables, define_fuzzy_rules, predict_attrition, normalize_input)

# Modify the main function to return the prediction

import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


def load_data(file_path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """Preprocess the data by encoding categorical variables and handling missing values."""
    # Convert categorical variables to numeric using one-hot encoding
    data = pd.get_dummies(data, drop_first=True)
    # Convert OverTime_Yes to numeric (True=1, False=0)
    if 'OverTime_Yes' in data.columns:
        data['OverTime_Yes'] = data['OverTime_Yes'].astype(int)
    if 'Gender_Male' in data.columns:
        data['Gender_Male'] = data['Gender_Male'].astype(int)
    print(data)
    # Check for missing values and handle them
    data.fillna(data.mean(), inplace=True)

    return data


def check_numeric_features(data, features):
    """Check for non-numeric features in the selected columns."""
    non_numeric_features = []
    for feature in features:
        if not np.issubdtype(data[feature].dtype, np.number):
            non_numeric_features.append(feature)

    if non_numeric_features:
        print("Non-numeric features found:", non_numeric_features)

    return len(non_numeric_features) == 0  # Return True if all are numeric


def normalize_features(data, features):
    """Normalize the selected features."""
    features_df = data[features]

    # Ensure all columns are numeric before normalization
    if not check_numeric_features(data, features):
        raise ValueError("All features must be numeric for normalization.")

    normalized_features = (features_df - features_df.min()) / \
        (features_df.max() - features_df.min())

    print("Normalized Features")
    print(normalize_features)
    return normalized_features


def define_fuzzy_variables():
    """Define fuzzy variables and membership functions."""
    over_time = ctrl.Antecedent(
        np.arange(0, 2, 1), 'OverTime_Yes')  # Binary: Yes=1, No=0
    years_at_company = ctrl.Antecedent(
        np.arange(0, 1.1, 0.1), 'YearsAtCompany')
    age = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Age')
    work_life_balance = ctrl.Antecedent(
        np.arange(0, 1.1, 0.1), 'WorkLifeBalance')
    gender = ctrl.Antecedent(np.arange(0, 2, 1), 'Gender_Male')
    environment_satisfaction = ctrl.Antecedent(
        np.arange(0, 1.1, 0.1), 'EnvironmentSatisfaction')
    job_involvement = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'JobInvolvement')
    gender = ctrl.Antecedent(np.arange(0, 2, 1), 'Gender_Male')

    # Define membership functions
    over_time['no'] = fuzz.trimf(over_time.universe, [0, 0, 0.5])
    over_time['yes'] = fuzz.trimf(over_time.universe, [0.5, 1, 1])

    years_at_company['short'] = fuzz.trimf(
        years_at_company.universe, [0, 0, 0.5])
    years_at_company['medium'] = fuzz.trimf(
        years_at_company.universe, [0, 0.5, 1])
    years_at_company['long'] = fuzz.trimf(
        years_at_company.universe, [0.5, 1, 1])

    age['young'] = fuzz.trimf(age.universe, [0, 0, 0.5])
    age['middle_aged'] = fuzz.trimf(age.universe, [0, 0.5, 1])
    age['old'] = fuzz.trimf(age.universe, [0.5, 1, 1])

    work_life_balance['poor'] = fuzz.trimf(
        work_life_balance.universe, [0, 0, 0.5])
    work_life_balance['average'] = fuzz.trimf(
        work_life_balance.universe, [0, 0.5, 1])
    work_life_balance['good'] = fuzz.trimf(
        work_life_balance.universe, [0.5, 1, 1])

    gender['female'] = fuzz.trimf(gender.universe, [0, 0, 0.5])
    gender['male'] = fuzz.trimf(gender.universe, [0.5, 1, 1])

    environment_satisfaction['low'] = fuzz.trimf(
        environment_satisfaction.universe, [0, 0, 0.5])
    environment_satisfaction['medium'] = fuzz.trimf(
        environment_satisfaction.universe, [0, 0.5, 1])
    environment_satisfaction['high'] = fuzz.trimf(
        environment_satisfaction.universe, [0.5, 1, 1])

    job_involvement['low'] = fuzz.trimf(job_involvement.universe, [0, 0, 0.5])
    job_involvement['medium'] = fuzz.trimf(
        job_involvement.universe, [0, 0.5, 1])
    job_involvement['high'] = fuzz.trimf(job_involvement.universe, [0.5, 1, 1])

    return over_time, years_at_company, age, work_life_balance, environment_satisfaction, job_involvement, gender


def define_fuzzy_rules(over_time_var, years_var, age_var,
                       work_var, env_var, job_var, gender):
    """Define fuzzy rules based on decision tree logic."""

    attrition = ctrl.Consequent(np.arange(0, 101, 1), 'Attrition')

    attrition['low'] = fuzz.trimf(attrition.universe, [0, 10, 50])
    attrition['medium'] = fuzz.trimf(attrition.universe, [30, 50, 70])
    attrition['high'] = fuzz.trimf(attrition.universe, [70, 80, 100])

    # Define fuzzy rules
    rule1 = ctrl.Rule(over_time_var['yes'], attrition['high'])
    rule2 = ctrl.Rule(over_time_var['no'], attrition['low'])

    rule3 = ctrl.Rule(years_var['short'], attrition['high'])
    rule4 = ctrl.Rule(years_var['long'], attrition['low'])

    rule5 = ctrl.Rule(age_var['young'], attrition['high'])
    rule6 = ctrl.Rule(age_var['old'], attrition['low'])

    rule7 = ctrl.Rule(work_var["poor"], attrition["high"])
    rule8 = ctrl.Rule(work_var["good"], attrition["low"])

    rule9 = ctrl.Rule(env_var["low"], attrition["high"])
    rule10 = ctrl.Rule(env_var["high"], attrition["low"])

    rule11 = ctrl.Rule(job_var["low"], attrition["high"])
    rule12 = ctrl.Rule(job_var["high"], attrition["low"])

    rule13 = ctrl.Rule(gender["male"], attrition["high"])
    rule14 = ctrl.Rule(gender["female"], attrition["low"])

    # Create control system
    attrition_ctrl = ctrl.ControlSystem([
        rule1, rule2, rule3, rule4, rule5, rule6,
        rule7, rule8, rule9, rule10, rule11, rule12,
        rule13, rule14
    ])

    return attrition_ctrl


def predict_attrition(simulation, input_data):
    """Simulate and predict attrition likelihood."""
    simulation.input['OverTime_Yes'] = input_data.get(
        'OverTime_Yes', 0)  # Default to no overtime
    simulation.input['YearsAtCompany'] = input_data.get('YearsAtCompany', 0)
    simulation.input['Age'] = input_data.get('Age', 0)
    simulation.input['WorkLifeBalance'] = input_data.get('WorkLifeBalance', 0)
    simulation.input['EnvironmentSatisfaction'] = input_data.get(
        'EnvironmentSatisfaction', 0)
    simulation.input['JobInvolvement'] = input_data.get('JobInvolvement', 0)
    simulation.input['Gender_Male'] = input_data.get(
        'Gender_Male', 0)  # Default to female

    # Compute the result
    simulation.compute()
    return simulation.output["Attrition"]


def normalize_input(input_data, data, features):
    """
    Normalize input data based on the minimum and maximum values of the dataset.
    """
    normalized_data = {}
    for feature in features:
        if feature in input_data:
            min_val = data[feature].min()
            max_val = data[feature].max()
            if max_val - min_val != 0:
                normalized_data[feature] = (
                    input_data[feature] - min_val) / (max_val - min_val)
            else:
                normalized_data[feature] = 0
        else:
            normalized_data[feature] = 0
    return normalized_data


def predict_attrition_likelihood(input_data):
    data_file_path = 'HR-Employee-Attrition.csv'
    data = load_data(data_file_path)
    processed_data = preprocess_data(data)

    features_to_use = [
        'OverTime_Yes', 'YearsAtCompany', 'Age', 'WorkLifeBalance',
        'EnvironmentSatisfaction', 'JobInvolvement', 'Gender_Male'
    ]

    try:
        normalized_features = normalize_features(
            processed_data, features_to_use)
        over_time_var, years_var, age_var, work_var, env_var, job_var, gender_var = define_fuzzy_variables()
        simulation_control_system = define_fuzzy_rules(
            over_time_var, years_var, age_var, work_var, env_var, job_var, gender_var)
        simulation_instance = ctrl.ControlSystemSimulation(
            simulation_control_system)

        normalized_input = normalize_input(
            input_data, processed_data, features_to_use)
        print(normalize_input)
        likelihood = predict_attrition(simulation_instance, normalized_input)
        return likelihood
    except ValueError as ve:
        return str(ve)
