import unittest
import pandas as pd
import numpy as np
from model import preprocess_data, normalize_input, predict_attrition_likelihood, load_data, check_numeric_features, define_fuzzy_variables, define_fuzzy_rules, predict_attrition

class TestModel(unittest.TestCase):

    def test_preprocess_data(self):
        data = pd.DataFrame({
            'Age': [25, 30, np.nan],
            'Gender': ['Male', 'Female', 'Male'],
            'OverTime': ['Yes', 'No', 'Yes'],
            'Department': ['Sales', 'Research', 'Sales'], # Example of another categorical column
            'YearsAtCompany': [1, 5, np.nan] # Example of another numerical column with NaN
        })
        
        # Add other relevant columns as per features used in model.py
        # For this test, we'll assume these are the only ones needed to test preprocess_data's core logic
        
        processed_data = preprocess_data(data.copy())
        
        self.assertIn('Gender_Male', processed_data.columns)
        self.assertEqual(list(processed_data['Gender_Male']), [1, 0, 1])
        
        self.assertIn('OverTime_Yes', processed_data.columns)
        self.assertTrue(pd.api.types.is_integer_dtype(processed_data['OverTime_Yes']))
        self.assertEqual(list(processed_data['OverTime_Yes']), [1, 0, 1])
        
        self.assertFalse(processed_data['Age'].isnull().any())
        # Check if the NaN in Age was filled with the mean of the non-NaN values (25, 30) -> mean = 27.5
        self.assertTrue(27.5 in processed_data['Age'].values) 

        self.assertFalse(processed_data['YearsAtCompany'].isnull().any())
        # Check if the NaN in YearsAtCompany was filled with the mean (1, 5) -> mean = 3.0
        self.assertTrue(3.0 in processed_data['YearsAtCompany'].values)

        # Check if other categorical columns (like Department) were dummified
        self.assertIn('Department_Sales', processed_data.columns) # Department_Research would be the other

    def test_normalize_input(self):
        processed_df_for_norm = pd.DataFrame({
            'Age': [20.0, 30.0, 40.0],
            'YearsAtCompany': [0.0, 5.0, 10.0],
            'OverTime_Yes': [0, 1, 0], # Integer type after preprocessing
            'Gender_Male': [1, 0, 1],   # Integer type after preprocessing
            'WorkLifeBalance': [1.0, 2.0, 3.0],
            'EnvironmentSatisfaction': [1.0, 2.0, 3.0],
            'JobInvolvement': [1.0, 2.0, 3.0],
            'FeatureWithNoVariance': [5.0, 5.0, 5.0] # For testing max_val - min_val == 0
        })
        features_to_use = ['Age', 'YearsAtCompany', 'OverTime_Yes', 'Gender_Male', 
                             'WorkLifeBalance', 'EnvironmentSatisfaction', 'JobInvolvement', 
                             'FeatureWithNoVariance', 'MissingFeatureTest']

        input_dict = {
            'Age': 25, 
            'YearsAtCompany': 2, 
            'OverTime_Yes': 1, 
            'Gender_Male': 1, 
            'WorkLifeBalance': 2, 
            'EnvironmentSatisfaction': 3, 
            'JobInvolvement': 4, # Test value outside of min/max in processed_df_for_norm
            'FeatureWithNoVariance': 5
        }
        # 'MissingFeatureTest' is deliberately not in input_dict to test that case

        normalized = normalize_input(input_dict, processed_df_for_norm, features_to_use)

        self.assertEqual(normalized['Age'], 0.25) # (25-20)/(40-20) = 0.25
        self.assertEqual(normalized['YearsAtCompany'], 0.2) # (2-0)/(10-0) = 0.2
        self.assertEqual(normalized['OverTime_Yes'], 1.0) # (1-0)/(1-0) = 1.0 (assuming 0 and 1 are min/max after get_dummies)
        self.assertEqual(normalized['Gender_Male'], 1.0) # (1-0)/(1-0) = 1.0
        self.assertEqual(normalized['WorkLifeBalance'], 0.5) # (2-1)/(3-1) = 0.5
        self.assertEqual(normalized['EnvironmentSatisfaction'], 1.0) # (3-1)/(3-1) = 1.0
        
        # For JobInvolvement, input is 4, max in data is 3. (4-1)/(3-1) = 1.5. 
        # The current normalize_input doesn't clip, so this will be > 1. This is expected based on its implementation.
        self.assertEqual(normalized['JobInvolvement'], 1.5) 

        # Test for feature with no variance in the dataset (max_val - min_val == 0)
        self.assertEqual(normalized['FeatureWithNoVariance'], 0)

        # Test for a feature in features_to_use but missing from input_dict
        self.assertEqual(normalized['MissingFeatureTest'], 0)
        
    def test_predict_attrition_likelihood_smoke(self):
        # Create a dummy HR-Employee-Attrition.csv for the test to run without external dependency
        # This ensures the test is self-contained.
        sample_csv_data = {
            'Age': [30, 40, 25, 35, 28],
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'OverTime': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'YearsAtCompany': [5, 10, 2, 8, 3],
            'WorkLifeBalance': [2, 3, 1, 4, 2],
            'EnvironmentSatisfaction': [3, 4, 2, 3, 1],
            'JobInvolvement': [3, 2, 4, 3, 2],
            # Add all other columns that preprocess_data and the model might expect
            'BusinessTravel': ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
            'Department': ['Sales', 'Research & Development', 'Human Resources', 'Sales', 'Research & Development'],
            'DistanceFromHome': [10, 2, 15, 5, 20],
            'Education': [3, 4, 2, 3, 4],
            'EducationField': ['Life Sciences', 'Medical', 'Other', 'Marketing', 'Technical Degree'],
            'EmployeeCount': [1, 1, 1, 1, 1], # Should be dropped by get_dummies if not useful
            'EmployeeNumber': [101, 102, 103, 104, 105],
            'JobLevel': [2, 3, 1, 2, 1],
            'JobRole': ['Sales Executive', 'Research Scientist', 'Human Resources', 'Sales Representative', 'Laboratory Technician'],
            'JobSatisfaction': [4, 3, 2, 4, 1],
            'MaritalStatus': ['Married', 'Single', 'Married', 'Divorced', 'Single'],
            'MonthlyIncome': [5000, 8000, 3000, 6000, 3500],
            'MonthlyRate': [15000, 20000, 10000, 18000, 12000],
            'NumCompaniesWorked': [2, 1, 3, 0, 4],
            'PercentSalaryHike': [15, 12, 20, 10, 18],
            'PerformanceRating': [3, 3, 4, 3, 3], # Usually 3 or 4
            'RelationshipSatisfaction': [3, 4, 2, 3, 1],
            'StandardHours': [80, 80, 80, 80, 80], # Should be dropped by get_dummies
            'StockOptionLevel': [1, 0, 2, 1, 0],
            'TotalWorkingYears': [10, 15, 3, 12, 5],
            'TrainingTimesLastYear': [2, 3, 1, 2, 4],
            'YearsInCurrentRole': [3, 7, 1, 5, 2],
            'YearsSinceLastPromotion': [1, 5, 0, 3, 1],
            'YearsWithCurrManager': [2, 8, 0, 4, 2]
        }
        dummy_df = pd.DataFrame(sample_csv_data)
        dummy_df.to_csv('HR-Employee-Attrition.csv', index=False) # Create the dummy file

        sample_input = {
            'OverTime_Yes': 1,
            'YearsAtCompany': 5,
            'Age': 30,
            'WorkLifeBalance': 2,
            'EnvironmentSatisfaction': 3,
            'JobInvolvement': 3,
            'Gender_Male': 1
        }
        
        likelihood = predict_attrition_likelihood(sample_input)
        
        # Check if likelihood is a number (float or int)
        self.assertTrue(isinstance(likelihood, (int, float, np.floating)), f"Likelihood is not a number: {type(likelihood)}")
        
        # Check if likelihood is within the expected range (0-100)
        self.assertTrue(0 <= likelihood <= 100, f"Likelihood out of range: {likelihood}")

if __name__ == '__main__':
    unittest.main()
