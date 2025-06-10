import pandas as pd

# Load the CSV files
educations = pd.read_csv('filtered_education_data.csv')

print(educations.describe())