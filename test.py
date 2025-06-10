import pandas as pd

# Load the CSV files
def load_data(filename = "un_member_states_income.csv"):
    """Load the CSV file with semicolon delimiter"""
    return pd.read_csv(filename, delimiter=";")

income_df = load_data()
low = income_df[income_df["Income group"] == "Low income"]
low_mid = income_df[income_df["Income group"] == "Lower middle income"]
upp_mid = income_df[income_df["Income group"] == "Upper middle income"]
high = income_df[income_df["Income group"] == "High income"]

South_Asia = income_df[income_df["Region"] == "South Asia"]
Europe_and_Central_Asia = income_df[income_df["Region"] == "Europe & Central Asia"]
Middle_East_and_North_Africa = income_df[income_df["Region"] == "Middle East & North Africa"]
Sub_Saharan_Africa = income_df[income_df["Region"] == "Sub-Saharan Africa"]
Latin_America_and_Caribbean = income_df[income_df["Region"] == "Latin America & Caribbean"]
East_Asia_and_Pacific = income_df[income_df["Region"] == "East Asia & Pacific"]
North_America = income_df[income_df["Region"] == "North America"]

education_df = load_data("filtered_education_data.csv")

col_name = 'Economy,Year,Economy Code,"Educational attainment, at least completed primary, population 25+ years, total (%) (cumulative)"'

# Split by comma and take the last part, then convert to numeric
numeric_values = pd.to_numeric(education_df[col_name].str.split(',').str[-1], errors='coerce')

# Filter values
educational_low = education_df[numeric_values < 50]
educational_med = education_df[(numeric_values >= 50) & (numeric_values < 90)]
educational_high = education_df[numeric_values > 90]