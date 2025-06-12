import pandas as pd

# Load the CSV files
def load_data(filename = "income_un_members_only.csv"):
    """Load the CSV file with semicolon delimiter"""
    return pd.read_csv(filename, delimiter=";")

income_df = load_data()

# Add the income group label for Venezuela
income_df.loc[income_df['Economy'] == 'Venezuela (Bolivarian Republic of)', 'Income group'] = 'Upper middle income'

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

education_df = load_data("education_un_members_only.csv")

col_name = 'Economy,Year,Economy Code,"Educational attainment, at least completed primary, population 25+ years, total (%) (cumulative)"'

# Filter values
education_df = education_df[education_df.columns[0]].str.split(',', expand=True)
education_df.columns = ['Economy', 'Year', 'Economy_Code', 'Educational_Attainment']
education_df["Educational_Attainment"] = pd.to_numeric(education_df["Educational_Attainment"], errors="coerce")
education_df.fillna(education_df["Educational_Attainment"].mean(), inplace=True)
educational_low = education_df[education_df["Educational_Attainment"] < 50]
educational_med = education_df[(education_df["Educational_Attainment"] >= 50) & (education_df["Educational_Attainment"] < 90)]
educational_high = education_df[education_df["Educational_Attainment"] > 90]