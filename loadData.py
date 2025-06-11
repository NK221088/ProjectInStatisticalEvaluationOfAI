import numpy as np
import pandas as pd
from loadGroups import (low, low_mid, upp_mid, high, 
                       South_Asia, Europe_and_Central_Asia, Middle_East_and_North_Africa, 
                       Sub_Saharan_Africa, Latin_America_and_Caribbean, East_Asia_and_Pacific, North_America,
                       educational_low, educational_med, educational_high)

def load_data(filename="un_member_states_income.csv"):
    """Load the CSV file with semicolon delimiter"""
    return pd.read_csv(filename, delimiter=",")

allData = load_data("answer_Data.csv")

# Fix educational data - split columns for all three DataFrames
educational_low_split = educational_low[educational_low.columns[0]].str.split(',', expand=True)
educational_low_split.columns = ['Economy', 'Year', 'Economy_Code', 'Educational_Attainment_1', 'Educational_Attainment_2']
educational_low = educational_low_split

educational_med_split = educational_med[educational_med.columns[0]].str.split(',', expand=True)
educational_med_split.columns = ['Economy', 'Year', 'Economy_Code', 'Educational_Attainment_1', 'Educational_Attainment_2']
educational_med = educational_med_split

educational_high_split = educational_high[educational_high.columns[0]].str.split(',', expand=True)
educational_high_split.columns = ['Economy', 'Year', 'Economy_Code', 'Educational_Attainment_1', 'Educational_Attainment_2']
educational_high = educational_high_split

# 1. ECONOMIC CATEGORIES (1-4)
economic_conditions = [
    allData['country'].isin(low['Economy']),        # Low income = 1
    allData['country'].isin(low_mid['Economy']),    # Lower middle income = 2
    allData['country'].isin(upp_mid['Economy']),    # Upper middle income = 3
    allData['country'].isin(high['Economy'])        # High income = 4
]
economic_choices = [1, 2, 3, 4]
allData['economic_category'] = np.select(economic_conditions, economic_choices, default=0)

# 2. GEOGRAPHICAL CATEGORIES (1-7)
geographical_conditions = [
    allData['country'].isin(South_Asia['Economy']),                          # 1
    allData['country'].isin(Europe_and_Central_Asia['Economy']),             # 2
    allData['country'].isin(Middle_East_and_North_Africa['Economy']),        # 3
    allData['country'].isin(Sub_Saharan_Africa['Economy']),                  # 4
    allData['country'].isin(Latin_America_and_Caribbean['Economy']),         # 5
    allData['country'].isin(East_Asia_and_Pacific['Economy']),               # 6
    allData['country'].isin(North_America['Economy'])                        # 7
]
geographical_choices = [1, 2, 3, 4, 5, 6, 7]
allData['geographical_category'] = np.select(geographical_conditions, geographical_choices, default=0)

# 3. EDUCATIONAL CATEGORIES (1-3)
educational_conditions = [
    allData['country'].isin(educational_low['Economy']),    # Low education = 1
    allData['country'].isin(educational_med['Economy']),    # Medium education = 2
    allData['country'].isin(educational_high['Economy'])    # High education = 3
]
educational_choices = [1, 2, 3]
allData['educational_category'] = np.select(educational_conditions, educational_choices, default=0)

# OPTIONAL: Check for overlaps and distribution
print("=== ECONOMIC CATEGORIES ===")
print("Distribution:")
print(allData['economic_category'].value_counts().sort_index())
print(f"Uncategorized countries: {sum(allData['economic_category'] == 0)}")

print("\n=== GEOGRAPHICAL CATEGORIES ===")
print("Distribution:")
print(allData['geographical_category'].value_counts().sort_index())
print(f"Uncategorized countries: {sum(allData['geographical_category'] == 0)}")

print("\n=== EDUCATIONAL CATEGORIES ===")
print("Distribution:")
print(allData['educational_category'].value_counts().sort_index())
print(f"Uncategorized countries: {sum(allData['educational_category'] == 0)}")

# Check for potential overlaps in each category type
print("\n=== OVERLAP CHECKS ===")

# Economic overlaps
econ_sum = (allData['country'].isin(low['Economy']).astype(int) + 
           allData['country'].isin(low_mid['Economy']).astype(int) + 
           allData['country'].isin(upp_mid['Economy']).astype(int) + 
           allData['country'].isin(high['Economy']).astype(int))
print(f"Countries in multiple economic categories: {sum(econ_sum > 1)}")

# Geographical overlaps  
geo_sum = (allData['country'].isin(South_Asia['Economy']).astype(int) + 
          allData['country'].isin(Europe_and_Central_Asia['Economy']).astype(int) + 
          allData['country'].isin(Middle_East_and_North_Africa['Economy']).astype(int) + 
          allData['country'].isin(Sub_Saharan_Africa['Economy']).astype(int) + 
          allData['country'].isin(Latin_America_and_Caribbean['Economy']).astype(int) + 
          allData['country'].isin(East_Asia_and_Pacific['Economy']).astype(int) + 
          allData['country'].isin(North_America['Economy']).astype(int))
print(f"Countries in multiple geographical categories: {sum(geo_sum > 1)}")

# Educational overlaps
edu_sum = (allData['country'].isin(educational_low['Economy']).astype(int) + 
          allData['country'].isin(educational_med['Economy']).astype(int) + 
          allData['country'].isin(educational_high['Economy']).astype(int))
print(f"Countries in multiple educational categories: {sum(edu_sum > 1)}")

# Show some examples
print("\n=== SAMPLE DATA ===")
print(allData[['country', 'economic_category', 'geographical_category', 'educational_category']].head(10))

# Show countries with overlaps if any exist
if sum(econ_sum > 1) > 0:
    print("\nCountries in multiple economic categories:")
    print(allData[econ_sum > 1][['country', 'economic_category']])

if sum(geo_sum > 1) > 0:
    print("\nCountries in multiple geographical categories:")
    print(allData[geo_sum > 1][['country', 'geographical_category']])
    
if sum(edu_sum > 1) > 0:
    print("\nCountries in multiple educational categories:")
    print(allData[edu_sum > 1][['country', 'educational_category']])