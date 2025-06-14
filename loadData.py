import numpy as np
import pandas as pd
from loadGroups import (low, low_mid, upp_mid, high, 
                       South_Asia, Europe_and_Central_Asia, Middle_East_and_North_Africa, 
                       Sub_Saharan_Africa, Latin_America_and_Caribbean, East_Asia_and_Pacific, North_America,
                       educational_low, educational_med, educational_high)
from dataExtraction import all_df

def load_data(filename="education_un_members_only.csv"):
    """Load the CSV file with semicolon delimiter"""
    return pd.read_csv(filename, delimiter=",")

allData = all_df

# 1. ECONOMIC CATEGORIES (1-4)
economic_conditions = [
    allData['country'].isin(low['Economy']),        # Low income = 1
    allData['country'].isin(low_mid['Economy']),    # Lower middle income = 2
    allData['country'].isin(upp_mid['Economy']),    # Upper middle income = 3
    allData['country'].isin(high['Economy'])        # High income = 4
]
economic_choices = [1, 2, 3, 4]
allData['economic_category'] = np.select(economic_conditions, economic_choices)

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
allData['geographical_category'] = np.select(geographical_conditions, geographical_choices)

# 3. EDUCATIONAL CATEGORIES (1-3)
educational_conditions = [
    allData['country'].isin(educational_low['Economy']),    # Low education = 1
    allData['country'].isin(educational_med['Economy']),    # Medium education = 2
    allData['country'].isin(educational_high['Economy'])    # High education = 3
]
educational_choices = [1, 2, 3]
allData['educational_category'] = np.select(educational_conditions, educational_choices)

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

# Add these diagnostic sections to your existing code

# 1. EXAMINE THE EDUCATIONAL CATEGORY DATA STRUCTURE
print("\n=== EDUCATIONAL CATEGORY DATA INSPECTION ===")
print("Educational_low structure:")
print(f"Type: {type(educational_low)}")
print(f"Keys: {list(educational_low.keys()) if isinstance(educational_low, dict) else 'Not a dict'}")
if isinstance(educational_low, dict) and 'Economy' in educational_low:
    print(f"Sample countries in educational_low: {list(educational_low['Economy'])[:5]}")
    print(f"Total countries in educational_low: {len(educational_low['Economy'])}")

print("\nEducational_med structure:")
print(f"Type: {type(educational_med)}")
print(f"Keys: {list(educational_med.keys()) if isinstance(educational_med, dict) else 'Not a dict'}")
if isinstance(educational_med, dict) and 'Economy' in educational_med:
    print(f"Sample countries in educational_med: {list(educational_med['Economy'])[:5]}")
    print(f"Total countries in educational_med: {len(educational_med['Economy'])}")

print("\nEducational_high structure:")
print(f"Type: {type(educational_high)}")
print(f"Keys: {list(educational_high.keys()) if isinstance(educational_high, dict) else 'Not a dict'}")
if isinstance(educational_high, dict) and 'Economy' in educational_high:
    print(f"Sample countries in educational_high: {list(educational_high['Economy'])[:5]}")
    print(f"Total countries in educational_high: {len(educational_high['Economy'])}")

# 2. CHECK YOUR DATA COUNTRIES
print("\n=== YOUR DATA COUNTRIES ===")
print(f"Sample countries in your data: {list(allData['country'].head(10))}")
print(f"Total countries in your data: {len(allData['country'].unique())}")
print(f"Unique countries: {sorted(allData['country'].unique())}")

# 3. FIND UNCATEGORIZED COUNTRIES
uncategorized_edu = allData[allData['educational_category'] == 0]['country'].unique()
print(f"\n=== UNCATEGORIZED EDUCATIONAL COUNTRIES ===")
print(f"Number of uncategorized countries: {len(uncategorized_edu)}")
print(f"Uncategorized countries: {sorted(uncategorized_edu)}")

# 4. CHECK FOR EXACT MATCHES
print("\n=== EXACT MATCH TESTING ===")
# Test a few countries manually
test_countries = list(allData['country'].unique())[:5]
for country in test_countries:
    print(f"\nTesting country: '{country}'")
    in_low = country in educational_low['Economy'] if isinstance(educational_low, dict) and 'Economy' in educational_low else False
    in_med = country in educational_med['Economy'] if isinstance(educational_med, dict) and 'Economy' in educational_med else False
    in_high = country in educational_high['Economy'] if isinstance(educational_high, dict) and 'Economy' in educational_high else False
    print(f"  In educational_low: {in_low}")
    print(f"  In educational_med: {in_med}")
    print(f"  In educational_high: {in_high}")

# 5. CHECK FOR CASE SENSITIVITY AND WHITESPACE ISSUES
print("\n=== CASE SENSITIVITY & WHITESPACE CHECK ===")
# Sample a few countries from each educational category
sample_countries = []
if isinstance(educational_low, dict) and 'Economy' in educational_low:
    sample_countries.extend(list(educational_low['Economy'])[:3])
if isinstance(educational_med, dict) and 'Economy' in educational_med:
    sample_countries.extend(list(educational_med['Economy'])[:3])
if isinstance(educational_high, dict) and 'Economy' in educational_high:
    sample_countries.extend(list(educational_high['Economy'])[:3])

for country in sample_countries:
    print(f"Educational category country: '{country}' (length: {len(country)})")
    # Check if it exists in your data with different casing
    matches = allData[allData['country'].str.lower().str.strip() == country.lower().strip()]
    print(f"  Matches in your data: {len(matches)}")
    if len(matches) > 0:
        print(f"  Your data version: '{matches.iloc[0]['country']}'")

# 6. FUZZY MATCHING TO FIND SIMILAR COUNTRY NAMES
print("\n=== POTENTIAL NAME MISMATCHES ===")
# Check if countries exist with slight variations
your_countries = set(allData['country'].str.lower().str.strip())
edu_countries = set()
if isinstance(educational_low, dict) and 'Economy' in educational_low:
    edu_countries.update([c.lower().strip() for c in educational_low['Economy']])
if isinstance(educational_med, dict) and 'Economy' in educational_med:
    edu_countries.update([c.lower().strip() for c in educational_med['Economy']])
if isinstance(educational_high, dict) and 'Economy' in educational_high:
    edu_countries.update([c.lower().strip() for c in educational_high['Economy']])

print(f"Countries in your data but not in educational categories: {len(your_countries - edu_countries)}")
print(f"Countries in educational categories but not in your data: {len(edu_countries - your_countries)}")

# Show some examples
missing_from_edu = list(your_countries - edu_countries)[:10]
missing_from_data = list(edu_countries - your_countries)[:10]
print(f"Sample countries missing from educational categories: {missing_from_edu}")
print(f"Sample countries missing from your data: {missing_from_data}")

# 7. VERIFY EDUCATIONAL CATEGORY ASSIGNMENT LOGIC
print("\n=== EDUCATIONAL CATEGORY ASSIGNMENT VERIFICATION ===")
# Test the np.select logic manually
for i, country in enumerate(allData['country'].head(5)):
    conditions_result = [
        country in educational_low['Economy'] if isinstance(educational_low, dict) and 'Economy' in educational_low else False,
        country in educational_med['Economy'] if isinstance(educational_med, dict) and 'Economy' in educational_med else False,
        country in educational_high['Economy'] if isinstance(educational_high, dict) and 'Economy' in educational_high else False
    ]
    print(f"Country: {country}")
    print(f"  Conditions: {conditions_result}")
    print(f"  Assigned category: {allData.iloc[i]['educational_category']}")
    print(f"  Expected category: {np.select(conditions_result, [1, 2, 3], default=0)}")