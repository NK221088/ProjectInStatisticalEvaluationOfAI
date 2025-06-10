import pandas as pd
from loadGroups import low, low_mid, upp_mid, high, South_Asia, Europe_and_Central_Asia, Middle_East_and_North_Africa, Sub_Saharan_Africa, Latin_America_and_Caribbean, East_Asia_and_Pacific, North_America, educational_low, educational_med, educational_high

def load_data(filename = "un_member_states_income.csv"):
    """Load the CSV file with semicolon delimiter"""
    return pd.read_csv(filename, delimiter=",")

allData = load_data("answer_Data.csv")

# Load economical data
allData_lowIncome = allData[allData["country"].isin(low["Economy"])]
allData_lowMiddleIncome = allData[allData["country"].isin(low_mid["Economy"])]
allData_uppMiddleIncome = allData[allData["country"].isin(upp_mid["Economy"])]
allData_highIncome = allData[allData["country"].isin(high["Economy"])]

# Load geographical data
allData_South_Asia = allData[allData["country"].isin(South_Asia["Economy"])]
allData_Europe_and_Central_Asia = allData[allData["country"].isin(Europe_and_Central_Asia["Economy"])]
allData_Middle_East_and_North_Africa = allData[allData["country"].isin(Middle_East_and_North_Africa["Economy"])]
allData_Sub_Saharan_Africa = allData[allData["country"].isin(Sub_Saharan_Africa["Economy"])]
allData_Latin_America_and_Caribbean = allData[allData["country"].isin(Latin_America_and_Caribbean["Economy"])]
allData_East_Asia_and_Pacific = allData[allData["country"].isin(East_Asia_and_Pacific["Economy"])]
allData_North_America = allData[allData["country"].isin(North_America["Economy"])]

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

# Load educational data
allData_educational_low = allData[allData["country"].isin(educational_low["Economy"])]
allData_educational_med = allData[allData["country"].isin(educational_med["Economy"])]
allData_educational_high = allData[allData["country"].isin(educational_high["Economy"])]

# Generate descriptive statistics for all datasets
print("=== ECONOMIC DATA ===")
print("\n--- Low Income Countries ---")
print(allData_lowIncome.describe())

print("\n--- Low-Middle Income Countries ---")
print(allData_lowMiddleIncome.describe())

print("\n--- Upper-Middle Income Countries ---")
print(allData_uppMiddleIncome.describe())

print("\n--- High Income Countries ---")
print(allData_highIncome.describe())

print("\n=== GEOGRAPHICAL DATA ===")
print("\n--- South Asia ---")
print(allData_South_Asia.describe())

print("\n--- Europe and Central Asia ---")
print(allData_Europe_and_Central_Asia.describe())

print("\n--- Middle East and North Africa ---")
print(allData_Middle_East_and_North_Africa.describe())

print("\n--- Sub-Saharan Africa ---")
print(allData_Sub_Saharan_Africa.describe())

print("\n--- Latin America and Caribbean ---")
print(allData_Latin_America_and_Caribbean.describe())

print("\n--- East Asia and Pacific ---")
print(allData_East_Asia_and_Pacific.describe())

print("\n--- North America ---")
print(allData_North_America.describe())

print("\n=== EDUCATIONAL DATA ===")
print("\n--- Low Educational Attainment ---")
print(allData_educational_low.describe())

print("\n--- Medium Educational Attainment ---")
print(allData_educational_med.describe())

print("\n--- High Educational Attainment ---")
print(allData_educational_high.describe())