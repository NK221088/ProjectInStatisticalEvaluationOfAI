import pandas as pd
from loadGroups import low, low_mid, upp_mid, high, South_Asia, Europe_and_Central_Asia, Middle_East_and_North_Africa, Sub_Saharan_Africa, Latin_America_and_Caribbean, East_Asia_and_Pacific, North_America, educational_low, educational_med, educational_high

def load_data(filename = "un_member_states_income.csv"):
    """Load the CSV file with semicolon delimiter"""
    return pd.read_csv(filename, delimiter=",")

allData = load_data("answer_Data.csv")

allData_lowIncome = allData[allData["country"].isin(low["Economy"])]
allData_lowMiddleIncome = allData[allData["country"].isin(low_mid["Economy"])]
allData_uppMiddleIncome = allData[allData["country"].isin(upp_mid["Economy"])]
allData_highIncome = allData[allData["country"].isin(high["Economy"])]


allData_lowIncome = allData_lowIncome.describe()
allData_lowMiddleIncome = allData_lowMiddleIncome.describe()
allData_uppMiddleIncome = allData_uppMiddleIncome.describe()
allData_highIncome = allData_highIncome.describe()
