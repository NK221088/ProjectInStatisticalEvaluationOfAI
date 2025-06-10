import pandas as pd
from loadGroups import low

def load_data(filename = "un_member_states_income.csv"):
    """Load the CSV file with semicolon delimiter"""
    return pd.read_csv(filename, delimiter=",")

allData = load_data("answer_Data.csv")

allData_lowIncome = allData[allData["country"].isin(low["Economy"])]

(allData_lowIncome.describe())

# print(allData[allData["country"]])