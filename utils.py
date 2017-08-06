import numpy as np


def clean_data(data):
    data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
    data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    data["Embarked"] = data["Embarked"].fillna("S")
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2

    # convert empty cells to 0 = No Cabin
    data["Cabin"] = data["Cabin"].fillna(0)

    data.loc[data["Cabin"].str.contains("A")==True, "Cabin"] = 1
    data.loc[data["Cabin"].str.contains("B")==True, "Cabin"] = 2
    data.loc[data["Cabin"].str.contains("C")==True, "Cabin"] = 3
    data.loc[data["Cabin"].str.contains("D")==True, "Cabin"] = 4
    data.loc[data["Cabin"].str.contains("E")==True, "Cabin"] = 5
    data.loc[data["Cabin"].str.contains("F")==True, "Cabin"] = 6
    data.loc[data["Cabin"].str.contains("G")==True, "Cabin"] = 7

    """
    data["Cabin"] = data["Cabin"].to_numeric(convert_numeric=True)

    #cabins = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
    cabins = ["A", "B", "C", "D", "E", "F", "G"]
    data.loc[data["Cabin"].str.contains('|'.join(cabins)), "Cabin"] = 1
    """
