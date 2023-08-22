import pandas as pd

def createDummies(df, colList):
    dummieFrames = []
    for col in colList:
        dummiesFrame = pd.get_dummies(df[col], prefix = col, dtype=int)
        df = df.drop(col, axis = 1)
        dummieFrames.append(dummiesFrame)
    dummieFrames.append(df)
    df = pd.concat(dummieFrames, axis = 1)
    return df

# Find and print row and column labels of null or NaN values
def findNull(df):
    null_indices = df.isnull().stack()
    for row_col, is_null in null_indices.items():
        if is_null:
            row, col = row_col
            print(f"Null value at row {row} and column {col}")