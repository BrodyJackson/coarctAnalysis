import pandas as pd
import numpy as np
from datetime import datetime, date

from dataGlossary import renameColumns
from dataGlossary import cardiovascularEvents as cardioEvent
from dataGlossary import categoricalChoices
from dataGlossary import tableColumns
from dataGlossary import tableLevels
from descriptiveStats import createTableOne

data = 'coarctData.csv'

df = pd.read_csv(data)
# Getting rid of all the columns that just mark an instrument as complete
df.drop(list(df.filter(regex = '_complete')), axis = 1, inplace = True)
# Rename the coexistent lesions to their actual names in the data
df.rename(columns=renameColumns, inplace=True)

# Add a new column to each row to determine if that patient had a negative cardiovascular outcome
# Negative outcome columns are determined in the data glossary file
def hadEvent(row):
    for col in cardioEvent:
        if (row[col] != 0):
            return 1
        else: 
            continue
    return 0

df['cardiovascular_event'] = df.apply(hadEvent, axis = 1)
 
# Add a new column to each row to determine patients age
def determineAge(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

df['age'] = df['patient_birth_date'].apply(determineAge)

# Create descriptive statistics table
summaryTable = createTableOne(df)
summaryTableDf = summaryTable.tableone

def createProperLabels(table):
    tuplesList = []
    for label in table.index:
        string = label[0].split(',')[0]
        newLabel = categoricalChoices[string][int(label[1])] if string in categoricalChoices else label[1]
        tuplesList.append((label[0], newLabel))
    # print(tuplesList) 
    return tuplesList
print(summaryTableDf.columns.names)
summaryTableDf.index = pd.MultiIndex.from_tuples(createProperLabels(summaryTableDf), names=summaryTableDf.index.names)


summaryTableDf.columns = pd.MultiIndex.from_tuples(tableColumns, names=summaryTableDf.columns.names)



print(summaryTableDf.columns.levels)
print(summaryTableDf.to_string())
fileName = 'summaryTable.html'
summaryTableDf.to_html(fileName)
print(summaryTable.tabulate(tablefmt="html"))

