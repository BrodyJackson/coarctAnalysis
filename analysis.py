import pandas as pd
import numpy as np
from datetime import datetime, date

from dataGlossary import renameColumns
from dataGlossary import cardiovascularEvents as cardioEvent
from dataGlossary import categoricalChoices
from dataGlossary import tableLevels
from dataGlossary import surgeryOperations
from dataGlossary import cathOperations
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
 
# Calculate an age value based on an input date in format YYYY-MM-DD
def determineAge(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

# I need to the total number of sugical repairs and the total number of catheter repairs
# Then I need the total number of each individual type of operation
# Then I need the number of operations per patient
def determineSpecificOperationCounts(keyword, operationList):
    columnList = [col for col in operationList if keyword in operationList]

surgicalOperationNumber = df[list(surgeryOperations)].count().sum()
cathOperationNumber = df[list(cathOperations)].count().sum()

df['number_of_surgeries'] = df[list(surgeryOperations)].count(axis=1)
print(df['number_of_surgeries'].to_string())

# Get the total number of operatioTypes across all a patients surgeries
operationDetails = {
    'unknownSurgNum': determineSpecificOperationCounts('unknown', surgeryOperations),
    'anastamosisNum': determineSpecificOperationCounts('anastamosis', surgeryOperations),
    'angioplastyNum': determineSpecificOperationCounts('angioplasty', surgeryOperations),
    'subclavianNum': determineSpecificOperationCounts('subclavian', surgeryOperations),
    'interpositionNum': determineSpecificOperationCounts('interposition', surgeryOperations),
    'unknownCathNum': determineSpecificOperationCounts('unknown', cathOperations),
    'balloonCathNum': determineSpecificOperationCounts('balloon', cathOperations),
    'coveredCathNum': determineSpecificOperationCounts('covered', cathOperations),
    'bareCathNum': determineSpecificOperationCounts('bare', cathOperations),
    'endoCathNum': determineSpecificOperationCounts('endovascular', cathOperations),
    'hybridCathNum': determineSpecificOperationCounts('hybrid', cathOperations)
}



print(surgicalOperationNumber)
print(cathOperationNumber)



# df[]


# Add a new column to each row to determine patients age
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

# rename the columns in df if we want
# d = dict(zip(summaryTableDf.columns.levels[1], tableLevels))
# summaryTableDf = summaryTableDf.rename(columns=d, level=1)
# print(summaryTableDf.columns.levels[1])

# print(summaryTableDf.to_string())
# fileName = 'summaryTable.html'
# summaryTableDf.to_html(fileName)
# print(summaryTable.tabulate(headers=tableLevels, tablefmt="github"))

