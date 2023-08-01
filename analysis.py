import pandas as pd
import numpy as np
from datetime import datetime, date

from dataGlossary import renameColumns
from dataGlossary import cardiovascularEvents as cardioEvent
from dataGlossary import categoricalChoices
from dataGlossary import tableLevels, tableColumns
from dataGlossary import surgeryOperations
from dataGlossary import cathOperations
from descriptiveStats import createTableOne

from survivalAnalysis import generateSurvivalAnalysis

data = 'coarctData.csv'

df = pd.read_csv(data)
# Getting rid of all the columns that just mark an instrument as complete
df.drop(list(df.filter(regex = '_complete')), axis = 1, inplace = True)
# Rename the coexistent lesions to their actual names in the data
df.rename(columns=renameColumns, inplace=True)
# find the completely empty columns
empty_cols = [col for col in df.columns if df[col].isnull().all()]
# print(empty_cols)

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
df['number_of_cath'] = df[list(cathOperations)].count(axis=1)

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

# Add a new column to each row to determine patients age
df['age'] = df['patient_birth_date'].apply(determineAge)

operations = cathOperations.union(surgeryOperations)
def createOperationBool(row, original): 
    if (pd.isnull(row[original])):
        return 'No'
    else: 
        return 'Yes'
for x in operations:
    df[f'{x}_bool'] = df.apply(createOperationBool, original = x, axis = 1)

# Create descriptive statistics table
def createSummaryTable(table, renameColumns):
    summaryTable = createTableOne(df, table)
    summaryTableDf = summaryTable.tableone

    def createProperLabels(table):
        tuplesList = []
        for label in table.index:
            string = label[0].split(',')[0]
            newLabel = categoricalChoices[string][int(float(label[1]))] if string in categoricalChoices else label[1]
            tuplesList.append((label[0], newLabel))
        return tuplesList

    # print(summaryTableDf.columns.names)
    summaryTableDf.index = pd.MultiIndex.from_tuples(createProperLabels(summaryTableDf), names=summaryTableDf.index.names)
    
    if renameColumns:
        summaryTableDf.columns = pd.MultiIndex.from_tuples(tableColumns, names=summaryTableDf.columns.names)
    # rename the columns in df if we want
    # d = dict(zip(summaryTableDf.columns.levels[1], tableLevels))
    # summaryTableDf = summaryTableDf.rename(columns=d, level=1)

    return summaryTable

def filterNoEffect(table):
    unstacked = table.tableone.reset_index()
    onlyEffects = unstacked[~unstacked.level_1.isin(['No', 'Unknown'])]
    unaffected = unstacked[unstacked.level_1.isin(['No', 'Unknown'])]
    for index, row in unaffected.iterrows():
        rowWithP = row['Grouped by cardiovascular_event']['P-Value']
        rowWithPLabel = row['level_0'][0]
        occuranceRow = onlyEffects.loc[[onlyEffects.level_0.eq(rowWithPLabel).idxmax()]]
        if rowWithP and rowWithPLabel == occuranceRow['level_0']:
            # Find the first occurance of matching title and move the P value of removed column onto it
            occuranceRow[P-Value] = row['Grouped by cardiovascular_event']['P-Value']
    
    # print(onlyEffects.to_string())
    table = onlyEffects.set_index(['level_0', 'level_1'], inplace=True)
    return table


# print(summaryTableDf.columns.levels[1])

survivalCurves = generateSurvivalAnalysis(df)
tablesToCreate = ['demographics', 'surgeries', 'outcomes']
for x in tablesToCreate:
    fileName = f"{x}.html"
    table = createSummaryTable(x, True) if x != 'outcomes' else createSummaryTable(x, False)
    # table = filterNoEffect(table)
    table.tableone.to_html(fileName)
    tableHeaders = tableLevels 
    with open(fileName, 'w') as f:
        f.write(table.tabulate(headers=tableHeaders, tablefmt="html"))
    # print(table.tabulate(headers=tableHeaders, tablefmt="latex"))

# print(summaryTableDf.to_string())
# fileName = 'summaryTable.html'
# summaryTableDf.to_html(fileName)
# print(table.tabulate(headers=tableLevels, tablefmt="github"))

