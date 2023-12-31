import pandas as pd
import numpy as np
from datetime import datetime, date
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from dataGlossary import renameColumns
from dataGlossary import cardiovascularEvents as cardioEvent
from dataGlossary import categoricalChoices
from dataGlossary import tableLevels, tableColumns
from dataGlossary import surgeryOperations
from dataGlossary import cathOperations
from dataGlossary import hiddenAttributes
from dataGlossary import needsAZeroValue
from dataGlossary import majorCardioEvent
from dataGlossary import operationDateColumns
from descriptiveStats import createTableOne

from survivalAnalysis import generateSurvivalAnalysis

data = 'coarctData_imaging.csv'

df = pd.read_csv(data)
# Getting rid of all the columns that just mark an instrument as complete
df.drop(list(df.filter(regex = '_complete$')), axis = 1, inplace = True)
# Rename the coexistent lesions to their actual names in the data
df.rename(columns=renameColumns, inplace=True)
# find the completely empty columns
empty_cols = [col for col in df.columns if df[col].isnull().all()]

df['number_of_surgeries'] = df[list(surgeryOperations)].count(axis=1)
df['number_of_cath'] = df[list(cathOperations)].count(axis=1)

# Add a new column to each row to determine if that patient had exclusively sugeries, or catheter, or vice versa
# 0 means they had combo, 1 means they had only surgeries, 2 means they had only catheters
def hadOnlyOneOpType(row):
    if row["number_of_cath"] == 0 and row["number_of_surgeries"] >= 1:
        return 1
    elif row["number_of_cath"] >= 1 and row["number_of_surgeries"] == 0:
        return 2
    return 0

df['had_one_op_type'] = df.apply(hadOnlyOneOpType, axis = 1)
df['only_catheters'] = df.apply(lambda x: 1 if x['had_one_op_type'] == 2 else 0, axis = 1)
df['only_surgeries'] = df.apply(lambda x: 1 if x['had_one_op_type'] == 1 else 0, axis = 1)

#Filter out the attributes that need to be hidden
for choice in hiddenAttributes:
    if choice in needsAZeroValue:
        df[choice] = df[choice].apply(lambda x: int(x + 1) if not pd.isnull(x) and x != 9 else np.nan)
    df[choice] = df[choice].fillna(0)
    df[choice] = df[choice].astype(int)

#Combine unknown and no values for categorical variables into one value
for choice in categoricalChoices:
    if 9 in categoricalChoices[choice].keys():
        # df[choice][df[choice] == 9] = 0
        df[choice] = df[choice].replace(9,0)
        # df[choice] = df.apply(lambda x: 0 if x == 9 else x)


# Add a new column to each row to determine if that patient had a negative cardiovascular outcome
# Negative outcome columns are determined in the data glossary file
def hadEvent(row):
    for col in majorCardioEvent: #Just swapped to major temporarily. Swap back if need all outcomes
        if (row[col] != 0):
            return 1
        else: 
            continue
    return 0

df['cardiovascular_event'] = df.apply(hadEvent, axis = 1)

# Add a new column to each row to determine if that patient had a negative cardiovascular outcome
# Negative outcome columns are determined in the data glossary file
def hadMajorEvent(row):
    for col in majorCardioEvent:
        if (row[col] != 0):
            return 1
        else: 
            continue
    return 0

df['major_cardiovascular_event'] = df.apply(hadMajorEvent, axis = 1)

# Create resistive hypertension column
def resistiveHypertension(row):
    if row['total_num_antihypertensives'] >= 3 and row['systemic_hypertension'] > 0:
        return 1
    return 0
    
df['resistive_hypertension'] = df.apply(resistiveHypertension, axis = 1)

# Create coarctation ratio from imaging column
df['imaging_coarct_ratio'] = df.apply(lambda row: row['diameter_at_coarct_site_max']/row['diameter_at_diaphragm_max'], axis=1)

print('counts for imaging')
print(df['diameter_at_coarct_site_max'].count())
print(df['diameter_at_widest_ascending_aorta_max'].count())
print(df['diameter_at_post_stenotic_site_max'].count())
print(df['diameter_at_diaphragm_max'].count())

# Calculate an age value based on an input date in format YYYY-MM-DD
def determineAge(born):
    born = datetime.strptime(born, "%Y-%m-%d").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

# Total number of sugical repairs and the total number of catheter repairs
# The total number of each individual type of operation
# The number of operations per patient
def determineSpecificOperationCounts(keyword, operationList):
    columnList = [col for col in operationList if keyword in operationList]

surgicalOperationNumber = df[list(surgeryOperations)].count().sum()
cathOperationNumber = df[list(cathOperations)].count().sum()

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

#Determine if they had a sugery of catheter first (make two columns of yes/no)
df['first_op_type'] = 0
for index, row in df.iterrows():
    earliestOpDate = datetime.today().date()
    for column in operationDateColumns:
        if pd.isna(row[column]):
            continue
        date = datetime.strptime(str(row[column]), '%Y-%m-%d %H:%M').date()
        if  date < earliestOpDate:
            earliestOpDate = date
            df.loc[index, 'first_op_type'] = 1 if column in surgeryOperations else 2
    
operations = cathOperations.union(surgeryOperations)
def createOperationBool(row, original): 
    if (pd.isnull(row[original])):
        return 'No'
    else: 
        return 'Yes'
for x in operations:
    df[f'{x}_bool'] = df.apply(createOperationBool, original = x, axis = 1)

fillNa = ['number_of_transcatheter_interventions', 'number_of_open_surgical_interventions', 'total_num_antihypertensives']
fillNaMean = ['height', 'weight', 'bmi']
for x in fillNa:
    df[x] = df[x].fillna(0)
for x in fillNaMean:
    df[x].fillna((df[x].mean()), inplace=True)

df['smoking_status'] = df['smoking_status'].replace(2,1)

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
noEffectList = ['No', 'Unknown', 'Never smoked', 'Normal', 'No (Normal)', 'No aneurysm']
def filterNoEffect(table, tableName):
    unstacked = table.tableone.reset_index()
    onlyEffects = unstacked[~unstacked.level_1.isin(noEffectList)]
    unaffected = unstacked[unstacked.level_1.isin(noEffectList)]
    if tableName != 'outcomes':
        for index, row in unaffected.iterrows():
            rowWithP = row['Grouped by cardiovascular_event']['P-Value']
            rowWithPLabel = row['level_0'][0]
            occuranceRow = onlyEffects.loc[[onlyEffects.level_0.eq(rowWithPLabel).idxmax()]]
            if rowWithP and rowWithPLabel == occuranceRow['level_0'].values[0]:
                # Find the first occurance of matching title and move the P value of removed column onto it
                location = occuranceRow.index[0]
                onlyEffects.loc[location, ('Grouped by cardiovascular_event', 'P-Value')] = row['Grouped by cardiovascular_event']['P-Value']
                occuranceRow['P-Value'] = row['Grouped by cardiovascular_event']['P-Value']
    
    onlyEffects.set_index(['level_0', 'level_1'], inplace=True)
    return onlyEffects

survivalCurves = generateSurvivalAnalysis(df)
tablesToCreate = ['demographics', 'surgeries', 'outcomes']
for x in tablesToCreate:
    fileName = f"{x}.html"
    table = createSummaryTable(x, True) if x != 'outcomes' else createSummaryTable(x, False)
    table.tableone = filterNoEffect(table, x)
    # table.tableone.to_html(fileName)
    tableHeaders = tableLevels if x != 'outcomes' else None
    with open(fileName, 'w') as f:
        f.write(table.tabulate(headers=tableHeaders, tablefmt="html"))
    print(table.tabulate(headers=tableHeaders, tablefmt="github"))

plt.show()
# print(summaryTableDf.to_string())
# fileName = 'summaryTable.html'
# summaryTableDf.to_html(fileName)
# print(table.tabulate(headers=tableLevels, tablefmt="github"))

