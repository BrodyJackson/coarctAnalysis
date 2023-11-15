import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter, SplineFitter, LogNormalFitter
from lifelines.statistics import pairwise_logrank_test
from statsmodels.stats.proportion import proportion_confint
from datetime import datetime
import pandas as pd
import numpy as np
import re
import pprint


from dataGlossary import outcomeDateColumns
from dataGlossary import operationDateColumns
from dataGlossary import imagingValues
from featureImportance import findFeatureImportance
from helpers import createDummies, findNull

def generateSurvivalAnalysis (df):
    print(df.head())
    def findEarliestOp(row):
        earliestOpDate = datetime.today().date()
        for column in operationDateColumns:
            if pd.isna(row[column]):
                continue
            date = datetime.strptime(str(row[column]), '%Y-%m-%d %H:%M').date()
            if  date < earliestOpDate:
                earliestOpDate = date
        return earliestOpDate
    
    def findEarliestEvent(row):
        earliestEvent = datetime.today().date()
        for column in outcomeDateColumns:
            if pd.isna(row[column]):
                continue
            date = datetime.strptime(str(row[column]), '%Y-%m-%d %H:%M').date()
            if  date < earliestEvent:
                earliestEvent = date
        return earliestEvent
    
    df['earliest_op'] = df.apply(findEarliestOp, axis=1)
    df['earliest_event'] = df.apply(findEarliestEvent, axis=1)
    df['time_to_event'] = df.apply(lambda x: (x['earliest_event'] - x['earliest_op']).days / 365.25, axis=1)
    df['time_to_event'] = df.apply(lambda x: 0.5 if x['time_to_event'] == 0 else x['time_to_event'], axis=1)
    df['age_first_surgery'] = df.apply(lambda x: (x['earliest_op'] - datetime.strptime(x['patient_birth_date'], "%Y-%m-%d").date()).days / 365.25, axis=1)

    dfCleaned = df.loc[df['time_to_event'] >= 0]

    generateTotalSurvival(dfCleaned)
    generateGroupedSurvival(dfCleaned)
    plt.show(block=False)
    importantFeatures, X = findFeatureImportance(dfCleaned)
    for imaging in imagingValues:
        dfCleaned[imaging] = X[imaging]
    generateCoxRegression(dfCleaned, importantFeatures)
 
    return True

def generateTotalSurvival (df):
    kmf = KaplanMeierFitter()
    naf = NelsonAalenFitter()
    kmf.fit(durations = df['time_to_event'], event_observed = df['cardiovascular_event'])
    print(kmf.event_table)
    print(kmf.survival_function_)
    print('median survival', kmf.median_survival_time_)
    plt.figure(1)
    kmf.plot_survival_function()
    plt.title("The Kaplan-Meier Estimate")
    plt.ylabel("Freedom from cardiovascular event")

    print(kmf.cumulative_density_)
    plt.figure(2)
    kmf.plot_cumulative_density()
    plt.title("The Cumulative Density Estimate")
    plt.ylabel("Probability of cv outcome")

    naf.fit(df["time_to_event"],event_observed = df["cardiovascular_event"])
    plt.figure(3)
    naf.plot_cumulative_hazard()
    plt.title("Cumulative Hazard by time to event")
    plt.ylabel("Probability of cv outcome")

    naf.fit(df["age"],event_observed = df["cardiovascular_event"])
    print (naf.cumulative_hazard_)
    plt.figure(4)
    naf.plot_cumulative_hazard()
    plt.title("Cumulative Hazard by age")
    plt.ylabel("Probability of cv outcome")

    
def Kaplan(df, timeColumn, eventColumn, info, plotNum):
    kmf = KaplanMeierFitter()
    for value in info:
        fitFrame = df.query(value[0])
        kmf.fit(durations=fitFrame[timeColumn], event_observed=fitFrame[eventColumn], label=value[1])
        plt.figure(plotNum[0])
        kmf.plot_survival_function(ci_alpha=0.5)
        plt.title("The Kaplan-Meier Estimate")
        plt.ylabel("Freedom from cardiovascular event")
        plt.figure(plotNum[1])
        kmf.plot_cumulative_density(ci_alpha=0.5)
        plt.title("The Cumulative Density Estimate")
        plt.ylabel("Probability of cv outcome")

def Hazard(df, timeColumn, eventColumn, info):
    naf = NelsonAalenFitter()
    for value in info:
        fitFrame = df.query(value[0])
        naf.fit(durations=fitFrame[timeColumn], event_observed=fitFrame[eventColumn], label=value[1])
        naf.plot_cumulative_hazard(ci_show=False)

def HazardDiffOutcomes(df, timeColumn, outcomesOfInterest):
    naf = NelsonAalenFitter()
    for value in outcomesOfInterest:
        # spf = SplineFitter([0,50,100]).fit(df[timeColumn], df[value], label=value.replace('_', ' '))
        #determine which fitter I shuold use
        # spf = LogNormalFitter().fit(durations=df[timeColumn], event_observed=df[value], label=value.replace('_', ' '))
        naf.fit(durations=df[timeColumn], event_observed=df[value], label=value.replace('_', ' '))
        naf.plot_cumulative_hazard(ci_show=False)

def generateGroupedSurvival (df):
       
    groupedInfo = [
        ("resistive_hypertension == 1", "Resistive Hypertension"),
        ("dyslipidemia == 1", "Dyslipidemia"),
        ("smoking_status == 1 | smoking_status == 2", "Smoking History"),
        ("diabetes == 1", "Diabetes"),
        ("family_premature_cad_hist == 1", "Family Early CAD")
    ]

    surgeryGroups = [
        ("had_one_op_type == 0", "Received both operation types"),
        ("had_one_op_type == 1", "Received only surgeries"),
        ("had_one_op_type == 2", "Received only catheterizations"),
    ]

    outcomesOfInterest = [
        'presence_of_aneurysm_location',
        'presence_of_aortic_dissection',
        'resistive_hypertension',
        'renal_failure',
        'heart_failure',
        'pulmonary_hypertension',
        'femoral_artery_occlusion',
        'coronary_artery_disease',
        'myocardial_infarction',
        'stroke',
        'intracranial_aneurysm',
        'infective_endocarditis',
        'death',
    ]

    Kaplan(df, "time_to_event", "cardiovascular_event", groupedInfo, (5,6))
    Kaplan(df, "time_to_event", "cardiovascular_event", surgeryGroups, (7,8))
    
    plt.figure(9)
    plt.title("Cumulative Hazard by time to event")
    plt.ylabel("Probability of cv outcome")
    Hazard(df, "time_to_event", "cardiovascular_event", groupedInfo)
    
    plt.figure(10)
    plt.title("Cumulative Hazard by age")
    plt.ylabel("Probability of cv outcome")
    Hazard(df, "age", "cardiovascular_event", groupedInfo)
    
    plt.figure(11)
    plt.title("Cumulative Hazard by time to event grouped by patient surgery")
    plt.ylabel("Probability of cv outcome")
    Hazard(df, "time_to_event", "cardiovascular_event", surgeryGroups)

    plt.figure(12)
    plt.title("Cumulative Hazard by time to event grouped by patient surgery by age")
    plt.ylabel("Probability of cv outcome")
    Hazard(df, "age", "cardiovascular_event", surgeryGroups)

    plt.figure(13)
    plt.title("Hazard risk of individual outcome by age")
    plt.ylabel("Probability of cv outcome")
    HazardDiffOutcomes(df, "age", outcomesOfInterest)

def generateCoxRegression(df, importantFeatures): 
    categoricalColumns = [
        'sex',
        'hypertension',
        'dyslipidemia',
        'smoking_status',
        'diabetes',
        'family_premature_cad_hist',
        'claudication_pain',
        # 'aortopathies',
        # 'aortic_valve_morphology',
        # 'indication_for_repair',
        'valve_current_condition',
        'valve_current_type',
        # 'aortic_aneurysm',
    ]

    columnsToInclude = [
        'time_to_event',
        'cardiovascular_event',
        'age',
        'age_first_surgery',
        'sex',
        # 'height',
        # 'weight',
        'bmi',
        'resistive_hypertension',
        'dyslipidemia',
        'smoking_status',
        'diabetes',
        'family_premature_cad_hist',
        'claudication_pain',
        'aortopathies',
        # 'aortic_valve_morphology',
        'indication_for_repair',
        'valve_current_condition',
        'valve_current_type',
        # 'aortic_aneurysm',

         #not working
        # 'aortic_aneurysm_repaired',
        # 'current_coarctation_present',
        # 'coarctation_type',

        # 'previous_coarctation_intervention',

         #not working
        'first_op_type',
        'number_of_transcatheter_interventions',
        'number_of_open_surgical_interventions',

        'coarctation_less_three_mm',
        'interrupted_aortic_arch',
        'presence_of_collaterals',
        # 'ecg_afib',

        #New imaging columns
        'diameter_at_widest_ascending_aorta_max',
        'diameter_at_coarct_site_max',
        'diameter_at_post_stenotic_site_max',
        'diameter_at_diaphragm_max',
        'imaging_coarct_ratio'
    ]

    needsDummies = [
        'smoking_status',
        'aortopathies',
        # 'aortic_valve_morphology',
        'indication_for_repair',
        'valve_current_condition',
        'valve_current_type',
        'first_op_type'
        # 'aortic_aneurysm',
    ]

    dfFiltered = df[columnsToInclude]

    dfFiltered = createDummies(dfFiltered, needsDummies)
    
    dfFiltered = dfFiltered.dropna()
    dfFiltered.rename(columns={'age': 'patient_age'}, inplace=True)
    # Getting rid of all the columns that indicate no or null
    # dfFiltered.drop(list(dfFiltered.filter(regex = '_0$')), axis = 1, inplace = True)
   
    # Only include the columns in the model that were determined as important by random forest importance levels
    dependentCols = ['time_to_event', 'cardiovascular_event']
    numericBins = {
        'age_first_surgery': [0, 5, 18, 30, 45, 60, 100],
        'number_of_transcatheter_interventions': [0, 1, 2, 3],
        'number_of_open_surgical_interventions': [0, 1, 2, 3],
        # 'patient_age': [0, 25, 35, 45, 65, 100],
        'bmi': [0, 18.5, 25, 30, 100]
    }
    quantileBins = {
        'patient_age': [0, .25, .5, .75, 1],
        'diameter_at_widest_ascending_aorta_max': [0, .25, .5, .75, 1],
        'diameter_at_coarct_site_max': [0, .25, .5, .75, 1],
        'diameter_at_post_stenotic_site_max': [0, .25, .5, .75, 1],
        'diameter_at_diaphragm_max': [0, .25, .5, .75, 1],
        'imaging_coarct_ratio': [0, .25, .5, .75, 1]
        # 'bmi': [0, .2, .4, .6, .8, 1]
    }
    
    importantFeatures = importantFeatures + dependentCols
    # Create dataframes for doing cox regressions between the sub covariates
    subRegressions = []
    for feat in importantFeatures:
        if feat in dependentCols:
            continue
        regex = r"_(\d+)$"
        substring  = re.sub(regex, "", feat) 
        subvariate_columns = [column for column in dfFiltered.columns if substring in column]
        subvariate_columns = subvariate_columns + dependentCols
        filtered_df = dfFiltered.loc[:, subvariate_columns]
        binsNeeded = numericBins | quantileBins
        if feat in binsNeeded.keys():
            if feat in numericBins.keys():
               filtered_df['binned'] = pd.cut(filtered_df[feat], bins=numericBins[feat], labels=False)
            elif feat in quantileBins.keys():
                res, bins = pd.qcut(filtered_df[feat], q=quantileBins[feat], labels=False, retbins=True)
                filtered_df['binned'] = res
                print('boundaries for', feat, bins)
            grouped_df = pd.get_dummies(filtered_df['binned'], prefix=f'{feat}_bin', dtype=int)
            filtered_df = pd.concat([filtered_df, grouped_df], axis=1)
            filtered_df = filtered_df.drop([feat], axis=1)
            filtered_df = filtered_df.drop(['binned'], axis=1)
            findNull(filtered_df)
            # Add the new binned columns into the main df as well
            newColumns = [x for x in filtered_df.columns if x not in dfFiltered.columns]
            newFrame = filtered_df[newColumns]
            dfFiltered = pd.concat([dfFiltered, newFrame], axis=1)
        subRegressions.append(filtered_df)

    # Turn hazards for a fittex cox model into percentage of importance for each covariate
    def percentageImportance(hazardRatios): 
        totalHazardMagnitude = np.sum(hazardRatios.values)
        covariateInfo = {}
        for key in hazardRatios.keys():
            covariateInfo[key] = {
                'Hazard Ratio': hazardRatios[key],
                'Percentage Importance': (hazardRatios[key] / totalHazardMagnitude) * 100
            }
        return covariateInfo
    
    dfAll = dfFiltered.copy()
    dfFiltered = dfFiltered.loc[:, dfFiltered.columns.isin(importantFeatures)]
    # plt.hist(dfFiltered['time_to_event'], bins = 50)
    # plt.show()
    kmf = KaplanMeierFitter()
    cph = CoxPHFitter(penalizer=0.001)
    cph.fit(dfFiltered,"time_to_event", event_col="cardiovascular_event")
    cph.print_summary()
    # Get recipricals of negative events so we are taking into account the INCREASE in risk to get a proper scaling factor
    turnIntoPositiveHazard = ['smoking_status_0', 'aortopathies_0', 'valve_current_type_0'] #'aortic_valve_morphology_0',]
    hazards = cph.hazard_ratios_
    for item in turnIntoPositiveHazard:
        if item in hazards:
            hazards[item] = 1.0 / hazards[item]
    # Calculate the importance of each main covariate in the main cox regression
    mainCovariateInfo = percentageImportance(hazards)
    # plt.figure(10)
    # cph.plot()
    # cph.plot_partial_effects_on_outcome(covariates = 'age_first_surgery', values = [0, 1, 5, 10, 20, 30, 50], cmap = 'coolwarm')
    cph.check_assumptions(dfFiltered, p_value_threshold = 0.05)

    # Run cox regression within the groups
    subCovariateInfo = []
    for frame in subRegressions:
        cph.fit(frame,"time_to_event", event_col="cardiovascular_event")
        cph.print_summary()
        # Get the relative importance values for each nested covariate and push into list
        covariateInfo = percentageImportance(cph.hazard_ratios_)
        subCovariateInfo.append(covariateInfo)
    
    subCovariateInfo = [x for x in subCovariateInfo if x != {}]


    # dictionary where each key is column name and it holds a tuple of percentage importance, and risk points
    finalGrading = {}
        #yuck
    for mainVariate in mainCovariateInfo:
        for value in subCovariateInfo:
            for subvariate in value:
                # Check if the variates or the same, or the main variate string is a substring
                if mainVariate == subvariate or mainVariate in subvariate or mainVariate in value:
                    ##do the multiplication schtuff 
                    scaledPercentage = (value[subvariate]['Percentage Importance'] / 100) * (mainCovariateInfo[mainVariate]['Percentage Importance'] / 100)
                    pointScaleFactor = 100
                    scaledPoints = scaledPercentage * pointScaleFactor
                    print('found a match of', mainVariate, ' and ', subvariate, ' scaled this became ', scaledPoints)
                    finalGrading[subvariate] = (scaledPercentage, scaledPoints)
    pprint.pprint(finalGrading)

    # Begin section to scale the numbers to sensible value (whole numbers between min and max)
    # Calculate the minimum and maximum second values in the tuples
    min_value = min(value[1] for value in finalGrading.values())
    max_value = max(value[1] for value in finalGrading.values())
    # Define the desired range for integers
    desired_min = 0  
    desired_max = 30
    # Calculate the range (R) of the second values
    range_value = max_value - min_value
    # Create a new dictionary to store the transformed values
    sensibleScaledDict = {}
    # Apply the scale transformation and store the results in the new dictionary
    for key, value in finalGrading.items():
        original_score = value[1]
        transformed_score = int(round((original_score - min_value) / range_value * (desired_max - desired_min + 1))) + desired_min
        sensibleScaledDict[key] = transformed_score
    # Print the transformed dictionary
    pprint.pprint(sensibleScaledDict)

    def performBootstrapping():
        # Set the number of bootstrap iterations
        num_bootstrap = 1
            # Initialize arrays to store bootstrapped p-values
        bootstrap_p_values = []
        print("Performing Bootstrapping with 1000 samples, this may take some time...")
        # Perform bootstrapping
        for i in range(num_bootstrap):
            # Generate bootstrap sample
            bootstrap_sample = dfAll.sample(len(dfAll), replace=True)
            # Apply risk point system to bootstrap sample and categorize into high and low-risk
            bootstrap_sample = generateRiskGroup(bootstrap_sample)
            # Perform survival analysis (log-rank test) on bootstrap sample
            log_rank = pairwise_logrank_test(bootstrap_sample['time_to_event'], bootstrap_sample['risk_group'], bootstrap_sample['cardiovascular_event'])
            # Store p-value nested array from bootstrapped sample
            bootstrap_p_values.append(log_rank.p_value)

        # Calculate summary statistics from the bootstrapped p-values
        for i in range(len(bootstrap_p_values[0])):
            elements = [sub_array[i] for sub_array in bootstrap_p_values]
            median_p_value = np.median(elements)
            confidence_interval = np.percentile(elements, [2.5, 97.5])
            
            # Count the number of p-values below 0.05
            count_below_threshold = sum(p_value < 0.05 for p_value in elements)
            # Calculate the proportion of p-values below 0.05
            proportion_below_threshold = count_below_threshold / len(elements)
            print('proportion below threshold', proportion_below_threshold)
            # Calculate the confidence interval for the proportion
            conf_interval = proportion_confint(proportion_below_threshold, len(elements), alpha=0.05, method='normal')
            # Check if the confidence interval is entirely below 0.05
            is_significant = conf_interval[1] < 0.05
    
            # Print results
            print(f"Median p-value for pairwise {i}:", median_p_value)
            print(f"95% Confidence Interval for pairwise {i}:", confidence_interval)
            print('conf interval for whole proportion was ', conf_interval[1])
            print(f"The overall significance of this pair was: {is_significant}")

    
    def calculateTotalPoints(row):
        pointTotal = 0
        for riskFactor in finalGrading:
            if row[riskFactor]:
                if type(finalGrading[riskFactor]) is tuple:
                    pointTotal = pointTotal + finalGrading[riskFactor][1]
                else:
                    pointTotal = pointTotal + finalGrading[riskFactor]
        return pointTotal
            
    def generateRiskGroup(frame):
        frame['risk_points'] = frame.apply(calculateTotalPoints, axis = 1)
        binLabels = ['Low Risk', 'Medium Risk', 'Medium Risk', 'Medium Risk', 'High Risk']
        res, bins = pd.qcut(frame['risk_points'], q=5, labels=False, retbins=True)
        frame['risk_group'] = np.array(binLabels)[res]
        print('bins were ', bins)
        # dfAll['risk_group'] = pd.qcut(dfAll['risk_points'], q=6, labels=binLabels)
        return frame

    dfAll = generateRiskGroup(dfAll)

    groupedInfo = [
        ("risk_group == 'Low Risk'", "Low Risk"),
        # ("risk_group == 'Medium Risk'", "Medium Risk"),
        ("risk_group == 'High Risk'", "High Risk"),
    ]

    # Perform bootstrapping with the full, non-transformed grading scale
    Kaplan(dfAll, "time_to_event", "cardiovascular_event", groupedInfo, (14,15))

    log_rank = pairwise_logrank_test(dfAll['time_to_event'], dfAll['risk_group'], dfAll['cardiovascular_event'])

    print("Log-Rank Test:")
    print(log_rank.summary)

    performBootstrapping()

    # Perform bootstrapping with the sensible graded scale
    finalGrading = sensibleScaledDict
    nullScores = ['smoking_status_0', 'number_of_open_surgical_interventions_bin_0.0','number_of_transcatheter_interventions_bin_0.0','aortopathies_0'] #'aortic_valve_morphology_0'
    for val in nullScores:
        del finalGrading[val]
    dfAll = generateRiskGroup(dfAll)
    # Generate graphs for the sensible scaling
    Kaplan(dfAll, "time_to_event", "cardiovascular_event", groupedInfo, (16,17))

    log_rank = pairwise_logrank_test(dfAll['time_to_event'], dfAll['risk_group'], dfAll['cardiovascular_event'])

    print("Log-Rank Test for scaled values:")
    print(log_rank.summary)
    performBootstrapping()