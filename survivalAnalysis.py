import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter, SplineFitter, LogNormalFitter
from datetime import datetime
import pandas as pd


from dataGlossary import outcomeDateColumns
from dataGlossary import operationDateColumns

def generateSurvivalAnalysis (df):
    df["cardiovascular_event"].info()
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

    dfCleaned = df.loc[df['time_to_event'] >= 0]
    # print(df.loc[107].to_string())
    # print(df['earliest_op'].to_string())
    # print(df['earliest_event'].to_string())
    # print(dfCleaned['time_to_event'].to_string())

    generateTotalSurvival(dfCleaned)
    generateGroupedSurvival(dfCleaned)
    generateCoxRegression(dfCleaned)
    plt.show()
    return True

def generateTotalSurvival (df):
    kmf = KaplanMeierFitter()
    naf = NelsonAalenFitter()
    kmf.fit(durations = df['time_to_event'], event_observed = df['cardiovascular_event'])
    print(kmf.event_table)
    print(kmf.survival_function_)
    print(kmf.median_survival_time_)
    plt.figure(1)
    kmf.plot_survival_function()
    plt.title("The Kaplan-Meier Estimate")
    plt.ylabel("Probability of no outcome")

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

    

def generateGroupedSurvival (df):

    def Kaplan(df, timeColumn, eventColumn, info):
        kmf = KaplanMeierFitter()
        for value in info:
            fitFrame = df.query(value[0])
            kmf.fit(durations=fitFrame[timeColumn], event_observed=fitFrame[eventColumn], label=value[1])
            plt.figure(5)
            kmf.plot_survival_function(ci_alpha=0.1)
            plt.title("The Kaplan-Meier Estimate")
            plt.ylabel("Probability of no outcome")
            plt.figure(6)
            kmf.plot_cumulative_density(ci_alpha=0.1)
            plt.title("The Cumulative Density Estimate")
            plt.ylabel("Probability of cv outcome")
        
    def Hazard(df, timeColumn, eventColumn, info):
        naf = NelsonAalenFitter()
        for value in info:
            fitFrame = df.query(value[0])
            naf.fit(durations=fitFrame[timeColumn], event_observed=fitFrame[eventColumn], label=value[1])
            naf.plot_cumulative_hazard(ci_show=False)

    def HazardDiffOutcomes(df, timeColumn):
        naf = NelsonAalenFitter()
        for value in outcomesOfInterest:
            # spf = SplineFitter([0,50,100]).fit(df[timeColumn], df[value], label=value.replace('_', ' '))
            #determine which fitter I shuold use
            # spf = LogNormalFitter().fit(durations=df[timeColumn], event_observed=df[value], label=value.replace('_', ' '))
            naf.fit(durations=df[timeColumn], event_observed=df[value], label=value.replace('_', ' '))
            naf.plot_cumulative_hazard(ci_show=False)
       
    groupedInfo = [
        ("hypertension == 1", "Hypertension"),
        ("dyslipidemia == 1", "Dyslipidemia"),
        ("smoking_status == 1 | smoking_status == 2", "Smoking History"),
        ("diabetes == 1", "Diabetes"),
        ("family_premature_cad_hist == 1", "Family Early CAD")
    ]

    outcomesOfInterest = [
        'presence_of_aneurysm_location',
        'presence_of_aortic_dissection',
        'systemic_hypertension',
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

    Kaplan(df, "time_to_event", "cardiovascular_event", groupedInfo)
    
    plt.figure(7)
    plt.title("Cumulative Hazard by time to event")
    plt.ylabel("Probability of cv outcome")
    Hazard(df, "time_to_event", "cardiovascular_event", groupedInfo)
    
    plt.figure(8)
    plt.title("Cumulative Hazard by age")
    plt.ylabel("Probability of cv outcome")
    Hazard(df, "age", "cardiovascular_event", groupedInfo)
    
    plt.figure(9)
    plt.title("Hazard risk of individual outcome by age")
    plt.ylabel("Probability of cv outcome")
    HazardDiffOutcomes(df, "age")

def generateCoxRegression(df): 
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
        'sex',
        'height',
        'weight',
        'bmi',
        'hypertension',
        'dyslipidemia',
        'smoking_status',
        'diabetes',
        'family_premature_cad_hist',
        'claudication_pain',
        'aortopathies',
        'aortic_valve_morphology',
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
        'number_of_transcatheter_interventions',
        'number_of_open_surgical_interventions',

        'coarctation_less_three_mm',
        'interrupted_aortic_arch',
        'presence_of_collaterals',
        # 'ecg_afib',

        #not working
        'total_num_antihypertensives'
    ]

    needsDummies = [
        'smoking_status',
        'aortopathies',
        'aortic_valve_morphology',
        'indication_for_repair',
        'valve_current_condition',
        'valve_current_type',
        # 'aortic_aneurysm',
    ]

    dfFiltered = df[columnsToInclude]

    for col in categoricalColumns:
        print(col, dfFiltered[col].unique())

    dummieFrames = []
    for col in needsDummies:
        dummiesFrame = pd.get_dummies(dfFiltered[col], prefix = col, dtype=int)
        dfFiltered = dfFiltered.drop(col, axis = 1)
        dummieFrames.append(dummiesFrame)
    dummieFrames.append(dfFiltered)
    dfFiltered = pd.concat(dummieFrames, axis = 1)
    print(dfFiltered.loc[:, dfFiltered.isna().any()].to_string())
    dfFiltered = dfFiltered.dropna()
    # Getting rid of all the columns that indicate no or null
    dfFiltered.drop(list(dfFiltered.filter(regex = '_0$')), axis = 1, inplace = True)
    # print(dfFiltered.head())
    # plt.hist(dfFiltered['time_to_event'], bins = 50)
    # plt.show()
    kmf = KaplanMeierFitter()
    cph = CoxPHFitter(penalizer=0.0001)
    cph.fit(dfFiltered,"time_to_event", event_col="cardiovascular_event", strata=['number_of_open_surgical_interventions', 'family_premature_cad_hist'])
    cph.print_summary()
    plt.figure(10)
    cph.plot()
    # cph.plot_partial_effects_on_outcome(covariates = 'age', values = [30, 40, 50, 60, 70, 80], cmap = 'coolwarm')
    cph.check_assumptions(dfFiltered, p_value_threshold = 0.05)