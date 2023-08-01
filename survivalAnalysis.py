import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
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
    df['time_to_event'] = df.apply(lambda x: (x['earliest_event'] - x['earliest_op']).days, axis=1)
    df['time_to_event'] = df.apply(lambda x: 0.5 if x['time_to_event'] == 0 else x['time_to_event'], axis=1)

    dfCleaned = df.loc[df['time_to_event'] >= 0]
    # print(df.loc[107].to_string())
    # print(df['earliest_op'].to_string())
    # print(df['earliest_event'].to_string())
    # print(dfCleaned['time_to_event'].to_string())

    # generateTotalSurvival(dfCleaned)
    generateGroupedSurvival(dfCleaned)
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
       
    groupedInfo = [
        ("hypertension == 1", "Hypertension"),
        ("dyslipidemia == 1", "Dyslipidemia"),
        ("smoking_status == 1 | smoking_status == 2", "Smoking History"),
        ("diabetes == 1", "Diabetes"),
        ("family_premature_cad_hist == 1", "Family Early CAD")
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