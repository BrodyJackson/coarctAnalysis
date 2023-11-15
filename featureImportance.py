import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
from dataGlossary import demographicFeatures, vascularRisk, currentClinicalFeatures, imagingValues
from sklearn.model_selection import train_test_split
from helpers import createDummies
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np

dummyNeeded = [
    'smoking_status',
    'aortopathies',
    'aortic_valve_morphology',
    'valve_current_condition',
    'valve_current_type',
    'first_op_type',
    'had_one_op_type'
]

dontInclude = [
    'aortic_aneurysm', 
    'aortic_aneurysm_repaired',
    'coarctation_type',
    'cardiopulmonary_exercise_test_performed',
    'indication_for_repair',
    'previous_coarctation_intervention',
    'ecg_sinus_rhythm'
]


def findFeatureImportance(df): 
    y = df['cardiovascular_event']
    X = df.drop(['cardiovascular_event'], axis=1)
    testColumns = demographicFeatures + vascularRisk + currentClinicalFeatures + imagingValues
    testColumns = testColumns + ['age_first_surgery', 'first_op_type', 'had_one_op_type', 'only_catheters', 'only_surgeries']
    for val in dontInclude:
        testColumns.remove(val)

    for val in X.columns: 
        if val not in testColumns:
            X = X.drop([val], axis=1)
    
    X = createDummies(X, dummyNeeded)
    #
    X.drop(list(X.filter(regex = '_([1-9][0-9]*)$')), axis = 1, inplace = True)
    # print(X.loc[:, X.isna().any()].to_string())
    
    #Commented out to see if I can just do before I pass in

    imp = SimpleImputer(strategy='mean')
    for val in imagingValues:
        X[val] = imp.fit_transform(X[val].values.reshape(-1, 1) )

    # scaler = MinMaxScaler()
    # print(X.head())
    # print(X.isna().sum())
    # X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)
    # imputer = KNNImputer(n_neighbors=5)
    # X = pd.DataFrame(imputer.fit_transform(X),columns = X.columns)
    # X = pd.DataFrame(scaler.inverse_transform(X), columns = X.columns)
    # print(X.head())
    # print(X.isna().sum())

    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    cols = X_train.columns
    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = pd.DataFrame(X_train, columns=[cols])
    X_test = pd.DataFrame(X_test, columns=[cols])

    # instantiate the classifier 
    rfc = RandomForestClassifier(n_estimators=600, random_state=0)
    # fit the model
    rfc.fit(X_train, y_train)
    # Predict the Test set results
    y_pred = rfc.predict(X_test)
    # Check accuracy score 
    print('Model accuracy score with 600 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    feature_scores = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    # feature_scores.nlargest(100).plot(kind='barh').invert_yaxis()
    # plt.show()
    # plt.savefig("test.svg", format="svg")
    print(feature_scores)

    # Get variable importances
    importances = rfc.feature_importances_
    # Sort variables based on importance scores
    sorted_indices = np.argsort(importances)[::-1]
    sorted_importances = importances[sorted_indices]
    # Calculate cumulative sum of importance scores
    cumulative_importance = np.cumsum(sorted_importances)
    # Define the desired percentage of total importance (e.g., 90%)
    desired_percentage = 0.90
    # Find the index where cumulative importance crosses the desired percentage
    selected_index = np.argmax(cumulative_importance >= desired_percentage)
    # Select variables based on the determined index
    selected_features = X_train.iloc[:, sorted_indices[:selected_index + 1]]
    if 'age' in selected_features.columns:
        selected_features.rename(columns={'age': 'patient_age'}, inplace=True)
    use_in_cox = [str(column[0]) for column in selected_features.columns]
    print(use_in_cox)
    # Print the selected features
    print("Selected Features:")
    for feature_index in sorted_indices[:selected_index + 1]:
        print(f"Feature {feature_index}: Importance = {importances[feature_index]:.4f}")

    # f, ax = plt.subplots(figsize=(30, 24))
    # ax = sns.barplot(x=feature_scores, y=feature_scores.index, data=df)
    # ax.set_title("Visualize feature scores of the features")
    # ax.set_yticklabels(feature_scores.index)
    # ax.set_xlabel("Feature importance score")
    # ax.set_ylabel("Features")
    # plt.show()

    # Return the features with greatest cumulative importance and the dataset
    return(use_in_cox, X)