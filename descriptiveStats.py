from tableone import TableOne

categoricalValues = [
    'diabetes',
    'smoking_status',
    'family_premature_cad_hist',
    'hypertension',
    'dyslipidemia',
    'claudication_pain',
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
    'claudication_pain_lower_or_upper_left',
    'stent_fracture',
    'surgery_related_complications',
    'death',
    'coronary_air_embolization',
    'stent_embolization',
    'emergent_surgery',
    'periprocedural_stroke',
    'aortic_dissection_post_angio',
    'arrhythmia_requiring_cardioversion',
    'need_for_blood_transfusion',
    'need_for_surgical_vascular_site_repair',
    'cardiovascular_event'
]

numericColumns = [
    'age',
    'height',
    'weight',
    'bmi',
    'sex', 
    'diabetes_latest_a1c',
    'diabetes',
    'smoking_status',
    'family_premature_cad_hist',
    'hypertension',
    'dyslipidemia',
    'claudication_pain',
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
    'claudication_pain_lower_or_upper_left',
    'stent_fracture',
    'surgery_related_complications',
    'death',
    'coronary_air_embolization',
    'stent_embolization',
    'emergent_surgery',
    'periprocedural_stroke',
    'aortic_dissection_post_angio',
    'arrhythmia_requiring_cardioversion',
    'need_for_blood_transfusion',
    'need_for_surgical_vascular_site_repair',
    'cardiovascular_event'   
]
def createTableOne(df):
    groupby = ['cardiovascular_event']
    # nonnormal = []
    # labels={}
    # categorical = []
    # mytable = TableOne(df, columns=columns, categorical=categorical, groupby=groupby, nonnormal=nonnormal, rename=labels, pval=False)
    myTable = TableOne(df, columns=numericColumns, categorical=categoricalValues, groupby=groupby, pval=True)
    return myTable
    