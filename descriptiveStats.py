from tableone import TableOne
import scipy.stats as stats

categoricalValuesDemographics = [
    'sex',
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
    'aortic_aneurysm',
    'aortic_aneurysm_repaired',
    'current_coarctation_present',
    'coarctation_type',
    'previous_coarctation_intervention',
    'had_one_op_type',
    'first_op_type',
    'coarctation_less_three_mm',
    'interrupted_aortic_arch',
    'presence_of_collaterals',
    'ecg_sinus_rhythm',
    'ecg_afib',
    'cardiopulmonary_exercise_test_performed',
    'beta_blockers',
    'calcium_channel_blockers',
    'ace_inhibitors_arbs',
    'asa',
    'statins',
    'diuretics',
]
    
allColumnsDemographics = [
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
    'aortic_aneurysm',
    'aortic_aneurysm_repaired',
    'current_coarctation_present',
    'coarctation_type',
    'previous_coarctation_intervention',
    'coarctation_less_three_mm',
    'interrupted_aortic_arch',
    'presence_of_collaterals',
    'ecg_sinus_rhythm',
    'ecg_afib',
    'cardiopulmonary_exercise_test_performed',
    'had_one_op_type',
    'first_op_type',
    'number_of_surgeries',
    'number_of_cath',
    'beta_blockers',
    'calcium_channel_blockers',
    'ace_inhibitors_arbs',
    'asa',
    'statins',
    'diuretics',
    'total_num_antihypertensives',
    'diameter_at_widest_ascending_aorta_max',
    'diameter_at_coarct_site_max',
    'diameter_at_post_stenotic_site_max',
    'diameter_at_diaphragm_max',
    'imaging_coarct_ratio'
]

categoricalValuesSurgeries = [
    'index_surgical_procedure_performed',
    'indication_for_first_surgery',
    'unknown_operation_first_bool',
    'resection_end_to_end_anastamosis_index_bool',
    'patch_angioplasty_date_index_bool',
    'patch_angioplasty_material_index',
    'subclavian_flap_angioplasty_index_bool',
    'interposition_graft_index_bool',
    'second_surgical_procedure_performed',
    'indication_for_second_surgery',
    'unknown_operation_second_bool',
    'resection_end_to_end_anastamosis_second_bool',
    'patch_angioplasty_date_second_bool',
    'patch_angioplasty_material_second',
    'subclavian_flap_angioplasty_second_bool',
    'interposition_graft_second_bool',
    'third_surgical_procedure_performed',
    'indication_for_third_surgery',
    'unknown_operation_third_bool',
    'resection_end_to_end_anastamosis_third_bool',
    'patch_angioplasty_date_third_bool',
    'patch_angioplasty_material_third',
    'subclavian_flap_angioplasty_third_bool',
    'interposition_graft_third_bool',
    'index_cath_procedure_performed',
    'indication_for_first_cath',
    'unknown_cath_operation_first_bool',
    'index_balloon_angioplasty_date_bool',
    'index_angioplasty_with_covered_stent_date_bool',
    'index_angioplasty_with_bare_metal_date_bool',
    'index_thoracic_endovascular_aneurysm_repair_date_bool',
    'index_hybrid_procedures_date_bool',
    'second_cath_procedure_performed',
    'indication_for_second_cath',
    'unknown_cath_operation_second_bool',
    'second_balloon_angioplasty_date_bool',
    'second_angioplasty_with_covered_stent_date_bool',
    'second_angioplasty_with_bare_metal_date_bool',
    'second_thoracic_endovascular_aneurysm_repair_date_bool',
    'second_hybrid_procedures_date_bool',
    'third_cath_procedure_performed',
    'indication_for_third_cath',
    'unknown_cath_operation_third_bool',
    'third_balloon_angioplasty_date_bool',
    'third_angioplasty_with_covered_stent_date_bool',
    'third_angioplasty_with_bare_metal_date_bool',
    'third_thoracic_endovascular_aneurysm_repair_date_bool',
    'third_hybrid_procedures_date_bool'
]

allColumnsSurgeries = [
    'number_of_transcatheter_interventions',
    'number_of_open_surgical_interventions',
    'index_surgical_procedure_performed',
    'indication_for_first_surgery',
    'unknown_operation_first_bool',
    'resection_end_to_end_anastamosis_index_bool',
    'patch_angioplasty_date_index_bool',
    'patch_angioplasty_material_index',
    'subclavian_flap_angioplasty_index_bool',
    'interposition_graft_index_bool',
    'second_surgical_procedure_performed',
    'indication_for_second_surgery',
    'unknown_operation_second_bool',
    'resection_end_to_end_anastamosis_second_bool',
    'patch_angioplasty_date_second_bool',
    'patch_angioplasty_material_second',
    'subclavian_flap_angioplasty_second_bool',
    'interposition_graft_second_bool',
    'third_surgical_procedure_performed',
    'indication_for_third_surgery',
    'unknown_operation_third_bool',
    'resection_end_to_end_anastamosis_third_bool',
    'patch_angioplasty_date_third_bool',
    'patch_angioplasty_material_third',
    'subclavian_flap_angioplasty_third_bool',
    'interposition_graft_third_bool',
    'index_cath_procedure_performed',
    'indication_for_first_cath',
    'unknown_cath_operation_first_bool',
    'index_balloon_angioplasty_date_bool',
    'index_angioplasty_with_covered_stent_date_bool',
    'index_angioplasty_with_bare_metal_date_bool',
    'index_thoracic_endovascular_aneurysm_repair_date_bool',
    'index_hybrid_procedures_date_bool',
    'second_cath_procedure_performed',
    'indication_for_second_cath',
    'unknown_cath_operation_second_bool',
    'second_balloon_angioplasty_date_bool',
    'second_angioplasty_with_covered_stent_date_bool',
    'second_angioplasty_with_bare_metal_date_bool',
    'second_thoracic_endovascular_aneurysm_repair_date_bool',
    'second_hybrid_procedures_date_bool',
    'third_cath_procedure_performed',
    'indication_for_third_cath',
    'unknown_cath_operation_third_bool',
    'third_balloon_angioplasty_date_bool',
    'third_angioplasty_with_covered_stent_date_bool',
    'third_angioplasty_with_bare_metal_date_bool',
    'third_thoracic_endovascular_aneurysm_repair_date_bool',
    'third_hybrid_procedures_date_bool'
]

categoricalValuesOutcomes = [
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
    'need_for_surgical_vascular_site_repair'
]

allColumnsOutcomes = [
    'death',
    'systemic_hypertension',
    'claudication_pain',
    'presence_of_aneurysm_location',
    'presence_of_aortic_dissection',
    'stroke',
    'intracranial_aneurysm',
    'renal_failure',
    'heart_failure',
    'pulmonary_hypertension',
    'femoral_artery_occlusion',
    'coronary_artery_disease',
    'myocardial_infarction',
    'infective_endocarditis',
    'claudication_pain_lower_or_upper_left',
    'stent_fracture',
    'surgery_related_complications',
    'coronary_air_embolization',
    'stent_embolization',
    'emergent_surgery',
    'periprocedural_stroke',
    'aortic_dissection_post_angio',
    'arrhythmia_requiring_cardioversion',
    'need_for_blood_transfusion',
    'need_for_surgical_vascular_site_repair'
]


def createTableOne(df, type):
    groupby = ['cardiovascular_event'] if type != 'outcomes' else None
    pvalue = True if type != 'outcomes' else False
    allColumns = allColumnsDemographics if type == 'demographics' else allColumnsSurgeries if type == 'surgeries' else allColumnsOutcomes
    categoricalValues = categoricalValuesDemographics if type == 'demographics' else categoricalValuesSurgeries if type == 'surgeries' else categoricalValuesOutcomes
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    allColumns = [ i for i in allColumns if i not in empty_cols]
    categoricalValues = [ i for i in categoricalValues if i not in empty_cols]

    allCategorical = categoricalValuesDemographics + categoricalValuesSurgeries + categoricalValuesOutcomes
    nonNormal = []

    # Find non-normal columns
    for col in list(df.select_dtypes('number')): 
        if (col not in allCategorical and col in allColumns):
            a,b = stats.shapiro(df[col])
            if b < 0.05: 
                nonNormal.append(col)
                
    myTable = TableOne(df, columns=allColumns, categorical=categoricalValues, groupby=groupby, missing=False, nonnormal=nonNormal, pval=pvalue)
    return myTable
    