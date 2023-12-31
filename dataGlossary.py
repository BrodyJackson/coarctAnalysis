demographicFeatures = [
    'age',
    'sex',
    'height',
    'weight',
    'bmi'
]

vascularRisk = [
    'hypertension',
    'dyslipidemia',
    'smoking_status',
    'diabetes',
    'family_premature_cad_hist',
]

currentClinicalFeatures = [
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
    'number_of_transcatheter_interventions',
    'number_of_open_surgical_interventions',
    'coarctation_less_three_mm',
    'interrupted_aortic_arch',
    'presence_of_collaterals',
    'ecg_sinus_rhythm',
    'ecg_afib',
    'cardiopulmonary_exercise_test_performed',
]

imagingValues = [
    'diameter_at_widest_ascending_aorta_max',
    'diameter_at_coarct_site_max',
    'diameter_at_post_stenotic_site_max',
    'diameter_at_diaphragm_max',
    'imaging_coarct_ratio'
]

currentMedications = [
    'beta_blockers',
    'calcium_channel_blockers',
    'ace_inhibitors_arbs',
    'asa',
    'statins',
    'diuretics',
    'total_num_antihypertensives'
]

surgicalHistory = {

}

#Hidden events in the main form (which may be null) such as pseudoaneurysm, all dates, and tranfusion numbers are not in this list
cardiovascularEvents = {
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
}

majorCardioEvent = {
    'presence_of_aneurysm_location',
    'presence_of_aortic_dissection',
    'renal_failure',
    'heart_failure',
    'myocardial_infarction',
    'stroke',
    'infective_endocarditis',
    'death'
}

surgeryOperations = {
    'unknown_operation_first',
    'resection_end_to_end_anastamosis_index',
    'patch_angioplasty_date_index',
    'subclavian_flap_angioplasty_index',
    'interposition_graft_index',
    'unknown_operation_second',
    'resection_end_to_end_anastamosis_second',
    'patch_angioplasty_date_second',
    'subclavian_flap_angioplasty_second',
    'interposition_graft_second',
    'unknown_operation_third',
    'resection_end_to_end_anastamosis_third',
    'patch_angioplasty_date_third',
    'subclavian_flap_angioplasty_third',
    'interposition_graft_third'
}

cathOperations = {
    'unknown_cath_operation_first',
    'index_balloon_angioplasty_date',
    'index_angioplasty_with_covered_stent_date',
    'index_angioplasty_with_bare_metal_date',
    'index_thoracic_endovascular_aneurysm_repair_date',
    'index_hybrid_procedures_date',
    'unknown_cath_operation_second',
    'second_balloon_angioplasty_date',
    'second_angioplasty_with_covered_stent_date',
    'second_angioplasty_with_bare_metal_date',
    'second_thoracic_endovascular_aneurysm_repair_date',
    'second_hybrid_procedures_date',
    'unknown_cath_operation_third',
    'third_balloon_angioplasty_date',
    'third_angioplasty_with_covered_stent_date',
    'third_angioplasty_with_bare_metal_date',
    'third_thoracic_endovascular_aneurysm_repair_date',
    'third_hybrid_procedures_date'
}

categoricalChoices = {
    'sex': {0: 'Female', 1: 'Male', 2: 'Other'},
    'diabetes': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    # 'smoking_status': {0: 'Never smoked', 1: 'Ex smoker', 2: 'Current smoker', 9: 'Unknown'},
    'smoking_status': {0: 'Never smoked', 1: 'History of smoking', 9: 'Unknown'},
    'family_premature_cad_hist': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'hypertension': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'dyslipidemia': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'claudication_pain': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'aortopathies': {0: 'No', 1: 'Turner', 2: 'Marfan', 3: 'Bicuspid aortic valve with ascending aortic aneurysm', 4: 'Hypoplastic aorta/aortic arch', 9: 'Unknown'},
    'aortic_valve_morphology': {0: 'Native tri-leaflet', 1: 'Repaired/replaced tri-leaflet', 2: 'Native bicuspid', 3: 'Repaired/replaced bicuspid'},
    #The one below changed from original data mapping
    'indication_for_repair': {0: 'Normal', 1: 'Aortic Stenosis', 2: 'Aortic Insufficiency/Regurgitation'},
    'valve_current_condition': {0: 'Normal', 1: 'Stenotic', 2: 'Regurgitant'},
    #The one below changed from original data mapping
    'valve_current_type': {0: 'Normal', 1: 'Bio-prosthetic ', 2: 'Mechanical ', 3: 'Previously ballooned', 4: 'Surgically repaired native valve', 9: 'Unknown'},
    'aortic_aneurysm': {0: 'No (Normal)', 1: 'Aortic root aneurysm', 2: 'Ascending aortic aneurysm', 3: 'Aneurysm extending to ascending aortic arch', 4: 'Diffuse aortic aneurysm', 5: 'Subclavian associated aneurysm', 6: 'Aneurysm of descending aorta', 7: 'Other aneurysm'},
    'aortic_aneurysm_repaired': {0: 'No', 1: 'Surgical Repair', 2: '', 3: 'Trans-catheter repair'},
    'current_coarctation_present': {0: 'No', 1: 'Yes'},
    'coarctation_type': {0: 'Native coarctation', 2: 'Re-coarctation'},
    'previous_coarctation_intervention': {0: 'No', 1: 'Yes'},
    'coarctation_less_three_mm': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'coarctation_with_complete_obstruction': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'interrupted_aortic_arch': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'presence_of_collaterals': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'ecg_sinus_rhythm': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'ecg_afib': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'cardiopulmonary_exercise_test_performed': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'beta_blockers': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'calcium_channel_blockers': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'ace_inhibitors_arbs': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'asa': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'statins': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'diuretics': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'index_surgical_procedure_performed': {0: 'No', 1: 'Yes'},
    'indication_for_first_surgery': {0: 'Initial coarctation repair', 1: 'Re-coarctation with 20 mmHg BP difference (or greater) across coarctation', 2: 'Re-coarctation with less than 20 mmHg BP difference with LV dysfunction, aortic insufficiency, collaterals, or hypertension', 3: 'Aneurysm repair (subclavian, transverse, or descending aorta)', 4: 'Coarctation fixed during other non-related cardiac operation', 5: 'Aortic dissection', 6: 'Staged expansion of a stent', 7: 'Infectious complications', 9: 'Unknown'},
    'patch_angioplasty_material_index': {0: 'Pericardial patch', 2: 'Aortic homograft', 3: 'Synthetic patches (dacron, gelweave, goretex)', 9: 'Unknown'},
    'second_surgical_procedure_performed': {0: 'No', 1: 'Yes'},
    'indication_for_second_surgery': {0: 'Re-coarctation with 20 mmHg BP difference (or greater) across coarctation', 1: 'Re-coarctation with less than 20 mmHg BP difference with LV dysfunction, aortic insufficiency, collaterals, or hypertension', 2: 'Aneurysm repair (subclavian, transverse, or descending aorta)', 3: 'Coarctation fixed during other non-related cardiac operation', 4: 'Aortic dissection', 5: 'Staged expansion of a stent', 6: 'Infectious complications', 9: 'Unknown'},
    'patch_angioplasty_material_second': {0: 'Pericardial patch', 2: 'Aortic homograft', 3: 'Synthetic patches (dacron, gelweave, goretex)', 9: 'Unknown'},
    'third_surgical_procedure_performed': {0: 'No', 1: 'Yes'},
    'indication_for_third_surgery': {0: 'Re-coarctation with 20 mmHg BP difference (or greater) across coarctation', 1: 'Re-coarctation with less than 20 mmHg BP difference with LV dysfunction, aortic insufficiency, collaterals, or hypertension', 2: 'Aneurysm repair (subclavian, transverse, or descending aorta)', 3: 'Coarctation fixed during other non-related cardiac operation', 4: 'Aortic dissection', 5: 'Staged expansion of a stent', 6: 'Infectious complications', 9: 'Unknown'},
    'patch_angioplasty_material_third': {0: 'Pericardial patch', 2: 'Aortic homograft', 3: 'Synthetic patches (dacron, gelweave, goretex)', 9: 'Unknown'},
    'index_cath_procedure_performed': {0: 'No', 1: 'Yes'},
    'indication_for_first_cath': {0: 'Initial coarctation repair', 1: 'Re-coarctation with 20 mmHg BP difference (or greater) across coarctation', 2: 'Re-coarctation with less than 20 mmHg BP difference with LV dysfunction, aortic insufficiency, collaterals, or hypertension', 3: 'Aneurysm repair (subclavian, transverse, or descending aorta)', 4: 'Coarctation fixed during other non-related cardiac operation', 5: 'Aortic dissection', 6: 'Staged expansion of a stent', 7: 'Infectious complications', 9: 'Unknown'},
    'second_cath_procedure_performed': {0: 'No', 1: 'Yes'},
    'indication_for_second_cath': {0: 'Re-coarctation with 20 mmHg BP difference (or greater) across coarctation', 1: 'Re-coarctation with less than 20 mmHg BP difference with LV dysfunction, aortic insufficiency, collaterals, or hypertension', 2: 'Aneurysm repair (subclavian, transverse, or descending aorta)', 3: 'Coarctation fixed during other non-related cardiac operation', 4: 'Aortic dissection', 5: 'Staged expansion of a stent', 6: 'Infectious complications', 9: 'Unknown'},
    'third_cath_procedure_performed': {0: 'No', 1: 'Yes'},
    'indication_for_third_cath': {0: 'Re-coarctation with 20 mmHg BP difference (or greater) across coarctation', 1: 'Re-coarctation with less than 20 mmHg BP difference with LV dysfunction, aortic insufficiency, collaterals, or hypertension', 2: 'Aneurysm repair (subclavian, transverse, or descending aorta)', 3: 'Coarctation fixed during other non-related cardiac operation', 4: 'Aortic dissection', 5: 'Staged expansion of a stent', 6: 'Infectious complications', 9: 'Unknown'},
    'presence_of_aneurysm_location': {0: 'No aneurysm', 1: 'At site of repair', 2: 'Ascending aortic aneurysm', 3: 'At subclavian artery', 4: 'Descending aortic aneurysm', 5: 'Other aneurysm site', 9: 'Unknown'},
    'pseudoaneurysm': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'presence_of_aortic_dissection': {0: 'No', 1: 'Yes (Type A)', 2: 'Yes (Type B)', 9: 'Unknown'},
    'systemic_hypertension': {0: 'No', 1: 'Yes - Now Resolved', 2: 'Yes - Persisting', 9: 'Unknown'},
    'renal_failure': {0: 'No', 1: 'Yes - Acute', 2: 'Yes - Chronic', 9: 'Unknown'},
    'heart_failure': {0: 'No', 1: 'Yes - Systolic', 2: 'Yes - Diastolic', 9: 'Unknown'},
    'pulmonary_hypertension': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'femoral_artery_occlusion': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'coronary_artery_disease': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'myocardial_infarction': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'stroke': {0: 'No', 1: 'Yes - Ischemic', 2: 'Yes - Intracerebral hemorrhage', 9: 'Unknown'},
    'intracranial_aneurysm': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'infective_endocarditis': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'claudication_pain_lower_or_upper_left': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'stent_fracture': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'surgery_related_complications': {0: 'No', 1: 'Yes - Spinal cord complications', 2: 'Yes - Other operative complications', 9: 'Unknown'},
    'death': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'coronary_air_embolization': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'stent_embolization': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'emergent_surgery': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'periprocedural_stroke': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'aortic_dissection_post_angio': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'arrhythmia_requiring_cardioversion': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'need_for_blood_transfusion': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'need_for_surgical_vascular_site_repair': {0: 'No', 1: 'Yes', 9: 'Unknown'},
    'transferred_centres': {0: 'No', 1: 'Yes'},
    'loss_to_follow_up': {0: 'No', 1: 'Yes'},
    'had_one_op_type': {0: 'No', 1: 'Only received surgeries', 2: 'Only received catheters'},
    'first_op_type': {0: 'No', 1: 'Surgery first', 2: 'Cath first'}
}

operationDateColumns = {
    'unknown_operation_first',
    'resection_end_to_end_anastamosis_index',
    'patch_angioplasty_date_index',
    'subclavian_flap_angioplasty_index',
    'interposition_graft_index',
    'unknown_operation_second',
    'resection_end_to_end_anastamosis_second',
    'patch_angioplasty_date_second',
    'subclavian_flap_angioplasty_second',
    'interposition_graft_second',
    'resection_end_to_end_anastamosis_third',
    'patch_angioplasty_date_third',
    'subclavian_flap_angioplasty_third',
    'interposition_graft_third',
    'unknown_cath_operation_first',
    'index_balloon_angioplasty_date',
    'index_angioplasty_with_covered_stent_date',
    'index_angioplasty_with_bare_metal_date',
    'index_thoracic_endovascular_aneurysm_repair_date',
    'index_hybrid_procedures_date',
    'second_balloon_angioplasty_date',
    'second_angioplasty_with_covered_stent_date',
    'second_angioplasty_with_bare_metal_date',
    'second_thoracic_endovascular_aneurysm_repair_date',
    'second_hybrid_procedures_date',
    'third_balloon_angioplasty_date',
    'third_angioplasty_with_covered_stent_date',
    'third_angioplasty_with_bare_metal_date',
    'third_thoracic_endovascular_aneurysm_repair_date',
    'third_hybrid_procedures_date'
}

outcomeDateColumns = {
    'date_of_aortic_aneurysm_repair',
    'date_of_afib',
    'date_of_aneurysm',
    'date_of_aortic_dissection',
    'date_of_renal_failure',
    'date_of_heart_failure',
    'date_of_femoral_artery_occlusion',
    'date_of_coronary_artery_disease',
    'date_of_mi',
    'date_of_stroke',
    'date_of_intracranial_aneurysm',
    'date_of_infective_endocarditis',
    'date_of_claudication_pain',
    'date_of_stent_fracture',
    'date_of_surgery_complications',
    'date_of_death',
    'date_of_coronary_air_embolization',
    'date_of_stent_embolization',
    'date_of_emergent_surgery',
    'date_of_periprocedural_stroke',
    'date_of_aortic_dissection_post_angio',
    'date_of_cardioversion',
    'date_of_blood_transfusion',
    'date_of_surgical_repair_vascular_access'
}

renameColumns = {
    'coexistent_lesions___0': 'lesion_none',
    'coexistent_lesions___1': 'lesion_asd',
    'coexistent_lesions___2': 'lesion_vsd',
    'coexistent_lesions___3': 'lesion_hypoplastic_left_heart',
    'coexistent_lesions___4': 'lesion_mitral_regurg_stenosis',
    'coexistent_lesions___5': 'lesion_pda',
    'coexistent_lesions___6': 'lesion_transposition',
    'coexistent_lesions___7': 'lesion_ascending_aneurysm_aorta',
    'coexistent_lesions___8': 'lesion_interrupted_aortic_arch',
    'coexistent_lesions___9': 'lesion_other'
}
tableColumns = [('Grouped by cardiovascular_event', 'Overall'),('Grouped by cardiovascular_event','No Cardiovascular Event'),('Grouped by cardiovascular_event','Cardiovascular Event'),('Grouped by cardiovascular_event', 'P-Value')]
tableLevels = ['Overall','No Cardiovascular Event','Cardiovascular Event', 'P-Value']

hiddenAttributes = {
    'indication_for_repair',
    'valve_current_condition',
    'valve_current_type',
}

needsAZeroValue = {
    'indication_for_repair',
    'valve_current_type',
}