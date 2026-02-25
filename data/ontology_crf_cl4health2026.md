# CRF:filling SharedTask @ CL4Health2026 — Output Ontology

Source: official `NLP-FBK/dyspnea-valid-options` dataset.  
Total items: **134**.

Important note (empirical, dev/train annotations):
- Even for items whose valid-options list suggests `measured | unknown`, the ground truth in `NLP-FBK/dyspnea-crf-development` contains **numeric / percent** values (e.g. `spo2=95%`, `leukocytes=24530`).
- For this repo’s prompts/normalization we therefore allow numeric values for lab/measurement items.

## 1) `certainly chronic | possibly chronic | certainly not chronic | unknown`

- chronic pulmonary disease
- chronic respiratory failure
- chronic cardiac failure
- chronic renal failure
- chronic metabolic failure
- chronic rheumatologic disease
- chronic dialysis

## 2) `certainly active | possibly active | certainly not active | unknown`

- active neoplasia

## 3) `short | long | unknown`

- duration of the patient's consciousness recovery
- duration of the patient's unconsciousness

## 4) `y | n | unknown`

- first episod of epilepsy
- known history of epilepsy
- history of allergy
- history of recent trauma
- pregnancy
- history of drug abuse
- history of alcohol abuse
- anticoagulants or antiplatelet drug therapy
- presence of prodromal symptoms
- compliance with antiepileptic therapy
- tloc during effort
- tloc while supine
- antiepileptic therapy already in place
- drowsiness, confusion, disorientation as postcritical state
- stiffness during the episode
- drooling during the episode
- tonic-clonic seizures
- poly-pharmacological therapy
- pale skin during the episode
- eye deviation during the episode
- diffuse vascular disease
- neuropsychiatric disorders
- presence of pacemaker
- presence of defibrillator
- cardio-pulmonary resuscitation
- antihypertensive therapy
- cardiovascular diseases
- neurodegenerative diseases
- peripheral neuropathy
- immunosuppression
- palliative care
- situation description, like coughing, prolonged periods of straining, sudden abdominal pain, phlebotomy
- problematic family context
- need but absence of a caregiver
- homelessness
- living alone
- chest pain
- head or other districts trauma
- tongue bite
- agitation
- foreign body in the airways
- improvement of dyspnea
- presence of dyspnea
- dementia
- general condition deterioration
- ab ingestis pneumonia
- further seizures in the ed
- improvement of patient’s conditions
- neurologist consultation
- ecg, any abnormality
- ecg monitoring, any abnormality
- eeg, any abnormality
- thoracic ultrasound, any abnormalities
- chest rx, any abnormalities
- gastroscopy , any abnormalities
- brain ct scan, any abnormality
- brain mri, any abnormality
- cardiac ultrasound, any abnormality
- chest ct scan, any abnormality
- pulmonary scintigraphy, any abnormality
- abdomen ct scan, any abnormality
- compression ultrasound (cus), any abnormality
- performance of thoracentesis
- administration of diuretics
- administration of steroids
- administration of bronchodilators
- administration of oxygen/ventilation
- blood transfusions
- administration of fluids
- heart failure
- pneumonia
- copd exacerbation
- acute pulmonary edema
- asthma exacerbation
- respiratory failure
- intoxication
- covid 19
- influenza and various infections
- pneumothorax
- situational syncope
- epilepsy / epileptic seizure
- pulmonary embolism
- arrhythmia
- cardiac tamponade
- aortic dissection
- acute coronary syndrome
- hemorrhage
- severe anemia
- concussive head trauma

## 5) `walking independently | walking with auxiliary aids | walking with physical assistance | bedridden | unknown`

- level of autonomy (mobility)

## 6) `A | V | P | unknown`

- level of consciousness

## 7) `bradypneic | eupneic | tachypneic | unknown`

- respiratory rate

## 8) `hypothermic | normothermic | hyperthermic | unknown`

- body temperature

## 9) `bradycardic | normocardic | tachycardic | unknown`

- heart rate

## 10) `hypotensive | normotensive | hypertensive | unknown`

- blood pressure

## 11) `current | past | unknown`

- presence of respiratory distress

## 12) `pos | neg | unknown`

- carotid sinus massage
- supine-to-standing systolic blood pressure test
- blood in the stool
- sars-cov-2 swab test

## 13) `measured | unknown`

- spo2
- ph
- pa02
- pac02
- hc03-
- lactates
- hemoglobin
- platelets
- leukocytes
- c-reactive protein
- blood sodium
- blood potassium
- blood glucose
- creatinine
- transaminases
- inr
- troponin
- bnp or nt-pro-bnp
- d-dimer
- blood calcium
- serum creatinine kinase
- blood alcohol
- blood drug dosage
- urine drug test
