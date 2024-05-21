import joblib
import numpy as np
import streamlit as st
import pandas as pd

loaded_model = joblib.load('C:/Users/niran/Documents/VS Code/Python/Heart Disease Prediction Model/heart_disease_prediction_model.pkl')
features = ['age',
    'sex',
    'chest pain type',
    'resting bp s',
    'cholesterol',
    'fasting blood sugar',
    'resting ecg',
    'max heart rate',
    'exercise angina',
    'oldpeak',
    'ST slope']

def prediction(input_data):
    values_reshaped = np.array(input_data).reshape(1,-1)
    values = pd.DataFrame(values_reshaped, columns = features)
    
    prediction = loaded_model.predict(values)[0]
    prediction_prob = loaded_model.predict_proba(values)

    if prediction == 1:
        st.write('The Model Predicts That You Have a HIGH Risk of Heart Disease!')
        st.write(f"Prediction Probability: {round(prediction_prob[0][1], 2)}")
    else:
        st.write('The Model Predicts That You Have a LOW Risk of Heart Disease!')
        st.write(f"Prediction Probability: {round(prediction_prob[0][0], 2)}")

def main():
    st.title("Heart Disease Prediction")

    age = st.slider("Age", min_value = 0, max_value = 100)
    sex = st.selectbox('Sex', ['Female', 'Male'])
    cpt = st.selectbox('Chest Pain Type', ["Typical Angina",
                                          "Atypical Angina",
                                          "Non-Anginal Pain",
                                          "Asymptomatic"])
    rbp = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Serum Cholesterol")
    fbs = st.selectbox('Fasting Blood Sugar', ["> 120 mg/dl", "< 120 mg/dl"])    
    rer = st.selectbox('Resting Electrocardiogram Results', ["Normal",
                                "Having ST-T Wave Abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)",
                                "Showing Probable Or Definite Left Ventricular Hypertrophy by Estes' Criteria"])
    mhr = st.number_input("Maximum Heart Rate Achieved")
    eia = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
    op = st.number_input("Old Peak")
    sts = st.selectbox('ST Slope', ["Upsloping",
                                    "Flat",
                                    "Downsloping"])
    # --- Mapping ---
    sex_mapping = {'Female': 0, 'Male': 1}
    sex_numeric = sex_mapping[sex]

    cpt_mapping = {'Typical Angina': 1,
                   'Atypical Angina': 2,
                   'Non-Anginal Pain': 3,
                   'Asymptomatic': 4}
    cpt_numeric = cpt_mapping[cpt]

    fbs_mapping = {'> 120 mg/dl': 1,
                   '< 120 mg/dl': 0}
    fbs_numeric = fbs_mapping[fbs]
    
    rer_mapping = {"Normal": 0,
                   'Having ST-T Wave Abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)': 1,
                   "Showing Probable Or Definite Left Ventricular Hypertrophy by Estes' Criteria": 2}
    rer_numeric = rer_mapping[rer]

    eia_mapping = {'No': 0, 'Yes': 1}
    eia_numeric = eia_mapping[eia]

    sts_mapping = {'Upsloping': 1, 'Flat': 2, 'Downsloping': 3}
    sts_numeric = sts_mapping[sts]
    # ---------

    if st.button("Predict"):
        input_data = [age,sex_numeric,cpt_numeric,rbp,chol,fbs_numeric,rer_numeric,mhr,eia_numeric,op,sts_numeric]
        prediction(input_data)
    
if __name__ == '__main__':
    main()