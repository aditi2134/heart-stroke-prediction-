import streamlit as st
import pandas as pd 
import joblib as jb 

model = jb.load('KNN_HEART.pkl')
scaler = jb.load('scaler.pkl')
expected_columns = jb.load('columns.pkl')

st.title('HEART STROKE PREDICTION')
st.markdown("Provide the details")

age = st.slider("age", 18, 100, 40)
sex = st.selectbox("sex", ["M", "F"])
chest_pain = st.selectbox("chest_pain_type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("resting_bp (mm hg)", 80, 200, 120)
cholesterol = st.number_input("cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("fasting_bloodsugar", [0, 1])
resting_ecg = st.selectbox("resting_ecg", ["Normal", "ST", "LVH"])
max_hr = st.slider("max heart rate", 60, 220, 150)
exercise_angina = st.selectbox("exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("olspeak (st depression)", 0.0, 0.6, 0.1)
st_slope = st.selectbox("ST_slope", ['up', 'flat', 'down'])


if st.button("Predict"):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    # Add missing columns
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Arrange columns in the same order as the training data
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
