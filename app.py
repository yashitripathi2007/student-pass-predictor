import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('model.pkl')

st.title("🎓 Will the Student Pass? Simple Predictor")
st.write("Enter student details to predict if they will pass the final exam (G3 ≥ 10)")

# Input fields
studytime = st.slider("Weekly study time (1 = <2h, 4 = >10h)", 1, 4, 2)
failures = st.slider("Number of past class failures", 0, 3, 0)
absences = st.slider("Number of school absences", 0, 93, 5)
higher = st.radio("Wants higher education?", ("yes", "no"))
internet = st.radio("Internet access at home?", ("yes", "no"))

if st.button("🔮 Predict"):
    higher_val = 1 if higher == "yes" else 0
    internet_val = 1 if internet == "yes" else 0

    input_data = pd.DataFrame({
        'studytime': [studytime],
        'failures': [failures],
        'absences': [absences],
        'higher_yes': [higher_val],
        'internet_yes': [internet_val]
    })

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"✅ Likely to **PASS**! Probability: {prob:.0%}")
    else:
        st.error(f"⚠️ Risk of **FAIL**. Probability of passing: {prob:.0%}")
    
    st.info("Note: This is a simple model for demonstration.")

# Run info
st.caption("Built with Python, scikit-learn & Streamlit")