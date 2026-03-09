import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


data = {
    "study_hours": [1,2,3,4,5,6,7,8,9,10],
    "attendance": [50,55,60,65,70,75,80,85,90,95],
    "previous_score": [40,42,45,50,55,60,65,70,75,80],
    "assignment_score": [30,35,40,45,50,55,60,65,70,75],
    "pass_fail": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df.drop("pass_fail", axis=1)
y = df["pass_fail"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)


st.title("Student Pass/Fail Predictor")

st.write("Enter student details to predict result")
study_hours = st.slider("Study Hours", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 70)
previous_score = st.slider("Previous Exam Score", 0, 100, 60)
assignment_score = st.slider("Assignment Score", 0, 100, 60)


if st.button("Predict Result"):

    input_data = [[study_hours, attendance, previous_score, assignment_score]]

    prediction = pipeline.predict(input_data)

    if prediction[0] == 1:
        st.success(" Student will PASS")
        st.balloons()
    else:
        st.error(" Student will FAIL")