import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Set page config for wide layout and title
st.set_page_config(layout="wide", page_title="Employee Salary Predictor")

# --- Custom CSS for enhanced styling with fixed visibility ---
# Updated CSS section in your Streamlit app
# Updated CSS section for metrics boxes
st.markdown(
    """
    <style>
    /* Metrics boxes styling with soft blue background */
    .metric-box {
        background-color: #e3f2fd !important;  /* Soft blue background */
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #bbdefb;  /* Light blue border */
        color: black !important;  /* Ensures text stays black */
    }
    
    /* Metric title styling */
    .metric-box h4 {
        color: #0d47a1 !important;  /* Dark blue for titles */
        font-size: 1.3em;
        margin-top: 0;
        margin-bottom: 15px;
    }
    
    /* Main metric value (large number) */
    .metric-box p:first-of-type {
        font-size: 2.5em;
        color: #1565c0 !important;  /* Medium blue for emphasis */
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    /* Description text */
    .metric-box p:not(:first-of-type) {
        color: #1e3a8a !important;  /* Slightly darker blue for readability */
        font-size: 0.95em;
        line-height: 1.4;
        margin-bottom: 0;
    }
    
    /* Hover effect for metrics */
    .metric-box:hover {
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load the model and data for plotting ---
model = None
ytest = None
predic1 = None
xtrain_columns = None

st.info("Select a Feature in the Navigation section to access them")

try:
    with open("model.pkl", "rb") as file:
        loaded_data = pickle.load(file)
    model = loaded_data.get('model')
    ytest = loaded_data.get('ytest')
    predic1 = loaded_data.get('predic1')
    xtrain_columns = loaded_data.get('xtrain_columns')

    if model is None or ytest is None or predic1 is None or xtrain_columns is None:
        st.error("Error: 'model.pkl' is missing required components.")
        st.stop()
except FileNotFoundError:
    st.error("Error: 'model.pkl' not found.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

# Define categories for dropdowns
GENDER_OPTIONS = ['Male', 'Female']
EDUCATION_OPTIONS = ['Bachelor', 'Master', 'PhD']

# Extract unique categories from xtrain_columns
job_titles_from_dummies = sorted(list(set([col.replace('Job_Title_', '') for col in xtrain_columns if col.startswith('Job_Title_')])))
department_from_dummies = sorted(list(set([col.replace('Department_', '') for col in xtrain_columns if col.startswith('Department_')])))
location_from_dummies = sorted(list(set([col.replace('Location_', '') for col in xtrain_columns if col.startswith('Location_')])))

COMMON_JOB_TITLES = sorted(list(set(['Software Engineer', 'Data Scientist', 'HR Manager', 'Sales Representative', 'Marketing Specialist', 'Financial Analyst'] + job_titles_from_dummies)))
COMMON_DEPARTMENTS = sorted(list(set(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance', 'IT'] + department_from_dummies)))
COMMON_LOCATIONS = sorted(list(set(['New York', 'London', 'San Francisco', 'Chicago', 'Houston'] + location_from_dummies)))

# --- Preprocessing function ---
def preprocess_input(input_data_dict, xtrain_columns):
    input_df = pd.DataFrame([input_data_dict])
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['Education_Level'] = input_df['Education_Level'].map({'Bachelor': 1, 'Master': 2, 'PhD': 3})
    processed_df = pd.DataFrame(0, index=[0], columns=xtrain_columns)

    processed_df['Age'] = input_df['Age']
    processed_df['Gender'] = input_df['Gender']
    processed_df['Education_Level'] = input_df['Education_Level']
    processed_df['Experience_Years'] = input_df['Experience_Years']

    job_title_col = f"Job_Title_{input_df['Job_Title'].iloc[0]}"
    if job_title_col in xtrain_columns:
        processed_df[job_title_col] = 1

    department_col = f"Department_{input_df['Department'].iloc[0]}"
    if department_col in xtrain_columns:
        processed_df[department_col] = 1

    location_col = f"Location_{input_df['Location'].iloc[0]}"
    if location_col in xtrain_columns:
        processed_df[location_col] = 1

    return processed_df[xtrain_columns]

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page_selection = st.sidebar.pills(
    "Go to",
    ("Predict Salary", "Model Description & Graphs", "Factors for Salary > 100k")
)

# --- Main Content Area ---
st.title("Employee Salary Prediction Web App Using ML")

if page_selection == "Predict Salary":
    st.header("Predict Employee Salary")
    st.markdown("<p style='color:light blue; font-weight:bold;'>Enter the employee's details to get a salary prediction.</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1, key="age_input")
        gender = st.selectbox("Gender", GENDER_OPTIONS, key="gender_select")
        education_level = st.selectbox("Education Level", EDUCATION_OPTIONS, key="education_select")

    with col2:
        experience_years = st.number_input("Experience Years", min_value=0, max_value=40, value=5, step=1, key="exp_input")
        job_title = st.selectbox("Job Title", COMMON_JOB_TITLES, key="job_title_select")
        department = st.selectbox("Department", COMMON_DEPARTMENTS, key="department_select")
        location = st.selectbox("Location", COMMON_LOCATIONS, key="location_select")

    st.markdown(
        """
        <div class="custom-info-box">
            <p style='color:white;'>
                <strong>Note:</strong> The lists for Job Title, Department, and Location are inferred from common values and the model's training data.
            </p>
        </div>
        """, unsafe_allow_html=True
    )

    if st.button("Predict Salary"):
        input_data = {
            'Age': age,
            'Gender': gender,
            'Education_Level': education_level,
            'Experience_Years': experience_years,
            'Job_Title': job_title,
            'Department': department,
            'Location': location
        }

        try:
            processed_input = preprocess_input(input_data, xtrain_columns)
            predicted_salary = model.predict(processed_input)[0]
            st.success(f"Predicted Salary: ${predicted_salary:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif page_selection == "Model Description & Graphs":
    st.header("Model Description & Performance")
    
    st.subheader("Model Performance Metrics")
    r2score = r2_score(ytest, predic1)
    mae = mean_absolute_error(ytest, predic1)

    col_metrics1, col_metrics2 = st.columns(2)
    with col_metrics1:
        st.markdown(f"""
        <div class="metric-box">
            <h4 style='color:black;'>R-squared (R²) Score</h4>
            <p style='color:black;'>{r2score:.4f}</p>
            <p style='color:black;'>Measures how well the regression predictions approximate the real data points.</p>
        </div>
        """, unsafe_allow_html=True)
    with col_metrics2:
        st.markdown(f"""
        <div class="metric-box">
            <h4 style='color:black;'>Mean Absolute Error (MAE)</h4>
            <p style='color:black;'>${mae:,.2f}</p>
            <p style='color:black;'>Represents the average absolute difference between predicted and actual values.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Feature Coefficients")
    coefficients_df = pd.DataFrame({'Feature': xtrain_columns, 'Coefficient': model.coef_})
    st.dataframe(coefficients_df.sort_values(by='Coefficient', ascending=False))

    st.subheader("Visualizations")
    st.markdown("---")
    st.markdown("### Actual vs. Predicted Salary Scatter Plot")
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    sns.scatterplot(x=ytest, y=predic1, alpha=0.6, color='#1976d2', label='Predicted Points', ax=ax1)
    sns.lineplot(x=ytest, y=ytest, color='red', linestyle='--', label='Perfect Prediction', ax=ax1)
    ax1.set_xlabel("Actual Salary")
    ax1.set_ylabel("Predicted Salary")
    ax1.legend()
    st.pyplot(fig1)

    st.markdown("---")
    st.markdown("### Distribution of Actual vs. Predicted Salaries")
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    sns.histplot(ytest, kde=True, color='green', label='Actual Salary', bins=30, ax=ax2)
    sns.histplot(predic1, kde=True, color='#1976d2', label='Predicted Salary', bins=30, ax=ax2)
    ax2.set_xlabel("Salary")
    ax2.legend()
    st.pyplot(fig2)

elif page_selection == "Factors for Salary > 100k":
    st.header("How to Achieve a Salary Greater Than $100,000")
    st.markdown("""
    <div style="background-color:#ffffff; padding:25px; border-radius:12px; box-shadow:0 4px 8px rgba(0,0,0,0.1);">
        <h4 style="color:black;">Key Influencing Factors for High Salaries:</h4>
        <ul style="color:black;">
            <li><strong>Extensive Experience:</strong> More years of relevant experience</li>
            <li><strong>Advanced Education:</strong> Master's degrees or PhDs</li>
            <li><strong>High-Demand Job Titles:</strong> Senior technical or leadership roles</li>
            <li><strong>Strategic Departments:</strong> Tech, Finance, or R&D</li>
            <li><strong>Major Economic Hubs:</strong> High-cost, high-salary locations</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)