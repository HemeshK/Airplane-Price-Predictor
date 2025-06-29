import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Airplane Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .subheader {
        font-size: 1.2rem;
        color: #475569;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-container {
        background-color: #F0F9FF;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .prediction-text {
        font-size: 1.8rem;
        font-weight: bold;
        color: #0369A1;
        text-align: center;
    }
    .form-container {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .input-label {
        font-weight: bold;
        color: #334155;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #64748B;
        font-size: 0.8rem;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        width: 100%;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# Load Dataset
file_path = "/Users/hemeshkukreja/Desktop/FinalProject/Dataset/Plane Price.csv"
df = pd.read_csv(file_path)


# Data Preprocessing Function
def data_preprocessing(df):
    categorical_features = ['Engine Type']
    numerical_features = [
        'Rcmnd cruise Knots', 'Stall Knots dirty', 'Fuel gal/lbs',
        'Eng out rate of climb', 'Takeoff over 50ft'
    ]

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    X = df.drop("Price", axis=1)
    y = df["Price"].dropna()
    X = X.loc[y.index]

    X_preprocessed = preprocessor.fit_transform(X)
    return X_preprocessed, y, preprocessor, numerical_features, categorical_features


X, y, preprocessor, numerical_features, categorical_features = data_preprocessing(df)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=43)
model.fit(X_train, y_train)

# Initialize session state variables
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'show_prediction' not in st.session_state:
    st.session_state.show_prediction = False
if 'form_submitted' not in st.session_state:
    st.session_state.form_submitted = False


# Function to reset form
def reset_form():
    st.session_state.form_submitted = True
    # Rerun will happen after this function completes


# Streamlit UI - Header Section
st.markdown('<div class="main-header">✈️ Airplane Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Enter aircraft specifications to estimate the price</div>', unsafe_allow_html=True)

# Create columns layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Display previous prediction if available
    if st.session_state.show_prediction and st.session_state.prediction_result is not None:
        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        st.markdown(f'<div class="prediction-text">Estimated Price: ${st.session_state.prediction_result:,.2f}</div>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Form container
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    # Form inputs - Reset keys if form was submitted
    with st.form(key=f'plane_predictor_form_{st.session_state.form_submitted}'):
        st.markdown('<div class="input-label">Aircraft Performance Metrics</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            cruiseKnots = st.text_input("Recommended Cruise (Knots)*",
                                        help="The recommended cruising speed of the aircraft in knots")
            stallKnotsDirty = st.text_input("Stall Speed Dirty (Knots)*",
                                            help="The stall speed with landing gear and flaps extended")
            fuel = st.text_input("Fuel Capacity (gal/lbs)*", help="The fuel capacity in gallons or pounds")

        with col_b:
            rateOfClimb = st.text_input("Engine Out Rate of Climb*",
                                        help="The rate of climb with one engine out (for multi-engine aircraft)")
            takeOffOver50ft = st.text_input("Takeoff Over 50ft*",
                                            help="The distance required to takeoff and clear a 50ft obstacle")
            engineType = st.selectbox("Engine Type*", options=["Piston", "Propjet", "Jet"],
                                      help="The type of engine powering the aircraft")

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label='PREDICT PRICE')

        if submit_button:
            try:
                with st.spinner("Calculating price..."):
                    new_data = {
                        'Rcmnd cruise Knots': float(cruiseKnots.strip()) if cruiseKnots.strip() else None,
                        'Stall Knots dirty': float(stallKnotsDirty.strip()) if stallKnotsDirty.strip() else None,
                        'Fuel gal/lbs': float(fuel.strip()) if fuel.strip() else None,
                        'Eng out rate of climb': float(rateOfClimb.strip()) if rateOfClimb.strip() else None,
                        'Takeoff over 50ft': float(takeOffOver50ft.strip()) if takeOffOver50ft.strip() else None,
                        'Engine Type': engineType,
                    }
                    if None in new_data.values():
                        raise ValueError("Please fill in all required fields")

                    new_data_df = pd.DataFrame([new_data])
                    new_data_processed = preprocessor.transform(new_data_df)

                    predicted_price = model.predict(new_data_processed)[0]

                    # Store prediction in session state
                    st.session_state.prediction_result = predicted_price
                    st.session_state.show_prediction = True

                    # Reset the form
                    reset_form()
                    st.rerun()

            except ValueError as e:
                st.error(f"⚠️ {str(e)}")
            except Exception as e:
                st.error(f"⚠️ An error occurred: {str(e)}")

    # Close form container
    st.markdown('</div>', unsafe_allow_html=True)

    # Add information section
    with st.expander("About this predictor"):
        st.markdown("""
        This aircraft price predictor uses machine learning to estimate the price of airplanes based on their specifications.

        **How it works:**
        - The model is trained on historical aircraft data
        - It analyzes the relationship between aircraft specifications and prices
        - Enter your desired aircraft specifications to get a price estimate

        **Note:** The prediction is an estimate and actual prices may vary depending on market conditions, 
        aircraft age, equipment options, and other factors not included in this model.
        """)

# Footer
st.markdown('<div class="footer">© 2025 Aircraft Price Prediction System. All rights reserved.</div>',
            unsafe_allow_html=True)