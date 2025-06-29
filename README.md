Airplane Price Predictor

This project is a machine learning-based web application designed to predict the resale price of airplanes based on key performance metrics. The tool allows users to input specifications such as cruising speed, stall speed, fuel capacity, and engine type, and receive a reliable estimate of the airplane’s market value.

Project Description

The model is trained on a curated dataset of airplane specifications and historical prices. It uses regression techniques to analyze how various performance features influence the resale value. The application is deployed with a user-friendly interface using Streamlit, allowing real-time predictions.

Key Features:
Interactive web interface for user input and predictions
Real-time price estimation based on aircraft specifications
Preprocessing pipeline for handling missing values and encoding
Trained regression model (Random Forest Regressor) with cross-validation
Modular codebase for easy expansion and deployment
Tech Stack

Category	Tools and Libraries
Programming Language-	Python
Data Handling-	pandas, numpy
Visualization-	matplotlib, seaborn
Machine Learning-	scikit-learn
Preprocessing-	MinMaxScaler, OneHotEncoder, Pipelines
Web Framework-	Streamlit
Model Used-	Random Forest Regressor


How It Works-

The dataset is preprocessed using a pipeline that includes:
Median imputation for missing numerical values
Min-max scaling for numerical features
One-hot encoding for categorical variables
The following input features are collected via the UI:
Recommended cruise speed
Stall speed (dirty configuration)
Fuel capacity
Engine-out rate of climb
Takeoff distance over a 50 ft obstacle
Engine type (Piston, Jet, or Propjet)
These inputs are processed and fed into the trained model, which returns a price estimate.
The prediction is displayed to the user along with appropriate UI feedback and information on the model’s scope and limitations.
