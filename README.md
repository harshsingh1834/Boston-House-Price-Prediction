Boston House Price Prediction

Overview

This repository contains a machine learning project to predict house prices in Boston using the Boston Housing dataset. The project implements a linear regression model to analyze relationships between housing features (e.g., crime rate, number of rooms, lower status population) and median house prices. It includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation, showcasing skills in Python, data science, and visualization.

This project was developed as part of a B.Tech in Computer Science and Engineering at COER University to demonstrate proficiency in machine learning and data analysis.

Features

Dataset: Boston Housing dataset, containing 506 samples with 13 features (e.g., CRIM, RM, LSTAT) and the target variable (MEDV: median house value).
EDA: Visualizations of feature distributions (histograms) and correlations (heatmap) using seaborn and matplotlib.
Preprocessing: Handling missing values with mean imputation and standardizing features using StandardScaler.
Model: Linear regression model trained to predict house prices.
Evaluation: Performance metrics including Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
Visualization: Scatter plot comparing predicted vs. actual house prices.

Technologies Used
Python: Core programming language.

Libraries:

pandas: Data manipulation and analysis.
numpy: Numerical computations.
matplotlib & seaborn: Data visualization.
scikit-learn: Machine learning (LinearRegression, StandardScaler, train_test_split, metrics).



Environment: Google Colab (or Kaggle notebook, as per the setup script).

Project Structure
boston-house-price-prediction/
│

├── boston-house-prices.csv    # Dataset file

├── boston_house_prediction.ipynb # Jupyter notebook with code

├── README.md                  # This file

Setup Instructions


Clone the Repository:

git clone https://github.com/your-username/boston-house-price-prediction.git
cd boston-house-price-prediction

Install Dependencies: Ensure Python 3.6+ is installed. Install required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn

Download Dataset: The dataset is included as boston-house-prices.csv in the repository. Alternatively, it can be sourced from Kaggle (as per the notebook’s import script).

Run the Notebook:

Open boston_house_prediction.ipynb in Jupyter Notebook or Google Colab.
Ensure the dataset path is correctly set to /kaggle/input/boston-house-price-prediction/ or update to your local path (e.g., ./boston-house-prices.csv).

How to Run

Open the Notebook: Launch boston_house_prediction.ipynb in Jupyter Notebook or Google Colab.

Execute Cells:

Run the initial cells to set up the Kaggle environment (if using Kaggle) or skip to the data loading section for local execution.
Load the dataset using pd.read_csv('boston-house-prices.csv').

Follow the notebook’s steps:

EDA: Visualize feature distributions and correlation heatmap.
Preprocessing: Impute missing values and standardize features.
Model Training: Train the linear regression model.
Evaluation: Compute MAE, MSE, and RMSE; visualize predicted vs. actual prices.

Expected Output:

Feature distribution plots (e.g., crime rate, number of rooms).
Correlation matrix heatmap.
Model coefficients and intercept.
Performance metrics (e.g., MAE: ~3.29, RMSE: ~4.95).
Scatter plot of predicted vs. actual house prices.

Results

Model Performance:
Mean Absolute Error (MAE): 3.29
Mean Squared Error (MSE): 24.48
Root Mean Squared Error (RMSE): 4.95

Key Insights:

Features with high positive correlation to house prices: RM (average number of rooms, 0.70), ZN (zoning, 0.36).
Features with high negative correlation: LSTAT (lower status population, -0.74), PTRATIO (pupil-teacher ratio, -0.51).
The linear regression model effectively captures linear relationships but may benefit from feature engineering or non-linear models for improved accuracy.

Future Improvements

Experiment with advanced models (e.g., Random Forest, XGBoost) to improve prediction accuracy.
Perform feature engineering (e.g., polynomial features) to capture non-linear relationships.
Address potential outliers or skewed features (e.g., CRIM, LSTAT) through robust preprocessing.
Incorporate cross-validation to ensure model generalizability.

About the Author

This project was developed by Harsh Singh, a B.Tech student in Computer Science and Engineering at COER University, Roorkee, India. Connect with me on LinkedIn or GitHub.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Dataset: Kaggle Boston Housing dataset.
Inspiration: COER University coursework and online data science resources (e.g., Coursera, Kaggle tutorials).
