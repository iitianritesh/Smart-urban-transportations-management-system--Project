#!/usr/bin/env python
# coding: utf-8

# #  Project : Smart Urban Transportation   Management  System

# # Project Overview:
# 
# In this project, students will develop a comprehensive urban transportation management system using real-time data from various sources. This project simulates a real-world scenario where a city wants to optimize its transportation network, reduce congestion, and improve overall efficiency using data-driven decision-making.

# # Key Concepts Covered:
# 
# Python basics and data structures,
# Data cleaning and preprocessing,
# Exploratory data analysis,
# Feature engineering,
# Time series analysis,
# Supervised machine learning (classification and regression).
# 

# # Data Collection and Preprocessing:
# 
# Work with a dataset simulating real-time urban transportation data (e.g., traffic flow, public transit usage, weather conditions, events).
# 
# 
# Implement efficient data structures to handle and process large volumes of time-stamped data.
# 
# 
# Clean the data, handle missing or erroneous values, and normalize/scale features as needed.
# 
# 
# # Exploratory Data Analysis: 
# 
# Conduct statistical analysis on the transportation data.
# 
# 
# Visualize patterns, trends, and correlations in traffic flow, public transit usage, and other relevant factors.
# 
# 
# Identify potential indicators of traffic congestion and transportation network inefficiencies.
# 
# # Feature Engineering: 
# 
# 
# Create new features from the raw data (e.g., peak hour indicators, day of week, proximity to major events).
# 
# Implement algorithms to extract relevant features from geospatial and temporal data.
# 
# 
# # Time Series Analysis: 
# 
# Decompose time series data to identify trends, seasonality, and cyclical patterns in transportation usage.
# 
# Develop forecasting models to predict future traffic patterns and public transit demand.
# 
# # Supervised Machine Learning: 
# 
# Develop a classification model to predict traffic congestion levels in different areas of the city.
# 
# Implement a regression model to estimate travel times between key points in the city.
# 
# Compare multiple algorithms (e.g., Random Forest, Gradient Boosting, Neural Networks) and evaluate their performance.
# 
# 
# # Model Evaluation and Interpretation:
# 
# 
# Evaluate model performance using appropriate metrics (e.g., accuracy, precision, recall, F1-score for congestion classification; RMSE, MAE for travel time estimation).
# 
# 
# Interpret model results and identify the most significant factors contributing to transportation efficiency or inefficiency.
# 
# 
# # Reporting and Visualization: 
# 
# 
# Create a comprehensive report summarizing findings, methodologies, and recommendations for improving urban transportation.
# 
# Develop interactive visualizations (e.g., heat maps, animated time-lapse visualizations) to effectively present traffic patterns and system performance.
# 
# 

# # Tools and Technologies & Data Collection: 
# 
# APIs, web scraping tools, IoT devices
# 
# # Data Storage: 
# 
# SQL (PostgreSQL, MySQL), NoSQL (MongoDB, Cassandra)
# 
# # Data Processing: 
# 
# Pandas, NumPy, Apache Spark
# 
# # Machine Learning:
# 
# Scikit-learn, TensorFlow, Keras
# 
# # Visualization:
# 
# Matplotlib, Seaborn, Plotly, D3.js
# 
# # Web Development:
# 
# Flask, Django, React.js, Angular.js
# 
# # Cloud Services: 
# 
# AWS, Azure, Google Cloud
# 
# # Version Control:
# 
# Git, GitHub, GitLab

# # Feature EngineeringFeature Creation:
# 
# Create new features from the raw data (e.g., peak hour indicators, day of week, proximity to major events).Algorithm Implementation: Implement algorithms to extract relevant features from geospatial and temporal data

# # Time Series Analysis Decomposition: 
# 
# Decompose time series data to identify trends, seasonality, and cyclical patterns in transportation usage.Forecasting Models: Develop forecasting models to predict future traffic patterns and public transit demand.

# # Supervised Machine LearningClassification Model:
# 
# 
# Develop a classification model to predict traffic congestion levels in different areas of the city.
# 
# 
# # Regression Model: 
# 
# 
# Implement a regression model to estimate travel times between key points in the city.
# 
# 
# # Algorithm Comparison: 
# 
# 
# Compare multiple algorithms (e.g., Random Forest, Gradient Boosting, Neural Networks) and evaluate their performance.

# # Python Basics and Data Structures:
# 
# 
# syntax, data types, and data structures (lists, dictionaries, sets).Practice with libraries such as NumPy and Pandas for handling data.

# # Exploratory Data Analysis:
# 
# 
# Use Pandas and Matplotlib/Seaborn for data visualization.Identify key statistics and visualize trends and correlations.
# 

# # Data Collection and Integration:
# 
# 
# 
# Traffic Flow Data: Collect data from traffic sensors, cameras, and GPS devices installed in vehicles.Public Transit Data: Integrate data from public transportation systems like buses, trams, and trains.
# 
# 
# 
# Weather Data: Use weather APIs to incorporate real-time weather conditions.
# event Data: Include data about public events, holidays, and construction activities that could affect traffic.

# # Supervised Machine Learning
# 
# 
# 
# # Classification Model: 
# 
# 
# Use classification algorithms (e.g., Logistic Regression, Random Forest) to predict congestion levels.
# 
# 
# # Regression Model: 
# 
# 
# Use regression algorithms (e.g., Linear Regression, Gradient Boosting) to estimate travel times.
# 
# 
# 
# # Algorithm Comparison:
# 
# 
# Evaluate multiple algorithms to find the best-performing models for your specific use cases.

# # Implementation

# #Technology Stack
# 
# 
# 
# # Data Processing: 
# 
# 
# Python with libraries like Pandas, NumPy, and Scikit-learn.
# 
# 
# # Time Series Analysis:
# 
# 
# Libraries like statsmodels, Prophet, and TensorFlow for deep learning models.
# 
# 
# # Visualization:
# 
# 
# Matplotlib, Seaborn, Plotly, and Dash for creating static and interactive visualizations.
# 
# 
# # APIs and Integration:
# 
# 
# Use APIs for real-time data collection (e.g., OpenWeatherMap for weather data, city-specific APIs for traffic and public transit data).

# # Project Phases
# 
# 
# # Phase 1:
# 
# 
# # Data Collection and Preprocessing
# 
# 
# Collect and clean the data.
# 
# 
# Ensure data alignment and consistency.
# 
# 
# # Phase 2: 
# 
# # Exploratory Data Analysis
# 
# 
# Perform statistical analysis and visualization.
# 
# 
# # Phase 3: 
# 
# # Feature Engineering
# 
# 
# Create new features and enhance existing ones.
# 
# 
# # Phase 4: 
# 
# 
# # Time Series Analysis and Forecasting
# 
# 
# Analyze trends and forecast future patterns.
# 
# 
# # Phase 5:
# 
# # Supervised Learning Models
# 
# 
# Build and evaluate classification and regression models.
# 
# 
# # Phase 6:
# 
# Model Interpretation and Reporting
# 
# 
# Interpret model results and compile the final report.
# 
# 
# # Phase 7: 
# 
# # Visualization and Dashboard Creation
# 
# 
# Develop interactive visualizations and dashboards.
# 
# 
# Possible Challenges and Solutions
# 
# 
# Data Quality: Ensure high-quality data by cleaning and preprocessing effectively.
# 
# 
# # Model Accuracy:
# 
# Experiment with different algorithms and hyperparameters to improve model performance.
# 
# 
# # Scalability: 
# 
# efficient data structures and algorithms to handle large datasets.
# 
# 
# # Real-Time Processing: 
# 
# 
# Consider using real-time data processing frameworks like Apache Kafka or Spark Streaming for real-time analysis.

#  # Python Basics and Data Structures

# In[ ]:





# 
# 

# In[ ]:





# In[ ]:





# # data preprocessing
# 

# In[50]:


import pandas as pd

# Load the dataset
data = pd.read_csv("C:/Users/rites/OneDrive/Desktop/futuristic_city_traffic.csv")

# Display column names
print(data.columns)
print(data)



# # Exploratory Data Analysis (EDA)
# 

# In[51]:


import matplotlib.pyplot as plt
import seaborn as sns

# Statistical summary
print(data.describe())

# Visualize traffic flow over time
plt.figure(figsize=(12, 6))

plt.title('Traffic Flow Over Time')
plt.show()

# Heatmap of correlations
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()



# # Feature Engineering
# 

# In[52]:


import pandas as pd
import numpy as np

# Load the dataset
data = pd.read_csv("C:/Users/rites/OneDrive/Desktop/futuristic_city_traffic.csv")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Print column names to ensure correctness
print(data.columns)

# Convert 'Hour Of Day' and 'Day Of Week' to numeric if they are not already
data['Hour Of Day'] = pd.to_numeric(data['Hour Of Day'], errors='coerce')
data['Day Of Week'] = pd.to_numeric(data['Day Of Week'], errors='coerce')

# Create a peak hour indicator (1 if peak hour, 0 otherwise)
data['is_peak_hour'] = data['Hour Of Day'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)

# Create a weekend indicator (1 if weekend, 0 otherwise)
data['is_weekend'] = data['Day Of Week'].apply(lambda x: 1 if x >= 5 else 0)

# Display the first few rows to verify the new columns
print(data.head())


# # Time Series Analysis

# In[54]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
data = pd.read_csv("C:/Users/rites/OneDrive/Desktop/futuristic_city_traffic.csv")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Ensure 'Traffic Density' is a numeric type
data['Traffic Density'] = pd.to_numeric(data['Traffic Density'], errors='coerce')

# Plot Traffic Density over time
plt.figure(figsize=(12, 6))
plt.plot(data['Traffic Density'])
plt.title('Traffic Density Over Time')
plt.xlabel('Index')
plt.ylabel('Traffic Density')
plt.show()

# Decompose time series data
decomposition = seasonal_decompose(data['Traffic Density'].dropna(), model='multiplicative', period=24)
decomposition.plot()
plt.show()


#  # Supervised Machine Learning

# In[61]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("C:/Users/rites/OneDrive/Desktop/futuristic_city_traffic.csv")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Ensure 'Hour Of Day' and 'Day Of Week' are numeric
data['Hour Of Day'] = pd.to_numeric(data['Hour Of Day'], errors='coerce')
data['Day Of Week'] = pd.to_numeric(data['Day Of Week'], errors='coerce')

# Create new features
data['is_peak_hour'] = data['Hour Of Day'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)
data['is_weekend'] = data['Day Of Week'].apply(lambda x: 1 if pd.notna(x) and x >= 5 else 0)

# Check for missing values in the target column and drop those rows
data['Traffic Density'] = pd.to_numeric(data['Traffic Density'], errors='coerce')
data = data.dropna(subset=['Traffic Density'])

# Define features and target
features = ['Hour Of Day', 'Day Of Week', 'is_peak_hour', 'is_weekend']
target = 'Traffic Density'

# Extract features and target variable
X = data[features]
y = data[target]

# Handle missing values in features (if any)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Optional: Visualize feature importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.show()


# # For Regression:

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/rites/OneDrive/Desktop/futuristic_city_traffic.csv")

# Print column names to ensure correctness
print(data.columns)

# Define features and target
# Replace 'Feature' with the actual feature you want to analyze
feature = 'Speed'  # Example feature
target = 'Traffic Density'  # Example target variable

# Ensure the feature and target are numeric
data[feature] = pd.to_numeric(data[feature], errors='coerce')
data[target] = pd.to_numeric(data[target], errors='coerce')

# Drop rows with NaN values in the feature or target
data = data.dropna(subset=[feature, target])

# Plot the regression line
plt.figure(figsize=(10, 6))
sns.regplot(x=feature, y=target, data=data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.title(f'Regression Line for {feature} vs {target}')
plt.xlabel(feature)
plt.ylabel(target)
plt.show()



# # reporting and visualizations

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
data = pd.read_csv("C:/Users/rites/OneDrive/Desktop/futuristic_city_traffic.csv")

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Convert 'Hour Of Day' and 'Day Of Week' to numeric
data['Hour Of Day'] = pd.to_numeric(data['Hour Of Day'], errors='coerce')
data['Day Of Week'] = pd.to_numeric(data['Day Of Week'], errors='coerce')

# Create indicators
data['is_peak_hour'] = data['Hour Of Day'].apply(lambda x: 1 if 7 <= x <= 9 or 16 <= x <= 19 else 0)
data['is_weekend'] = data['Day Of Week'].apply(lambda x: 1 if pd.notna(x) and x >= 5 else 0)

# Ensure 'Traffic Density' is numeric
data['Traffic Density'] = pd.to_numeric(data['Traffic Density'], errors='coerce')
data = data.dropna(subset=['Traffic Density'])

# Visualization: Traffic Density Over Time
plt.figure(figsize=(12, 6))
plt.plot(data['Traffic Density'])
plt.title('Traffic Density Over Time')
plt.xlabel('Index')
plt.ylabel('Traffic Density')
plt.grid(True)
plt.show()

# Decomposition of Traffic Density Time Series
decomposition = seasonal_decompose(data['Traffic Density'].dropna(), model='multiplicative', period=24)
plt.figure(figsize=(12, 8))
decomposition.plot()
plt.suptitle('Seasonal Decomposition of Traffic Density', fontsize=16)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()

# Regression Line Visualization for a specific feature
feature = 'Speed'  # Example feature
target = 'Traffic Density'

# Ensure feature and target are numeric
data[feature] = pd.to_numeric(data[feature], errors='coerce')
data[target] = pd.to_numeric(data[target], errors='coerce')

# Drop rows with NaN values in the feature or target
data = data.dropna(subset=[feature, target])

# Plot regression line
plt.figure(figsize=(10, 6))
sns.regplot(x=feature, y=target, data=data, scatter_kws={'s':10}, line_kws={'color':'red'})
plt.title(f'Regression Line for {feature} vs {target}')
plt.xlabel(feature)
plt.ylabel(target)
plt.grid(True)
plt.show()


# # Reporting time and summary of this code 
# 

# 
#  # Displays the dataset’s basic information and the first few rows.
# # Shows the statistical summary of the dataset.
# # Traffic Density Over Time:
# 
# # Plots the traffic density data over time to visualize trends.
# # Seasonal Decomposition:
# 
# # Decomposes the traffic density time series to identify seasonal patterns, trends, and residuals.
# # Correlation Heatmap:
# 
# # Visualizes correlations between different features in the dataset to identify potential relationships.
# # Regression Line Visualization:
# 
# # Plots a regression line for a chosen feature against the target variable to visualize the relationship.
# # Feature Importance from Random Forest Model:
# 
# # Trains a Random Forest model to predict traffic density and evaluates its performance.
# # Optionally visualizes feature importance to understand which features contribute most to the model

# 

# 

# 
