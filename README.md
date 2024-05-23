# Variables and Variable Selection

## Overview
This repository contains a Jupyter notebook focused on the concepts of variables and variable selection. It demonstrates the complete process from importing a dataset and libraries, through data preprocessing and encoding, to analyzing correlations and building a linear regression model. The notebook is designed to help data scientists and analysts understand the importance of variable selection in predictive modeling.

## Contents
- **Data Import and Libraries**: Import necessary libraries and load the dataset from a provided URL.
- **Data Preprocessing**: Clean and prepare the data for analysis, including renaming columns and filtering data.
- **Variable Types and Summary Statistics**: Explore the data with descriptive statistics and data types.
- **Dummy Variable Encoding**: Convert categorical variables into dummy/indicator variables to be used in modeling.
- **Correlation Analysis**: Analyze the relationships between variables using a correlation matrix.
- **Linear Regression Model**: Split the data into training and testing sets, fit a linear regression model, and evaluate its performance.
- **Multiple Linear Regression Analysis After Variable selection**

## Notebook Highlights

### 1. Data Import and Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/bootcamps/Personal_Loans.csv')
df.head()
```

### 2. Data Preprocessing
- **Rename Columns**: Replace spaces with underscores in column names for easier access.
- **Filter Data**: Focus on rows where 'Personal_Loan' is 1 and drop the 'Personal_Loan' column.

```python
df.columns = [col.replace(" ","_") for col in df.columns] 
df = df[df['Personal_Loan'] == 1]
df = df.drop(['Personal_Loan'],axis=1)
df.head()
```

### 3. Variable Types and Summary Statistics
- **Data Info**: Display data types and non-null counts.
- **Descriptive Statistics**: Provide summary statistics for numerical variables.

```python
df.info()
df.describe()
```

### 4. Dummy Variable Encoding
- **Encoding**: Convert categorical variables to dummy variables.
- **Rename Columns**: Ensure all column names use underscores.

```python
df_dummies = pd.get_dummies(df)
df_dummies.columns = [col.replace(" ","_") for col in df_dummies.columns] 
df_dummies.head()
```

### 5. Correlation Analysis and Model Structure
- **Correlation Matrix**: Visualize the correlations between variables.
- **Model Preparation**: Reorder columns to place 'Loan_Size' as the target variable.

```python
df_dummies.corr()
column_titles = [col for col in df_dummies.columns if col!= 'Loan_Size'] + ['Loan_Size']
df_dummies = df_dummies.reindex(columns=column_titles)
```

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/variables-and-variable-selection.git
   ```
2. **Open the Notebook**:
   Navigate to the cloned directory and open the Jupyter notebook:
   ```bash
   jupyter notebook variables_and_variable_selection.ipynb
   ```
Here's a GitHub description for your notebook:

---

## Multiple Linear Regression Analysis After Variable selection

### Overview
This notebook presents a comprehensive analysis of a dataset using multiple linear regression techniques. The primary goal is to predict loan sizes based on various predictor variables.

### Key Steps
1. **Data Exploration and Preprocessing**: 
   - Exploring the dataset to understand its structure and variables.
   - Preprocessing steps such as handling missing values and encoding categorical variables.

2. **Model Building**: 
   - Utilizing the Ordinary Least Squares (OLS) method to fit a linear regression model.
   - Constructing the OLS formula string "y ~ X" to specify the model.
   - Fitting the model to the data and evaluating its performance using summary statistics.

3. **Variable Selection**: 
   - Conducting variable selection based on correlation coefficients, p-values, and variance thresholds.
   - Identifying statistically significant features and removing highly correlated variables.

4. **Variance Thresholding**: 
   - Applying variance thresholding to select features with significant variance.
   - Normalizing data and analyzing column variances to identify relevant variables.
**Repository Name:** OLS-Regression-Analysis

**Notebook Contents:**
- **OLS Fit Summary:** The notebook begins by showcasing the OLS regression results, including model formula, coefficients, R-squared, F-statistic, and more.
- **Train and Compare Models:** It then proceeds to split the dataset into training and testing sets, followed by fitting three different linear regression models:
  - **No Threshold Model:** Utilizes all predictive variables.
  - **Correlation Threshold Model:** Uses variables selected based on correlation thresholding.
  - **Variance Threshold Model:** Incorporates variables selected based on variance thresholding.
- **Assessment of Model Accuracy:** The notebook assesses the accuracy of each model by comparing training and testing mean squared error (MSE) and R-squared ($R^2$) values.

**Insights:**
The analysis reveals that, contrary to intuition, reducing the number of predictors in the model can lead to improved performance. Specifically, while the "No Threshold" model exhibited the lowest training MSE, it demonstrated the highest testing MSE, indicating overfitting. In contrast, the other two models showed better generalization capabilities, highlighting the importance of feature selection in enhancing model robustness.

**Conclusion:**
By leveraging OLS regression and comprehensive model evaluation techniques, this notebook offers valuable insights into building predictive models and optimizing feature selection strategies for improved model performance and generalization.

**Keywords:** OLS Regression, Linear Regression, Feature Selection, Model Evaluation, Python, Jupyter Notebook.

Feel free to clone, explore, and contribute to this repository to further enrich our understanding of regression analysis techniques! 📊🔍
