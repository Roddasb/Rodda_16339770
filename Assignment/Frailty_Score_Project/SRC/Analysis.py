import pandas as pd
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Read the data from the CSV file
df = pd.read_csv("/content/Clean_Frailty_Score.csv")

# Perform descriptive statistics
descriptive_stats = df.describe()

# Display the descriptive statistics
print(descriptive_stats)

# Perform t-test
t_statistic, p_value = ttest_ind(df[df['Frailty'] == 0]['Grip strength'], df[df['Frailty'] == 1]['Grip strength'])

# Display the preprocessed dataset
print("\nPreprocessed Dataset:")
print(df)

# Print t-test results
print("\nT-Statistic:", t_statistic)
print("P-Value:", p_value)

# Calculate correlation matrix
correlation_matrix = df.corr()

# Display correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)



# Visualization of age distribution by frailty category
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Frailty', y='Age')
plt.title('Age Distribution by Frailty')
plt.xlabel('Frailty')
plt.ylabel('Age')
plt.show()

# Summary statistics of age by frailty category
summary_stats = df.groupby('Frailty')['Age'].describe()
print("Summary Statistics of Age by Frailty:")
print(summary_stats)

# Define the independent variables (age and grip strength)
X = df[['Age', 'Grip strength']]

# Add constant for the intercept term
X = sm.add_constant(X)

# Define the dependent variable (frailty score)
y = df['Frailty']

# Fit logistic regression model
logit_model = sm.Logit(y, X)
result = logit_model.fit()

# Print summary of logistic regression
print(result.summary())