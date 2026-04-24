import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# OBJECTIVE 1: COLLECT AND INTEGRATE AGRICULTURAL DATA
df = pd.read_csv("C:/Users/yadav/Downloads/yadavdivash818_1776529698713534.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.shape)
print(df.columns)


# OBJECTIVE 2: CLEAN AND PREPROCESS DATA (NumPy & Pandas)

# Rename columns (they are too long)
df.columns = [
    'Country', 'State', 'District', 'Year',
    'Net_Govt_Canal', 'Net_Pvt_Canal', 'Net_Total_Canal',
    'Net_Tank', 'Net_Tubewell', 'Net_Other_Well', 'Net_Total_Well',
    'Net_Other_Source', 'Net_Total_Irrigated',
    'Gross_Govt_Canal', 'Gross_Pvt_Canal', 'Gross_Total_Canal',
    'Gross_Tank', 'Gross_Tubewell', 'Gross_Other_Well', 'Gross_Total_Well',
    'Gross_Other_Source', 'Gross_Total_Irrigated'
]

# Clean Year column  "Agriculture Year (Jul - Jun), 2022" --> 2022
df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(int)
print(df['Year'].head())

# Check missing values
print(df.isnull().sum())

# Fill missing values
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print(df.isnull().sum())

# Check duplicates
print("Duplicates:", df.duplicated().sum())

# Min-Max Normalization on key columns
scaler = MinMaxScaler()
df[['Net_Total_Irrigated', 'Gross_Total_Irrigated']] = scaler.fit_transform(
    df[['Net_Total_Irrigated', 'Gross_Total_Irrigated']]
)
print(df[['Net_Total_Irrigated', 'Gross_Total_Irrigated']].describe())


# OBJECTIVE 3: DATA VISUALIZATION (Matplotlib & Seaborn)

# Line Plot - Top 5 States average Net Irrigated Area over Years
top5 = df.groupby('State')['Net_Total_Irrigated'].mean().nlargest(5).index
df_top5 = df[df['State'].isin(top5)]


plt.figure(figsize=(10, 5))
for state in top5:
    state_data = df_top5[df_top5['State'] == state].groupby('Year')['Net_Total_Irrigated'].mean()
    plt.plot(state_data.index, state_data.values, marker='o', label=state)

plt.title('Average Net Irrigated Area Over Years (Top 5 States)')
plt.xlabel('Year')
plt.ylabel('Net Total Irrigated (Normalized)')
plt.legend()
plt.grid()
plt.xticks(range(1998, 2023, 2), rotation=45)  
plt.tight_layout()                              
plt.show()

# Bar Chart - Average Gross Irrigated Area by Source
source_means = df[['Gross_Govt_Canal', 'Gross_Pvt_Canal', 'Gross_Tank', 'Gross_Tubewell']].mean()
plt.figure(figsize=(8, 4))
plt.bar(source_means.index, source_means.values, color='steelblue', edgecolor='black')
plt.title('Average Gross Irrigated Area by Source')
plt.xlabel('Source')
plt.ylabel('Average Area (Normalized)')
plt.show()

# Histogram - Distribution of Net Total Irrigated Area
plt.figure(figsize=(8, 4))
plt.hist(df['Net_Total_Irrigated'], bins=30, color='purple', edgecolor='black', alpha=0.6)
plt.title('Histogram of Net Total Irrigated Area')
plt.xlabel('Net Total Irrigated (Normalized)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot - Net vs Gross Irrigated Area
plt.figure(figsize=(8, 4))
sns.scatterplot(x='Net_Total_Irrigated', y='Gross_Total_Irrigated', data=df, alpha=0.4, color='green')
plt.title('Net vs Gross Irrigated Area')
plt.xlabel('Net Total Irrigated')
plt.ylabel('Gross Total Irrigated')
plt.show()

# Heatmap - Correlation
corr = df[['Net_Total_Irrigated', 'Gross_Total_Irrigated', 'Net_Tubewell', 'Net_Tank', 'Gross_Tubewell']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Irrigated Area Features')
plt.show()

# Box Plot - Net Irrigated by top States
plt.figure(figsize=(8, 4))
sns.boxplot(x='State', y='Net_Total_Irrigated', data=df_top5, palette='pastel')
plt.title('Boxplot of Net Irrigated Area (Top 5 States)')
plt.xticks(rotation=30)
plt.show()

# Pie Chart - Share of Irrigation Sources
source_totals = df[['Net_Total_Canal', 'Net_Total_Well', 'Net_Tank', 'Net_Other_Source']].sum()
source_totals.plot.pie(autopct='%1.1f%%', figsize=(6, 6))
plt.title('Share of Irrigation Sources')
plt.ylabel('')
plt.show()


# OBJECTIVE 4: EDA AND STATISTICAL ANALYSIS

# Summary Statistics
print(df[['Net_Total_Irrigated', 'Gross_Total_Irrigated']].describe())

# Skewness and Kurtosis
print("Skewness:\n", df[numeric_cols].skew())
print("Kurtosis:\n", df[numeric_cols].kurt())

# Correlation
print("Correlation:\n", df[['Net_Total_Irrigated', 'Gross_Total_Irrigated', 'Net_Tubewell']].corr())

# Outlier Detection using IQR
Q1 = df['Net_Total_Irrigated'].quantile(0.25)
Q3 = df['Net_Total_Irrigated'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Net_Total_Irrigated'] < lower_bound) | (df['Net_Total_Irrigated'] > upper_bound)]
print("Outliers found:", len(outliers))

# Hypothesis Testing - t-test
# Do Canal-irrigated and Tubewell-irrigated areas differ significantly?
t_stat, p_value = stats.ttest_ind(df['Net_Total_Canal'], df['Net_Tubewell'])
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
if p_value < 0.05:
    print("Result: Significant difference between Canal and Tubewell irrigation")
else:
    print("Result: No significant difference")

# Shapiro-Wilk Normality Test (sample of 500 for speed)
sample = df['Net_Total_Irrigated'].dropna().sample(500, random_state=42)
stat, p = stats.shapiro(sample)
print(f"Shapiro-Wilk: stat={stat:.4f}, p={p:.4f}")

# Pair Plot
sns.pairplot(df[['Net_Total_Irrigated', 'Gross_Total_Irrigated', 'Net_Tubewell', 'Net_Tank']].dropna())
plt.show()


# OBJECTIVE 5: MACHINE LEARNING - LINEAR REGRESSION

# Predict Gross Total Irrigated from Net Total Irrigated
x = df[['Net_Total_Irrigated']]
y = df['Gross_Total_Irrigated']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Scatter + Regression Line
plt.figure(figsize=(8, 4))
plt.scatter(x, y, color='blue', alpha=0.3)
plt.plot(x, model.predict(x), color='red', linewidth=2)
plt.xlabel('Net Total Irrigated (Normalized)')
plt.ylabel('Gross Total Irrigated (Normalized)')
plt.title('Linear Regression: Net vs Gross Irrigated Area')
plt.show()

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE : {mse:.4f}")
print(f"R2  : {r2:.4f}")
print(f"MAE : {mae:.4f}")

# Predict for a custom value
check = pd.DataFrame({'Net_Total_Irrigated': [0.5]})
result = model.predict(check)
print("Predicted Gross Irrigated for Net=0.5:", result)
