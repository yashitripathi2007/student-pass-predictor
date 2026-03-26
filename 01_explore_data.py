import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('student-por.csv', sep=';')

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Create target: Pass if G3 >= 10
df['pass'] = (df['G3'] >= 10).astype(int)
print("\nPass vs Fail count:")
print(df['pass'].value_counts())

# Quick plots (will open in separate windows)
sns.countplot(x='pass', data=df)
plt.title("Pass vs Fail Distribution")
plt.show()

sns.boxplot(x='pass', y='absences', data=df)
plt.title("Absences by Pass/Fail")
plt.show()

print("Exploration done!")