import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Load and prepare data
df = pd.read_csv('student-por.csv', sep=';')
df['pass'] = (df['G3'] >= 10).astype(int)

# Encode categorical features
df['higher_yes'] = (df['higher'] == 'yes').astype(int)
df['internet_yes'] = (df['internet'] == 'yes').astype(int)

# Features
features = ['studytime', 'failures', 'absences', 'higher_yes', 'internet_yes']
X = df[features]
y = df['pass']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, pred_lr))

# Train Random Forest (usually better)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, pred_rf))

# Save the better model (Random Forest)
joblib.dump(model_rf, 'model.pkl')
print("Model saved as model.pkl")
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_rf))