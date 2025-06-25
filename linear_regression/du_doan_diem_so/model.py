import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Linear_Regression import MultipleLinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('student_scores.csv')

X = data.drop('final_score', axis = 1).values
Y = data['final_score'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MultipleLinearRegression(learning_rate=0.01, n_interations=1000)
model.fit(X_train_scaled, Y_train)

train_score = model.score(X_train_scaled, Y_train)
test_score = model.score(X_test_scaled, Y_test)

print(f"Train score: {train_score:.4f}")
print(f"Test score: {test_score:.4f}")

feature_names = ['study_hours_per_week', 'group_study_per_month', 'extra_classes_per_month', 'sleep_hours_per_day', 'extracurricular_per_month']
print("\nTrọng số của các đặc trưng:")
for name, weight in zip(feature_names, model.weights):
    print(f"{name}: {weight:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(model.n_interations), model.history['loss'])
plt.title('Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()  