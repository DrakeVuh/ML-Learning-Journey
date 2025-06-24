import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('./ML-Learning-Journey/linear_regression/du_doan_diem_so/student_scores.csv')

print(data.head())
print(data.describe())
print(data.info())
print(data.isnull().sum())

X = data.iloc[:, :-1].values
y = data['final_score'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

print(X_train_scaled[:5])
print(X_test_scaled[:5])

print("Kích thước X_train:", X_train.shape)
print("Kích thước X_test:", X_test.shape)
print("Kích thước y_train:", y_train.shape)
print("Kích thước y_test:", y_test.shape)