import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Đọc dữ liệu
DATA_PATH = 'new_house_data.csv'
data = pd.read_csv(DATA_PATH)

# Chia features và target
X = data.drop('price', axis=1)
y = data['price']

# Chia train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Huấn luyện Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Đánh giá trên tập validation
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)
print(f"Validation MSE: {val_mse:.2f}, R2: {val_r2:.4f}")

# Đánh giá trên tập test
y_test_pred = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test MSE: {test_mse:.2f}, R2: {test_r2:.4f}")

# Lưu model
joblib.dump(model, 'linear_model.pkl')
print("Model đã được lưu vào linear_model.pkl") 