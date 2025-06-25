import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from LinearRegression import MultipleLinearRegression
import matplotlib.pyplot as plt

# Đọc dữ liệu từ thư mục data_generation
df = pd.read_csv('../Du_doan_gia_nha/house_prices.csv')

# Tách features và target
X = df.drop('price', axis=1).values
y = df['price'].values

# Chia dữ liệu thành train và test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo và huấn luyện mô hình
model = MultipleLinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train_scaled, y_train)

# Đánh giá mô hình
train_score = model.score(X_train_scaled, y_train)
test_score = model.score(X_test_scaled, y_test)

print("\nKết quả đánh giá:")
print(f"R2 score trên tập train: {train_score:.4f}")
print(f"R2 score trên tập test: {test_score:.4f}")

# In ra các trọng số của mô hình
feature_names = ['Diện tích', 'Số phòng ngủ', 'Số phòng tắm', 'Khoảng cách', 'Tuổi nhà']
print("\nTrọng số của các đặc trưng:")
for name, weight in zip(feature_names, model.weights):
    print(f"{name}: {weight:.4f}")

# Vẽ đồ thị loss
plt.figure(figsize=(10, 6))
plt.plot(range(model.n_iterations), model.history['loss'])
plt.title('Loss History')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show() 