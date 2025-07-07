import pandas as pd
import joblib
import sys

# Sử dụng: python predict.py <file_data_moi.csv>
if len(sys.argv) != 2:
    print("Cách dùng: python predict.py <file_data_moi.csv>")
    sys.exit(1)

DATA_PATH = sys.argv[1]

# Đọc model đã lưu
model = joblib.load('linear_model.pkl')

# Đọc dữ liệu mới
new_data = pd.read_csv(DATA_PATH)

# Nếu có cột 'price' thì bỏ đi (chỉ lấy features)
if 'price' in new_data.columns:
    X_new = new_data.drop('price', axis=1)
else:
    X_new = new_data

# Dự đoán
preds = model.predict(X_new)

# In kết quả
for i, p in enumerate(preds):
    print(f"Dòng {i+1}: Giá dự đoán = {p:.2f}") 