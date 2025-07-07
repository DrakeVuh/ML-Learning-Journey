# run_iris_logistic.py

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from iris_logistic_model import IrisLogisticModel

# 1. Load và chuẩn bị dữ liệu
iris = load_iris()
X = iris.data
y = (iris.target == 0).astype(int)  # Phân loại Setosa

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2. Khởi tạo và train model (L1 regularization)
model = IrisLogisticModel(penalty='l1', C=1.0)
model.train(X_train, y_train)

# 3. Đánh giá và in kết quả
print("Train accuracy:", model.score(X_train, y_train))
print("Test accuracy:", model.score(X_test, y_test))
print("Weights:", model.get_weights())