import numpy as np

# Dữ liệu mẫu
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

# Thêm bias
X_b = np.c_[np.ones((3, 1)), X]

# Hàm mất mát có regularization (L2)
def loss_with_l2(theta, X, y, lam):
    y_pred = X.dot(theta)
    mse = np.mean((y - y_pred) ** 2)
    l2 = lam * np.sum(theta[1:] ** 2)  # Không regularize bias
    return mse + l2

theta = np.random.randn(2)
lam = 0.1
print("Loss with L2 regularization:", loss_with_l2(theta, X_b, y, lam))