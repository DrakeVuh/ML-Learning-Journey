"""
softmaxregression_debug.py
==========================

# PIPELINE TỔNG QUÁT (WORKFLOW OVERVIEW)
# 1. Sinh dữ liệu giả lập 3 lớp
# 2. Chuyển nhãn sang one-hot
# 3. Khởi tạo trọng số W
# 4. Huấn luyện softmax regression (gradient descent)
# 5. Dự đoán, đánh giá, visualize ranh giới phân lớp
# Mục tiêu: Hiểu rõ từng bước xử lý, pipeline, và hoạt động của softmax regression
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

np.random.seed(1)

# 1. Sinh dữ liệu giả lập 3 lớp
means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

print("[DEBUG] X0 shape:", X0.shape, "X1 shape:", X1.shape, "X2 shape:", X2.shape)
print("[DEBUG] X0[:3]:\n", X0[:3])

# Ghép dữ liệu và thêm bias
X = np.concatenate((X0, X1, X2), axis=0).T  # (2, 1500)
X = np.concatenate((np.ones((1, 3*N)), X), axis=0)  # (3, 1500)
C = 3
original_label = np.asarray([0]*N + [1]*N + [2]*N)

print("[DEBUG] X shape:", X.shape)
print("[DEBUG] original_label shape:", original_label.shape)
print("[DEBUG] original_label[:10]:", original_label[:10])

# 2. Chuyển nhãn sang one-hot

def convert_labels(y, C):
    """
    Chuyển nhãn 1D sang ma trận one-hot (C, N)
    EN: Convert 1d label to one-hot matrix (C, N)
    """
    Y = sparse.coo_matrix((np.ones_like(y), (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y

Y = convert_labels(original_label, C)
print("[DEBUG] Y shape:", Y.shape)
print("[DEBUG] Y[:, :5]:\n", Y[:, :5])

# 3. Hàm softmax và softmax ổn định

def softmax(Z):
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis=0)
    return A

def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A

# 4. Hàm loss (cross-entropy)
def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y*np.log(A + 1e-8))

# 5. Hàm gradient
def grad(X, Y, W):
    A = softmax(W.T.dot(X))
    E = A - Y
    return X.dot(E.T)

# 6. Hàm train softmax regression (gradient descent)
def softmax_regression(X, y, W_init, eta, tol=1e-4, max_count=10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    print("[INFO] Bắt đầu train softmax regression...")
    while count < max_count:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta*xi.dot((yi - ai).T)
            count += 1
            if count % 500 == 0:
                loss = cost(X, Y, W_new)
                print(f"[DEBUG] Iter {count}: loss = {loss:.4f}, ||W_new - W_old|| = {np.linalg.norm(W_new - W[-1]):.6f}")
            if count % check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    print(f"[INFO] Dừng sớm ở iter {count}, ||W_new - W_old|| < tol")
                    return W
            W.append(W_new)
    print("[INFO] Train xong, số lần update:", count)
    return W

# 7. Khởi tạo trọng số và train
W_init = np.random.randn(X.shape[0], C)
print("[DEBUG] W_init shape:", W_init.shape)
print("[DEBUG] W_init:\n", W_init)
eta = 0.05
W = softmax_regression(X, original_label, W_init, eta)
print("[DEBUG] W cuối cùng:\n", W[-1])

# 8. Hàm dự đoán
def pred(W, X):
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis=0)

y_pred = pred(W[-1], X)
print("[DEBUG] y_pred[:10]:", y_pred[:10])
print("[DEBUG] original_label[:10]:", original_label[:10])
acc = np.mean(y_pred == original_label)
print(f"[RESULT] Độ chính xác trên tập train: {acc*100:.2f}%")

# 9. Hàm hiển thị dữ liệu và ranh giới phân lớp
def display(X, label):
    X0 = X[:, label == 0]
    X1 = X[:, label == 1]
    X2 = X[:, label == 2]
    plt.plot(X0[0, :], X0[1, :], 'b^', markersize=4, alpha=.8, label='Class 0')
    plt.plot(X1[0, :], X1[1, :], 'go', markersize=4, alpha=.8, label='Class 1')
    plt.plot(X2[0, :], X2[1, :], 'rs', markersize=4, alpha=.8, label='Class 2')
    plt.axis('off')
    plt.legend()
    plt.show()

display(X[1:, :], original_label)

# 10. Visualize ranh giới phân lớp
xm = np.arange(-2, 11, 0.025)
ym = np.arange(-3, 10, 0.025)
xx, yy = np.meshgrid(xm, ym)
xx1 = xx.ravel().reshape(1, xx.size)
yy1 = yy.ravel().reshape(1, yy.size)
XX = np.concatenate((np.ones((1, xx.size)), xx1, yy1), axis=0)

Z = pred(W[-1], XX)
Z = Z.reshape(xx.shape)
CS = plt.contourf(xx, yy, Z, 200, cmap='jet', alpha=.1)
plt.xlim(-2, 11)
plt.ylim(-3, 10)
plt.xticks(())
plt.yticks(())
display(X[1:, :], original_label)
plt.savefig('ex1_debug.png', bbox_inches='tight', dpi=300)
plt.show()

print("\n[INFO] === KẾT THÚC PIPELINE DEBUG SOFTMAX REGRESSION ===\n") 