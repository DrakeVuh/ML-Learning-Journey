from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(22)

def sigmoid(s):
    return 1/(1 + np.exp(-s))

def logistic_sigmoid_regression_debug(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    debug_limit = 50  # Số lần in chi tiết đầu tiên
    print(f"\n=== BẮT ĐẦU HUẤN LUYỆN LOGISTIC REGRESSION ===")
    print(f"N = {N} (số mẫu), d = {d} (số chiều)")
    print(f"w_init: {w_init.flatten()}")
    iteration = 0
    while count < max_count:
        iteration += 1
        print(f"\n--- LẦN LẶP {iteration} ---")
        mix_id = np.random.permutation(N)
        print(f"Thứ tự xử lý mẫu: {mix_id}")
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))[0, 0]
            w_before = w[-1].flatten()
            w_new = w[-1] + eta*(yi - zi)*xi
            w_after = w_new.flatten()
            count += 1
            # Loss (log-loss) cho mẫu này
            loss = - (yi * np.log(zi + 1e-8) + (1 - yi) * np.log(1 - zi + 1e-8))
            if count <= debug_limit:
                print(f"Mẫu {i}: xi = {xi.flatten()}, yi = {yi}")
                print(f"  Dự đoán sigmoid: {zi:.4f}")
                print(f"  Loss: {float(loss):.6f}")
                print(f"  w trước cập nhật: {w_before}")
                print(f"  w sau cập nhật:   {w_after}")
                if zi > 0.8:
                    print("  → Model rất tự tin là lớp 1 (sigmoid gần 1)")
                elif zi < 0.2:
                    print("  → Model rất tự tin là lớp 0 (sigmoid gần 0)")
                else:
                    print("  → Model chưa chắc chắn (sigmoid ≈ 0.5)")
            elif count == debug_limit + 1:
                print(f"... (Đã vượt quá {debug_limit} lần cập nhật, dừng in chi tiết) ...")
            if count % check_w_after == 0:
                norm_diff = np.linalg.norm(w_new - w[-check_w_after])
                print(f"  Kiểm tra hội tụ sau {check_w_after} lần cập nhật: ||w_new - w_old|| = {norm_diff:.6f}")
                if norm_diff < tol:
                    print(f"\n🎉 HỘI TỤ SAU {count} LẦN CẬP NHẬT!")
                    return w
            w.append(w_new)
    print(f"\n⚠️ ĐẠT GIỚI HẠN SỐ LẦN LẶP (max_count = {max_count})")
    return w

# ====== TẠO DỮ LIỆU ======
means = [[2, 2], [4, 2]]
cov = [[.7, 0], [0, .7]]
N = 20
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)

print("\n=== THÔNG TIN DỮ LIỆU ===")
print(f"X0 shape: {X0.shape}, X1 shape: {X1.shape}")
print(f"X0 (5 mẫu đầu):\n{X0[:5]}")
print(f"X1 (5 mẫu đầu):\n{X1[:5]}")

X = np.concatenate((X0, X1), axis = 0).T
print(f"\nX shape sau khi ghép: {X.shape}")

y = np.concatenate((np.zeros((1, N)), np.ones((1, N))), axis = 1).T
print(f"y shape: {y.shape}")
print(f"y (10 phần tử đầu): {y[:10].flatten()}")

# Xbar 
X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)
print(f"X shape sau khi thêm bias: {X.shape}")
print(f"X (5 cột đầu):\n{X[:, :5]}")

eta = .05 
d = X.shape[0]
w_init = np.random.randn(d, 1)

w = logistic_sigmoid_regression_debug(X, y, w_init, eta, tol = 1e-4, max_count= 10000)
print(f"\n=== KẾT QUẢ CUỐI CÙNG ===")
print(f"w cuối cùng: {w[-1].flatten()}")
print(f"Số lần cập nhật: {len(w)-1}") 