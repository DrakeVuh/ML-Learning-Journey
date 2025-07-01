import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(2)

print("=== BƯỚC 1: TẠO DỮ LIỆU ===")
print("=== STEP 1: CREATE DATA ===")

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10

print(f"Tạo {N} mẫu cho mỗi lớp / Create {N} samples for each class")
print(f"Lớp 0: mean = {means[0]}, cov = {cov}")
print(f"Lớp 1: mean = {means[1]}, cov = {cov}")

X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

print(f"\nX0 shape: {X0.shape} (2 features, {N} samples)")
print(f"X1 shape: {X1.shape} (2 features, {N} samples)")

print(f"\nX0 (5 mẫu đầu):")
print(X0[:, :5])
print(f"\nX1 (5 mẫu đầu):")
print(X1[:, :5])

print("\n=== BƯỚC 2: GHÉP DỮ LIỆU ===")
print("=== STEP 2: COMBINE DATA ===")

X = np.concatenate((X0, X1), axis = 1)
print(f"X shape sau khi ghép: {X.shape}")
print(f"X (5 cột đầu):")
print(X[:, :5])

y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)
print(f"\ny shape: {y.shape}")
print(f"y (10 phần tử đầu): {y[0, :10]}")

print("\n=== BƯỚC 3: THÊM BIAS TERM ===")
print("=== STEP 3: ADD BIAS TERM ===")

X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)
print(f"X shape sau khi thêm bias: {X.shape}")
print(f"X (5 cột đầu):")
print(X[:, :5])

print("\n=== BƯỚC 4: ĐỊNH NGHĨA CÁC HÀM ===")
print("=== STEP 4: DEFINE FUNCTIONS ===")

def h(w, x):    
    result = np.sign(np.dot(w.T, x))
    print(f"h(w, x) = sign({np.dot(w.T, x)[0,0]:.3f}) = {result[0,0]}")
    return result

def h_silent(w, x):    
    """Hàm h không in ra màn hình / Silent version of h function"""
    result = np.sign(np.dot(w.T, x))
    return result

def has_converged(X, y, w):    
    print("🔍 Kiểm tra hội tụ... / Checking convergence...")
    predictions = h_silent(w, X)  # Không in ra từng mẫu
    is_equal = np.array_equal(predictions, y)
    print(f"Tất cả dự đoán đúng? {is_equal}")
    if is_equal:
        print("✅ TẤT CẢ MẪU ĐƯỢC PHÂN LOẠI ĐÚNG!")
    else:
        print("❌ VẪN CÓ MẪU BỊ SAI, TIẾP TỤC HỌC...")
    return is_equal

def perceptron_debug(X, y, w_init):
    print("\n=== BẮT ĐẦU THUẬT TOÁN PERCEPTRON ===")
    print("=== START PERCEPTRON ALGORITHM ===")
    
    w = [w_init]
    N = X.shape[1]
    d = X.shape[0]
    mis_points = []
    
    print(f"N = {N} (số mẫu), d = {d} (số chiều)")
    print(f"w_init: {w_init.flatten()}")
    
    iteration = 0
    while True:
        iteration += 1
        print(f"\n--- LẦN LẶP {iteration} ---")
        print(f"--- ITERATION {iteration} ---")
        
        # mix data 
        mix_id = np.random.permutation(N)
        print(f"Thứ tự xử lý mẫu: {mix_id}")
        
        mis_count = 0
        for i in range(N):
            xi = X[:, mix_id[i]].reshape(d, 1)
            yi = y[0, mix_id[i]]
            
            print(f"\nMẫu {mix_id[i]}: xi = {xi.flatten()}, yi = {yi}")
            
            prediction = h(w[-1], xi)[0]
            if prediction != yi:
                print(f"❌ PHÂN LOẠI SAI! Dự đoán: {prediction}, Thực tế: {yi}")
                mis_points.append(mix_id[i])
                mis_count += 1
                
                w_new = w[-1] + yi*xi 
                print(f"Cập nhật w: {w[-1].flatten()} + {yi}*{xi.flatten()} = {w_new.flatten()}")
                w.append(w_new)
            else:
                print(f"✅ PHÂN LOẠI ĐÚNG! Dự đoán: {prediction}, Thực tế: {yi}")
        
        print(f"Số mẫu sai trong lần lặp này: {mis_count}")
        
        if has_converged(X, y, w[-1]):
            print(f"\n🎉 HỘI TỤ SAU {iteration} LẦN LẶP!")
            break
            
        if iteration > 10:  # Tránh vòng lặp vô hạn
            print("⚠️ Dừng sau 10 lần lặp để tránh vòng lặp vô hạn")
            break
    
    return (w, mis_points)

print("\n=== BƯỚC 5: CHẠY THUẬT TOÁN ===")
print("=== STEP 5: RUN ALGORITHM ===")

d = X.shape[0]
w_init = np.random.randn(d, 1)
(w, m) = perceptron_debug(X, y, w_init)

print(f"\n=== KẾT QUẢ CUỐI CÙNG ===")
print(f"=== FINAL RESULT ===")
print(f"w cuối cùng: {w[-1].flatten()}")
print(f"Số lần cập nhật: {len(w)-1}")
print(f"Các mẫu bị sai: {m}") 