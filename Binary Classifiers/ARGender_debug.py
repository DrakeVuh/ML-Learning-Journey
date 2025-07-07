"""
ARGender_debug.py
=================

# PIPELINE TỔNG QUÁT (WORKFLOW OVERVIEW)

1. Chuẩn bị dữ liệu: Xác định đường dẫn, chia train/test, chọn góc nhìn.
2. Tạo danh sách file ảnh cho từng nhóm (nam/nữ).
3. Đọc ảnh, chuyển sang grayscale, vector hóa.
4. Giảm chiều dữ liệu bằng random projection.
5. Xây dựng ma trận dữ liệu X, vector nhãn y.
6. Chuẩn hóa đặc trưng (feature normalization) chỉ trên tập train.
7. Huấn luyện mô hình Logistic Regression.
8. Dự đoán và đánh giá trên tập test.
9. Dự đoán và visualize cho từng ảnh cụ thể.

# Mục tiêu: Hiểu rõ từng bước xử lý dữ liệu, pipeline, và cách các thuật toán hoạt động trong bài toán phân loại giới tính từ ảnh khuôn mặt.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import imageio  # Sửa lỗi: import imageio ở đầu file

# 1. Đặt seed để đảm bảo kết quả reproducible
np.random.seed(1)
print("[INFO] 1. Đặt seed random để đảm bảo kết quả có thể lặp lại (reproducible).\n")

# 2. Đường dẫn tới thư mục chứa ảnh (bạn cần có dữ liệu thực tế để chạy)
path = '../data/AR/'
print(f"[INFO] 2. Đường dẫn dữ liệu: {path}\n")

# 3. Chia tập train và test, chọn các góc nhìn
train_ids = np.arange(1, 26)  # 25 người cho train
test_ids = np.arange(26, 50)  # 24 người cho test
view_ids = np.hstack((np.arange(1, 8), np.arange(14, 21)))  # 14 góc nhìn
print(f"[INFO] 3. Số người train: {len(train_ids)}, test: {len(test_ids)}, số góc nhìn: {len(view_ids)}\n")

# 4. Định nghĩa số chiều dữ liệu
D = 165*120  # Số chiều gốc của ảnh (original dimension)
d = 500      # Số chiều sau khi giảm (reduced dimension)
print(f"[INFO] 4. Số chiều gốc D: {D}, số chiều sau giảm d: {d}\n")

# 5. Tạo ma trận chiếu ngẫu nhiên (random projection matrix)
ProjectionMatrix = np.random.randn(D, d)
print(f"[DEBUG] 5. ProjectionMatrix shape: {ProjectionMatrix.shape} (D x d)\n")

# 6. Hàm tạo danh sách file ảnh cho từng nhóm (nam/nữ)
def build_list_fn(pre, img_ids, view_ids):
    """
    pre: 'M-' cho nam, 'W-' cho nữ
    img_ids: danh sách ID người
    view_ids: danh sách góc nhìn
    Trả về: list đường dẫn file ảnh
    """
    list_fn = []
    for im_id in img_ids:
        for v_id in view_ids:
            fn = path + pre + str(im_id).zfill(3) + '-' + str(v_id).zfill(2) + '.bmp'
            list_fn.append(fn)
    return list_fn

# 7. Chuyển ảnh RGB sang grayscale
# EN: Convert RGB image to grayscale
# VN: Chuyển ảnh màu sang ảnh xám

def rgb2gray(rgb):
    # Công thức: Y' = 0.299 R + 0.587 G + 0.114 B
    return rgb[:,:,0]*.299 + rgb[:, :, 1]*.587 + rgb[:, :, 2]*.114

# 8. Trích xuất đặc trưng từ file ảnh
# EN: Feature extraction from image file
# VN: Trích xuất đặc trưng từ file ảnh

def vectorize_img(filename):
    """
    Đọc ảnh, chuyển sang xám, vector hóa thành 1D
    Trả về: vector 1 x D
    """
    try:
        rgb = imageio.imread(filename)  # Sử dụng imageio.imread
        print(f"[DEBUG] Đọc ảnh thành công: {filename}, shape: {rgb.shape}")
        gray = rgb2gray(rgb)
        print(f"[DEBUG] Chuyển sang xám, shape: {gray.shape}, min: {gray.min()}, max: {gray.max()}")
        im_vec = gray.reshape(1, D)
        print(f"[DEBUG] Vector hóa ảnh, shape: {im_vec.shape}, ví dụ giá trị: {im_vec[0, :5]}")
        return im_vec
    except Exception as e:
        print(f"[ERROR] Không đọc được {filename}: {e}")
        return np.zeros((1, D))

# 9. Xây dựng ma trận dữ liệu từ danh sách ảnh
# EN: Build data matrix from image list
# VN: Xây dựng ma trận dữ liệu từ danh sách ảnh

def build_data_matrix(img_ids, view_ids):
    total_imgs = img_ids.shape[0]*view_ids.shape[0]*2
    print(f"[INFO] Xây dựng ma trận dữ liệu: {total_imgs} ảnh (nam + nữ)")
    X_full = np.zeros((total_imgs, D))
    y = np.hstack((np.zeros((total_imgs//2, )), np.ones((total_imgs//2, ))))
    list_fn_m = build_list_fn('M-', img_ids, view_ids)
    list_fn_w = build_list_fn('W-', img_ids, view_ids)
    list_fn = list_fn_m + list_fn_w
    for i in range(len(list_fn)):
        X_full[i, :] = vectorize_img(list_fn[i])
        if i % 20 == 0:
            print(f"[DEBUG] Đã xử lý {i+1}/{len(list_fn)} ảnh")
    print(f"[DEBUG] X_full shape: {X_full.shape}, y shape: {y.shape}")
    # Giảm chiều dữ liệu
    X = np.dot(X_full, ProjectionMatrix)
    print(f"[DEBUG] Sau giảm chiều: X shape: {X.shape}, ví dụ X[0,:5]: {X[0,:5]}")
    return (X, y)

# --- MAIN PIPELINE ---
print("\n[INFO] === BẮT ĐẦU PIPELINE ===\n")

# 10. Xây dựng dữ liệu train
print("[INFO] Xây dựng dữ liệu train...")
(X_train_full, y_train) = build_data_matrix(train_ids, view_ids)
print(f"[DEBUG] X_train_full shape: {X_train_full.shape}, y_train shape: {y_train.shape}")
print(f"[DEBUG] Nhãn train (0: nam, 1: nữ), ví dụ: {y_train[:10]}")

# 11. Tính mean và var trên tập train
x_mean = X_train_full.mean(axis=0)
x_var = X_train_full.var(axis=0)
print(f"[DEBUG] x_mean shape: {x_mean.shape}, x_var shape: {x_var.shape}")
print(f"[DEBUG] x_mean ví dụ: {x_mean[:5]}, x_var ví dụ: {x_var[:5]}")

# 12. Chuẩn hóa đặc trưng
# EN: Feature normalization
# VN: Chuẩn hóa đặc trưng

def feature_extraction(X):
    X_norm = (X - x_mean) / (x_var + 1e-8)
    print(f"[DEBUG] Đã chuẩn hóa đặc trưng, shape: {X_norm.shape}, min: {X_norm.min()}, max: {X_norm.max()}")
    return X_norm

X_train = feature_extraction(X_train_full)
X_train_full = None  # Giải phóng bộ nhớ

# 13. Xây dựng dữ liệu test
print("\n[INFO] Xây dựng dữ liệu test...")
(X_test_full, y_test) = build_data_matrix(test_ids, view_ids)
print(f"[DEBUG] X_test_full shape: {X_test_full.shape}, y_test shape: {y_test.shape}")
print(f"[DEBUG] Nhãn test (0: nam, 1: nữ), ví dụ: {y_test[:10]}")
X_test = feature_extraction(X_test_full)
X_test_full = None

# 14. Huấn luyện Logistic Regression
print("\n[INFO] Huấn luyện Logistic Regression...")
logreg = linear_model.LogisticRegression(C=1e5, max_iter=200)
logreg.fit(X_train, y_train)
print("[DEBUG] Đã huấn luyện xong mô hình.")

# 15. Dự đoán trên tập test
print("\n[INFO] Dự đoán trên tập test...")
y_pred = logreg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[RESULT] Độ chính xác trên tập test: {acc*100:.2f}%\n")

# 16. In ra một số dự đoán mẫu
for i in range(5):
    print(f"[DEBUG] Test sample {i}: Nhãn thực tế = {y_test[i]}, Dự đoán = {y_pred[i]}, Xác suất = {logreg.predict_proba([X_test[i]])}")

# 17. Hàm trích xuất đặc trưng từ file ảnh cho dự đoán đơn lẻ
# EN: Feature extraction for single file
# VN: Trích xuất đặc trưng cho một file ảnh

def feature_extraction_fn(fn):
    im = vectorize_img(fn)
    im1 = np.dot(im, ProjectionMatrix)
    return feature_extraction(im1)

# 18. Hàm hiển thị kết quả dự đoán cho một ảnh
# EN: Display prediction result for an image
# VN: Hiển thị kết quả dự đoán cho một ảnh

def display_result(fn):
    x1 = feature_extraction_fn(fn)
    p1 = logreg.predict_proba(x1)
    pred = logreg.predict(x1)[0]
    print(f"[DEBUG] File: {fn}")
    print(f"[DEBUG] Xác suất dự đoán: {p1}")
    print(f"[DEBUG] Nhãn dự đoán: {pred} (0: nam, 1: nữ)")
    try:
        rgb = imageio.imread(fn)  # Sử dụng imageio.imread
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.axis('off')
        plt.imshow(rgb)
        plt.title('Input Image')
        plt.subplot(1,2,2)
        plt.barh([0, 1], p1[0], align='center', alpha=0.9)
        plt.yticks([0, 1], ('man', 'woman'))
        plt.xlim([0,1])
        plt.title('Predicted Probabilities')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[ERROR] Không thể hiển thị ảnh: {e}")

# 19. Ví dụ sử dụng (bạn cần có file ảnh thực tế để chạy)
fn1 = path + 'M-036-18.bmp'
fn2 = path + 'W-045-01.bmp'
fn3 = path + 'M-048-01.bmp'
fn4 = path + 'W-027-02.bmp'

# Uncomment các dòng dưới nếu bạn có dữ liệu thực tế
# display_result(fn1)
# display_result(fn2)
# display_result(fn3)
# display_result(fn4)

print("\n[INFO] === KẾT THÚC PIPELINE DEBUG ===\n") 