import numpy as np
import pandas as pd

# Tạo dữ liệu mẫu
np.random.seed(42)  # Để kết quả có thể tái lập
n_samples = 500

# Tạo các đặc trưng với mối quan hệ thực tế
data = {
    'study_hours_per_week': np.random.uniform(5, 25, n_samples),  # Số giờ học mỗi tuần (5-25h)
    'group_study_per_month': np.random.randint(0, 8, n_samples),  # Số buổi học nhóm mỗi tháng (0-7)
    'extra_classes_per_month': np.random.randint(0, 6, n_samples),  # Số buổi học thêm mỗi tháng (0-5)
    'sleep_hours_per_day': np.random.uniform(5, 9, n_samples),  # Số giờ ngủ mỗi ngày (5-9h)
    'extracurricular_per_month': np.random.randint(0, 5, n_samples)  # Số lần ngoại khóa mỗi tháng (0-4)
}

# Tạo DataFrame
df = pd.DataFrame(data)

# Tạo điểm số dựa trên các đặc trưng với mối quan hệ thực tế
df['final_score'] = (
    # Tác động tích cực từ thời gian học
    df['study_hours_per_week'] * 0.15 +  # Mỗi giờ học tăng 0.15 điểm
    
    # Tác động tích cực từ học nhóm
    df['group_study_per_month'] * 0.3 +  # Mỗi buổi học nhóm tăng 0.3 điểm
    
    # Tác động tích cực từ học thêm
    df['extra_classes_per_month'] * 0.25 +  # Mỗi buổi học thêm tăng 0.25 điểm
    
    # Tác động tích cực từ giấc ngủ (có điểm tối ưu)
    np.where(df['sleep_hours_per_day'] >= 7, 
             (df['sleep_hours_per_day'] - 7) * 0.2,  # Ngủ đủ giấc tốt
             (df['sleep_hours_per_day'] - 7) * 0.1) +  # Ngủ ít ảnh hưởng tiêu cực
    
    # Tác động nhỏ từ hoạt động ngoại khóa
    df['extracurricular_per_month'] * 0.1 +  # Mỗi lần ngoại khóa tăng 0.1 điểm
    
    # Điểm cơ bản
    4.0 +
    
    # Nhiễu ngẫu nhiên
    np.random.normal(0, 0.5, n_samples)
)

# Đảm bảo điểm số trong khoảng 0-10
df['final_score'] = np.clip(df['final_score'], 0, 10)

# Làm tròn các giá trị
df['study_hours_per_week'] = df['study_hours_per_week'].round(1)
df['sleep_hours_per_day'] = df['sleep_hours_per_day'].round(1)
df['final_score'] = df['final_score'].round(2)

# Sắp xếp lại các cột
df = df[['study_hours_per_week', 'group_study_per_month', 'extra_classes_per_month', 
         'sleep_hours_per_day', 'extracurricular_per_month', 'final_score']]

# Lưu vào file CSV
df.to_csv('student_scores.csv', index=False)

print("Đã tạo file student_scores.csv thành công!")
print(f"Số lượng mẫu: {len(df)}")
print("\nThông tin về dữ liệu:")
print(df.describe())
print("\nMẫu dữ liệu (5 mẫu đầu):")
print(df.head())
print("\nMối quan hệ giữa thời gian học và điểm số:")
print(f"Trung bình điểm của sinh viên học < 10h/tuần: {df[df['study_hours_per_week'] < 10]['final_score'].mean():.2f}")
print(f"Trung bình điểm của sinh viên học 10-15h/tuần: {df[(df['study_hours_per_week'] >= 10) & (df['study_hours_per_week'] < 15)]['final_score'].mean():.2f}")
print(f"Trung bình điểm của sinh viên học >= 15h/tuần: {df[df['study_hours_per_week'] >= 15]['final_score'].mean():.2f}") 