# ========================================
# SALARY PREDICTION WITH LINEAR REGRESSION
# DỰ ĐOÁN LƯƠNG VỚI HỒI QUY TUYẾN TÍNH
# ========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ========================================
# PHẦN 1: TẠO DỮ LIỆU LƯƠNG / PART 1: CREATE SALARY DATA
# ========================================

def create_salary_data(n_samples=200, noise_level=0.1):
    """
    Tạo dữ liệu lương dựa trên năm kinh nghiệm
    Create salary data based on years of experience
    
    Args:
        n_samples (int): Số lượng mẫu / Number of samples
        noise_level (float): Mức độ nhiễu / Noise level
    
    Returns:
        pd.DataFrame: Dữ liệu lương / Salary data
    """
    np.random.seed(42)
    
    # Tạo năm kinh nghiệm (0-20 năm) / Create years of experience (0-20 years)
    years_experience = np.random.uniform(0, 20, n_samples)
    
    # Tạo lương cơ bản với công thức thực tế / Create base salary with realistic formula
    # Lương = 30000 + 5000 * năm kinh nghiệm + nhiễu
    # Salary = 30000 + 5000 * years + noise
    base_salary = 30000  # Lương khởi điểm (USD) / Starting salary (USD)
    salary_increase = 5000  # Tăng lương mỗi năm (USD) / Salary increase per year (USD)
    
    # Tạo lương với nhiễu / Create salary with noise
    noise = np.random.normal(0, base_salary * noise_level, n_samples)
    salary = base_salary + salary_increase * years_experience + noise
    
    # Đảm bảo lương không âm / Ensure salary is not negative
    salary = np.maximum(salary, 20000)
    
    # Tạo DataFrame
    data = pd.DataFrame({
        'years_experience': years_experience,
        'salary': salary
    })
    
    return data

def explore_salary_data(data):
    """Khám phá dữ liệu lương / Explore salary data"""
    print("=== KHÁM PHÁ DỮ LIỆU LƯƠNG ===")
    print("=== EXPLORE SALARY DATA ===")
    
    print(f"📊 Kích thước dữ liệu / Data shape: {data.shape}")
    print(f"📋 5 mẫu đầu tiên / First 5 samples:")
    print(data.head())
    
    print(f"\n📈 Thống kê mô tả / Descriptive statistics:")
    print(data.describe())
    
    # Vẽ biểu đồ phân phối / Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram năm kinh nghiệm / Years of experience histogram
    ax1.hist(data['years_experience'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Năm kinh nghiệm / Years of Experience')
    ax1.set_ylabel('Tần suất / Frequency')
    ax1.set_title('Phân phối năm kinh nghiệm\nDistribution of Years of Experience')
    ax1.grid(True, alpha=0.3)
    
    # Histogram lương / Salary histogram
    ax2.hist(data['salary'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Lương (USD) / Salary (USD)')
    ax2.set_ylabel('Tần suất / Frequency')
    ax2.set_title('Phân phối lương\nDistribution of Salary')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('salary_data_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Vẽ scatter plot / Plot scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(data['years_experience'], data['salary'], alpha=0.6, color='blue')
    plt.xlabel('Năm kinh nghiệm / Years of Experience')
    plt.ylabel('Lương (USD) / Salary (USD)')
    plt.title('Mối quan hệ: Lương vs Năm kinh nghiệm\nRelationship: Salary vs Years of Experience')
    plt.grid(True, alpha=0.3)
    plt.savefig('salary_vs_experience.png', dpi=300, bbox_inches='tight')
    plt.show()

# ========================================
# PHẦN 2: HUẤN LUYỆN MÔ HÌNH / PART 2: TRAIN MODEL
# ========================================

def train_salary_model(data):
    """
    Huấn luyện mô hình Linear Regression cho dự đoán lương
    Train Linear Regression model for salary prediction
    """
    print("\n=== HUẤN LUYỆN MÔ HÌNH LINEAR REGRESSION ===")
    print("=== TRAIN LINEAR REGRESSION MODEL ===")
    
    # Chuẩn bị dữ liệu / Prepare data
    X = data[['years_experience']].values
    y = data['salary'].values
    
    # Chia dữ liệu / Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"🔢 Kích thước dữ liệu / Data sizes:")
    print(f"   Training: {X_train.shape[0]} mẫu / samples")
    print(f"   Testing: {X_test.shape[0]} mẫu / samples")
    
    # Huấn luyện mô hình / Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Dự đoán / Predict
    y_pred = model.predict(X_test)
    
    return model, X_train, X_test, y_train, y_test, y_pred

# ========================================
# PHẦN 3: SO SÁNH HIỆU SUẤT VỚI CÁC METRICS KHÁC NHAU
# PART 3: COMPARE PERFORMANCE WITH DIFFERENT METRICS
# ========================================

def compare_performance_metrics(y_test, y_pred):
    """
    So sánh hiệu suất với các metrics khác nhau
    Compare performance with different metrics
    """
    print("\n=== SO SÁNH HIỆU SUẤT VỚI CÁC METRICS KHÁC NHAU ===")
    print("=== COMPARE PERFORMANCE WITH DIFFERENT METRICS ===")
    
    # Tính các metrics / Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Tính MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("📊 Các metrics đánh giá mô hình / Model evaluation metrics:")
    print(f"   Mean Squared Error (MSE): {mse:,.2f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:,.2f}")
    print(f"   Mean Absolute Error (MAE): {mae:,.2f}")
    print(f"   Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"   R² Score: {r2:.4f}")
    
    # Giải thích ý nghĩa / Explain meanings
    print(f"\n💡 Giải thích ý nghĩa / Interpretation:")
    print(f"   - MSE: Sai số bình phương trung bình (càng thấp càng tốt)")
    print(f"     MSE: Mean squared error (lower is better)")
    print(f"   - RMSE: Sai số trung bình (cùng đơn vị với lương)")
    print(f"     RMSE: Root mean squared error (same unit as salary)")
    print(f"   - MAE: Sai số tuyệt đối trung bình (dễ hiểu hơn)")
    print(f"     MAE: Mean absolute error (easier to understand)")
    print(f"   - MAPE: Sai số phần trăm trung bình (so sánh tương đối)")
    print(f"     MAPE: Mean absolute percentage error (relative comparison)")
    print(f"   - R²: Hệ số xác định (0-1, càng cao càng tốt)")
    print(f"     R²: Coefficient of determination (0-1, higher is better)")
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def visualize_model_results(model, data, X_test, y_test, y_pred):
    """Trực quan hóa kết quả mô hình / Visualize model results"""
    print("\n=== TRỰC QUAN HÓA KẾT QUẢ MÔ HÌNH ===")
    print("=== VISUALIZE MODEL RESULTS ===")
    
    # Tạo figure với 2 subplot / Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Dữ liệu và đường hồi quy / Plot 1: Data and regression line
    ax1.scatter(data['years_experience'], data['salary'], alpha=0.6, 
                label='Dữ liệu thực tế / Actual data', color='blue')
    
    # Vẽ đường hồi quy / Draw regression line
    years_range = np.linspace(data['years_experience'].min(), 
                             data['years_experience'].max(), 100)
    salary_pred = model.predict(years_range.reshape(-1, 1))
    ax1.plot(years_range, salary_pred, 'r-', linewidth=3, 
             label='Đường hồi quy / Regression line')
    
    ax1.set_xlabel('Năm kinh nghiệm / Years of Experience')
    ax1.set_ylabel('Lương (USD) / Salary (USD)')
    ax1.set_title('Hồi quy tuyến tính: Lương vs Năm kinh nghiệm\nLinear Regression: Salary vs Years of Experience')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: So sánh dự đoán vs thực tế / Plot 2: Predicted vs Actual
    ax2.scatter(y_test, y_pred, alpha=0.6, color='green')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Đường hoàn hảo / Perfect line')
    ax2.set_xlabel('Lương thực tế / Actual Salary')
    ax2.set_ylabel('Lương dự đoán / Predicted Salary')
    ax2.set_title('Dự đoán vs Thực tế\nPredicted vs Actual')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('salary_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Hiển thị thông tin mô hình / Display model information
    print(f"\n📈 Thông tin mô hình / Model information:")
    print(f"   Hệ số góc / Slope (β₁): {model.coef_[0]:.2f}")
    print(f"   Hệ số tự do / Intercept (β₀): {model.intercept_:.2f}")
    print(f"   Công thức: Lương = {model.intercept_:.0f} + {model.coef_[0]:.0f} × Năm kinh nghiệm")
    print(f"   Formula: Salary = {model.intercept_:.0f} + {model.coef_[0]:.0f} × Years of Experience")

# ========================================
# PHẦN 4: THỬ NGHIỆM VỚI DỮ LIỆU CÓ NHIỄU KHÁC NHAU
# PART 4: EXPERIMENT WITH DIFFERENT NOISE LEVELS
# ========================================

def experiment_with_noise_levels():
    """
    Thử nghiệm với các mức nhiễu khác nhau
    Experiment with different noise levels
    """
    print("\n=== THỬ NGHIỆM VỚI MỨC NHIỄU KHÁC NHAU ===")
    print("=== EXPERIMENT WITH DIFFERENT NOISE LEVELS ===")
    
    noise_levels = [0.05, 0.1, 0.2, 0.3]  # 5%, 10%, 20%, 30%
    results = []
    
    for noise in noise_levels:
        print(f"\n🔬 Thử nghiệm với nhiễu {noise*100}% / Testing with {noise*100}% noise")
        
        # Tạo dữ liệu với mức nhiễu khác nhau / Create data with different noise
        data = create_salary_data(n_samples=200, noise_level=noise)
        
        # Huấn luyện mô hình / Train model
        model, X_train, X_test, y_train, y_test, y_pred = train_salary_model(data)
        
        # Đánh giá hiệu suất / Evaluate performance
        metrics = compare_performance_metrics(y_test, y_pred)
        metrics['noise_level'] = noise
        results.append(metrics)
    
    # So sánh kết quả / Compare results
    print(f"\n📊 SO SÁNH KẾT QUẢ THEO MỨC NHIỄU ===")
    print(f"📊 COMPARISON BY NOISE LEVEL ===")
    
    results_df = pd.DataFrame(results)
    print(results_df.round(4))
    
    # Vẽ biểu đồ so sánh / Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # R² Score
    ax1.plot(results_df['noise_level']*100, results_df['R2'], 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Mức nhiễu (%) / Noise Level (%)')
    ax1.set_ylabel('R² Score')
    ax1.set_title('R² Score vs Noise Level')
    ax1.grid(True, alpha=0.3)
    
    # RMSE
    ax2.plot(results_df['noise_level']*100, results_df['RMSE'], 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Mức nhiễu (%) / Noise Level (%)')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs Noise Level')
    ax2.grid(True, alpha=0.3)
    
    # MAE
    ax3.plot(results_df['noise_level']*100, results_df['MAE'], 'go-', linewidth=2, markersize=8)
    ax3.set_xlabel('Mức nhiễu (%) / Noise Level (%)')
    ax3.set_ylabel('MAE')
    ax3.set_title('MAE vs Noise Level')
    ax3.grid(True, alpha=0.3)
    
    # MAPE
    ax4.plot(results_df['noise_level']*100, results_df['MAPE'], 'mo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Mức nhiễu (%) / Noise Level (%)')
    ax4.set_ylabel('MAPE (%)')
    ax4.set_title('MAPE vs Noise Level')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('noise_experiment_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df

def practical_predictions(model):
    """Dự đoán thực tế / Practical predictions"""
    print("\n=== DỰ ĐOÁN THỰC TẾ ===")
    print("=== PRACTICAL PREDICTIONS ===")
    
    # Dự đoán cho các năm kinh nghiệm cụ thể / Predict for specific years
    test_years = [0, 2, 5, 8, 10, 15, 20]
    
    print("💰 Dự đoán lương cho các năm kinh nghiệm khác nhau:")
    print("   Salary prediction for different years of experience:")
    
    for years in test_years:
        predicted_salary = model.predict([[years]])[0]
        print(f"   {years} năm kinh nghiệm → Lương dự đoán: ${predicted_salary:,.0f}")
        print(f"   {years} years experience → Predicted salary: ${predicted_salary:,.0f}")

# ========================================
# HÀM CHÍNH / MAIN FUNCTION
# ========================================

def main():
    """Hàm chính / Main function"""
    print("🚀 BẮT ĐẦU DỰ ĐOÁN LƯƠNG VỚI LINEAR REGRESSION")
    print("🚀 STARTING SALARY PREDICTION WITH LINEAR REGRESSION")
    
    # 1. Tạo dữ liệu lương / Create salary data
    print("\n" + "="*60)
    data = create_salary_data(n_samples=200, noise_level=0.1)
    explore_salary_data(data)
    
    # 2. Huấn luyện mô hình / Train model
    print("\n" + "="*60)
    model, X_train, X_test, y_train, y_test, y_pred = train_salary_model(data)
    
    # 3. So sánh hiệu suất / Compare performance
    print("\n" + "="*60)
    metrics = compare_performance_metrics(y_test, y_pred)
    visualize_model_results(model, data, X_test, y_test, y_pred)
    
    # 4. Dự đoán thực tế / Practical predictions
    print("\n" + "="*60)
    practical_predictions(model)
    
    # 5. Thử nghiệm với nhiễu khác nhau / Experiment with different noise
    print("\n" + "="*60)
    noise_results = experiment_with_noise_levels()
    
    print(f"\n🎉 HOÀN THÀNH! Tất cả biểu đồ đã được lưu.")
    print(f"🎉 COMPLETED! All charts have been saved.")

if __name__ == "__main__":
    main() 