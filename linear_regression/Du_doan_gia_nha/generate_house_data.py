import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_house_data(n_samples=1000, random_state=42):
    """
    Tạo dataset giả lập về giá nhà với các features:
    - square_feet: Diện tích (sq ft)
    - bedrooms: Số phòng ngủ
    - bathrooms: Số phòng tắm
    - age: Tuổi của nhà (năm)
    - location_score: Điểm đánh giá vị trí (1-10)
    - price: Giá nhà ($)
    """
    np.random.seed(random_state)
    
    # Tạo các features
    square_feet = np.random.normal(2000, 500, n_samples)
    square_feet = np.clip(square_feet, 800, 4000)  # Giới hạn từ 800-4000 sq ft
    
    bedrooms = np.random.poisson(3, n_samples)
    bedrooms = np.clip(bedrooms, 1, 6)  # Giới hạn từ 1-6 phòng ngủ
    
    bathrooms = np.random.poisson(2, n_samples)
    bathrooms = np.clip(bathrooms, 1, 4)  # Giới hạn từ 1-4 phòng tắm
    
    age = np.random.exponential(15, n_samples)
    age = np.clip(age, 0, 50)  # Giới hạn từ 0-50 năm
    
    location_score = np.random.normal(6, 1.5, n_samples)
    location_score = np.clip(location_score, 1, 10)  # Giới hạn từ 1-10
    
    # Tạo giá nhà dựa trên các features với một số noise
    base_price = (
        square_feet * 100 +  # $100 per sq ft
        bedrooms * 15000 +   # $15,000 per bedroom
        bathrooms * 20000 +  # $20,000 per bathroom
        location_score * 25000 -  # $25,000 per location point
        age * 2000           # Giảm $2,000 per year
    )
    
    # Thêm noise ngẫu nhiên
    noise = np.random.normal(0, 20000, n_samples)
    price = base_price + noise
    price = np.clip(price, 100000, 800000)  # Giới hạn từ $100k-$800k
    
    # Tạo DataFrame
    data = pd.DataFrame({
        'square_feet': square_feet,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age': age,
        'location_score': location_score,
        'price': price
    })
    
    return data

def analyze_data(data):
    """Phân tích dữ liệu cơ bản"""
    print("=== THÔNG TIN DATASET ===")
    print(f"Số lượng mẫu: {len(data)}")
    print(f"Số lượng features: {len(data.columns) - 1}")  # Trừ đi target variable
    
    print("\n=== THỐNG KÊ MÔ TẢ ===")
    print(data.describe())
    
    print("\n=== KIỂM TRA MISSING VALUES ===")
    missing_values = data.isnull().sum()
    print(missing_values)
    
    print("\n=== CORRELATION MATRIX ===")
    correlation_matrix = data.corr()
    print(correlation_matrix['price'].sort_values(ascending=False))

def visualize_data(data):
    """Tạo các biểu đồ phân tích dữ liệu"""
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Phân Tích Dữ Liệu House Prices', fontsize=16, fontweight='bold')
    
    # Distribution plots
    features = ['square_feet', 'bedrooms', 'bathrooms', 'age', 'location_score', 'price']
    titles = ['Diện tích (sq ft)', 'Số phòng ngủ', 'Số phòng tắm', 'Tuổi nhà', 'Điểm vị trí', 'Giá nhà ($)']
    
    for i, (feature, title) in enumerate(zip(features, titles)):
        row = i // 3
        col = i % 3
        axes[row, col].hist(data[feature], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row, col].set_title(title)
        axes[row, col].set_xlabel(title)
        axes[row, col].set_ylabel('Tần suất')
    
    plt.tight_layout()
    plt.savefig('house_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Correlation Matrix - House Prices Dataset', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('house_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🚀 Tạo dataset House Prices...")
    
    # Tạo dataset
    house_data = generate_house_data(n_samples=1000)
    
    # Phân tích dữ liệu
    analyze_data(house_data)
    
    # Lưu dataset
    house_data.to_csv('house_prices.csv', index=False)
    print(f"\n✅ Đã lưu dataset vào file: house_prices.csv")
    print(f"📊 Dataset có {len(house_data)} mẫu và {len(house_data.columns)} cột")
    
    # Hiển thị 5 mẫu đầu tiên
    print("\n=== 5 MẪU ĐẦU TIÊN ===")
    print(house_data.head())
    
    # Tạo biểu đồ phân tích
    print("\n📈 Đang tạo biểu đồ phân tích...")
    visualize_data(house_data)
    
    print("\n🎉 Hoàn thành tạo dataset House Prices!") 