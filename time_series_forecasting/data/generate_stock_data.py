import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def generate_stock_data(n_days=1000, start_date='2020-01-01', random_state=42):
    """
    Tạo dữ liệu giá cổ phiếu giả lập với các đặc điểm thực tế:
    - Trend (xu hướng tăng/giảm)
    - Seasonality (tính chu kỳ)
    - Volatility (biến động)
    - Volume patterns
    """
    np.random.seed(random_state)
    
    # Tạo timeline
    start = datetime.strptime(start_date, '%Y-%m-%d')
    dates = [start + timedelta(days=i) for i in range(n_days)]
    
    # Base price
    base_price = 100.0
    
    # 1. Trend component (xu hướng dài hạn)
    trend = np.linspace(0, 50, n_days)  # Tăng 50 điểm trong n_days ngày
    
    # 2. Seasonality component (tính chu kỳ)
    # Weekly pattern (thứ 2 thường thấp, thứ 6 thường cao)
    day_of_week = np.array([d.weekday() for d in dates])
    weekly_pattern = np.where(day_of_week == 0, -2,  # Thứ 2 thấp
                    np.where(day_of_week == 4, 2,    # Thứ 6 cao
                    np.where(day_of_week == 5, -1,   # Thứ 7 thấp
                    np.where(day_of_week == 6, -1.5, # Chủ nhật thấp
                    0))))  # Các ngày khác bình thường
    
    # Monthly pattern (tháng 1, 7 thường cao)
    month = np.array([d.month for d in dates])
    monthly_pattern = np.where((month == 1) | (month == 7), 3, 0)
    
    # 3. Random walk component (biến động ngẫu nhiên)
    random_walk = np.cumsum(np.random.normal(0, 0.5, n_days))
    
    # 4. Volatility clustering (biến động theo cụm)
    volatility = np.random.gamma(2, 0.5, n_days)
    volatility_shock = np.random.normal(0, 1, n_days) * volatility
    
    # 5. Combine all components
    price = base_price + trend + weekly_pattern + monthly_pattern + random_walk + volatility_shock
    
    # Đảm bảo giá không âm
    price = np.maximum(price, 10)
    
    # Tạo OHLC data
    data = []
    for i, (date, close_price) in enumerate(zip(dates, price)):
        # Tạo Open, High, Low từ Close
        daily_volatility = np.random.uniform(0.5, 2.0)
        
        open_price = close_price + np.random.normal(0, daily_volatility)
        high_price = max(open_price, close_price) + np.random.uniform(0, daily_volatility)
        low_price = min(open_price, close_price) - np.random.uniform(0, daily_volatility)
        
        # Volume (có pattern theo ngày trong tuần)
        base_volume = 1000000
        volume_factor = np.where(day_of_week[i] in [0, 4], 1.5,  # Thứ 2, 6 cao
                       np.where(day_of_week[i] in [5, 6], 0.5,   # Cuối tuần thấp
                       1.0))  # Các ngày khác bình thường
        
        volume = int(base_volume * volume_factor * np.random.uniform(0.8, 1.2))
        
        data.append({
            'date': date,
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    return df

def add_technical_indicators(df):
    """
    Thêm các chỉ báo kỹ thuật cơ bản
    """
    # Moving Averages
    df['MA_5'] = df['close'].rolling(window=5).mean()
    df['MA_20'] = df['close'].rolling(window=20).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    
    # Price changes
    df['price_change'] = df['close'].pct_change()
    df['price_change_5d'] = df['close'].pct_change(periods=5)
    
    # Volatility
    df['volatility_5d'] = df['price_change'].rolling(window=5).std()
    df['volatility_20d'] = df['price_change'].rolling(window=20).std()
    
    # Volume indicators
    df['volume_MA_5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_MA_5']
    
    # Price momentum
    df['momentum_5d'] = df['close'] - df['close'].shift(5)
    df['momentum_20d'] = df['close'] - df['close'].shift(20)
    
    # RSI (Relative Strength Index) - simplified
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def add_time_features(df):
    """
    Thêm các features dựa trên thời gian
    """
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear
    
    # Cyclical encoding cho day_of_week và month
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def create_target_variable(df, forecast_days=7):
    """
    Tạo target variable: giá sau forecast_days ngày
    """
    df[f'price_{forecast_days}d_ahead'] = df['close'].shift(-forecast_days)
    df[f'return_{forecast_days}d_ahead'] = df['close'].pct_change(periods=forecast_days).shift(-forecast_days)
    
    return df

def analyze_data(df):
    """
    Phân tích dữ liệu cơ bản
    """
    print("=== THÔNG TIN DATASET ===")
    print(f"Số ngày: {len(df)}")
    print(f"Thời gian: {df.index[0].strftime('%Y-%m-%d')} đến {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Số features: {len(df.columns)}")
    
    print("\n=== THỐNG KÊ MÔ TẢ ===")
    print(df[['open', 'high', 'low', 'close', 'volume']].describe())
    
    print("\n=== KIỂM TRA MISSING VALUES ===")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Correlation với target
    if 'price_7d_ahead' in df.columns:
        correlation = df.corr()['price_7d_ahead'].sort_values(ascending=False)
        print("\n=== CORRELATION VỚI TARGET (price_7d_ahead) ===")
        print(correlation.head(10))

def visualize_data(df):
    """
    Tạo các biểu đồ phân tích
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Phân Tích Dữ Liệu Giá Cổ Phiếu', fontsize=16, fontweight='bold')
    
    # 1. Price chart
    axes[0, 0].plot(df.index, df['close'], label='Close Price', linewidth=1)
    axes[0, 0].plot(df.index, df['MA_20'], label='MA 20', alpha=0.7)
    axes[0, 0].plot(df.index, df['MA_50'], label='MA 50', alpha=0.7)
    axes[0, 0].set_title('Giá đóng cửa và Moving Averages')
    axes[0, 0].set_ylabel('Giá ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 2. Volume
    axes[0, 1].bar(df.index, df['volume'], alpha=0.6, color='green')
    axes[0, 1].set_title('Volume giao dịch')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].grid(True)
    
    # 3. Price changes
    axes[1, 0].hist(df['price_change'].dropna(), bins=50, alpha=0.7, color='blue')
    axes[1, 0].set_title('Phân phối thay đổi giá')
    axes[1, 0].set_xlabel('Thay đổi giá (%)')
    axes[1, 0].set_ylabel('Tần suất')
    axes[1, 0].grid(True)
    
    # 4. RSI
    axes[1, 1].plot(df.index, df['RSI'], color='purple')
    axes[1, 1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
    axes[1, 1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
    axes[1, 1].set_title('RSI Indicator')
    axes[1, 1].set_ylabel('RSI')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # 5. Volatility
    axes[2, 0].plot(df.index, df['volatility_20d'], color='orange')
    axes[2, 0].set_title('Volatility 20 ngày')
    axes[2, 0].set_ylabel('Volatility')
    axes[2, 0].grid(True)
    
    # 6. Volume ratio
    axes[2, 1].plot(df.index, df['volume_ratio'], color='brown')
    axes[2, 1].axhline(y=1, color='black', linestyle='--', alpha=0.7)
    axes[2, 1].set_title('Volume Ratio (Volume / MA5)')
    axes[2, 1].set_ylabel('Ratio')
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('stock_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("🚀 Tạo dataset giá cổ phiếu...")
    
    # Tạo dữ liệu cơ bản
    df = generate_stock_data(n_days=1000)
    
    # Thêm technical indicators
    df = add_technical_indicators(df)
    
    # Thêm time features
    df = add_time_features(df)
    
    # Tạo target variable
    df = create_target_variable(df, forecast_days=7)
    
    # Phân tích dữ liệu
    analyze_data(df)
    
    # Lưu dataset
    df.to_csv('stock_prices.csv')
    print(f"\n✅ Đã lưu dataset vào file: stock_prices.csv")
    print(f"📊 Dataset có {len(df)} ngày và {len(df.columns)} features")
    
    # Hiển thị 5 mẫu đầu tiên
    print("\n=== 5 MẪU ĐẦU TIÊN ===")
    print(df.head())
    
    # Tạo biểu đồ phân tích
    print("\n📈 Đang tạo biểu đồ phân tích...")
    visualize_data(df)
    
    print("\n🎉 Hoàn thành tạo dataset giá cổ phiếu!") 