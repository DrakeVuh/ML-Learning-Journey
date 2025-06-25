# 📈 Time Series Forecasting - Dự đoán giá cổ phiếu

## 🎯 Mục tiêu
Dự đoán giá cổ phiếu/bitcoin trong 7 ngày tới dựa trên dữ liệu lịch sử và các chỉ báo kỹ thuật.

## 🚀 Tính năng nâng cao so với Linear Regression

### 1. **Sequential Data Processing**
- Xử lý dữ liệu theo thời gian
- Time windows và lag features
- Seasonality và trend analysis

### 2. **Feature Engineering phức tạp**
- Technical indicators (MA, RSI, MACD, Bollinger Bands)
- Price momentum và volatility
- Volume analysis
- Time-based features (day of week, month, etc.)

### 3. **Multiple Models**
- Linear Regression với time features
- LSTM (Long Short-Term Memory)
- Random Forest với engineered features
- Ensemble methods

### 4. **Advanced Evaluation**
- Time series cross-validation
- Walk-forward analysis
- Multiple metrics (RMSE, MAE, MAPE)
- Backtesting

## 📁 Cấu trúc dự án
```
time_series_forecasting/
├── data/
│   ├── generate_stock_data.py      # Tạo dữ liệu giả lập
│   └── stock_prices.csv           # Dataset
├── models/
│   ├── linear_regression_ts.py    # Linear Regression với time features
│   ├── lstm_model.py              # LSTM model
│   └── ensemble_model.py          # Ensemble methods
├── utils/
│   ├── feature_engineering.py     # Tạo technical indicators
│   ├── evaluation.py              # Metrics và visualization
│   └── preprocessing.py           # Data preprocessing
├── notebooks/
│   └── analysis.ipynb             # Jupyter notebook phân tích
└── main.py                        # Script chính
```

## 🛠️ Công nghệ sử dụng
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **TensorFlow/Keras**: Deep learning (LSTM)
- **Matplotlib/Seaborn**: Visualization
- **TA-Lib**: Technical analysis indicators

## 📊 Dataset Features
- **Price data**: Open, High, Low, Close
- **Volume**: Trading volume
- **Technical indicators**: MA, RSI, MACD, Bollinger Bands
- **Time features**: Day of week, month, quarter
- **Target**: Price after 7 days

## 🎓 Learning Objectives
1. Hiểu về Time Series Analysis
2. Feature Engineering cho sequential data
3. Cross-validation cho time series
4. Deep Learning với LSTM
5. Model evaluation và backtesting 