# Machine Learning Study Journey

## 🛠️ Cài đặt

### **Cách 1: Sử dụng Conda (Khuyến nghị)**

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/machine-learning-study.git
cd machine-learning-study

# Tạo môi trường conda
conda create -n ml_study python=3.10
conda activate ml_study

# Cài đặt thư viện cơ bản
conda install numpy pandas matplotlib scikit-learn jupyter
```

### **Cách 2: Sử dụng pip (cho máy CPU-only)**

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/machine-learning-study.git
cd machine-learning-study

# Tạo môi trường ảo Python
python -m venv ml_study
source ml_study/bin/activate  # Linux/Mac
# hoặc
ml_study\Scripts\activate     # Windows

# Cài đặt thư viện
pip install numpy pandas matplotlib scikit-learn jupyter
```

### **Cách 3: Cài đặt cho macOS**

```bash
# Cài đặt Homebrew (nếu chưa có)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Cài đặt Python
brew install python@3.10

# Tạo môi trường ảo
python3.10 -m venv ml_study
source ml_study/bin/activate

# Cài đặt thư viện
pip install numpy pandas matplotlib scikit-learn jupyter
```

### **Cách 4: Cài đặt cho máy yếu (CPU-only, RAM thấp)**

```bash
# Clone repository
git clone https://github.com/DrakeVuh/machine-learning-study.git
cd machine-learning-study

# Tạo môi trường conda với Python phiên bản nhẹ
conda create -n ml_study_light python=3.9
conda activate ml_study_light

# Cài đặt thư viện phiên bản nhẹ
conda install numpy=1.21 pandas=1.3 matplotlib=3.4 scikit-learn=1.0
conda install jupyter

# Hoặc dùng pip với phiên bản cũ hơn
pip install numpy==1.21.6 pandas==1.3.5 matplotlib==3.4.3 scikit-learn==1.0.2 jupyter
```


## 🔧 Yêu cầu hệ thống

### **Tối thiểu:**
- RAM: 4GB
- CPU: Dual-core
- Dung lượng ổ cứng: 2GB trống

### **Khuyến nghị:**
- RAM: 8GB trở lên
- CPU: Quad-core trở lên
- Dung lượng ổ cứng: 5GB trống

### **Lưu ý cho máy yếu:**
- Sử dụng dataset nhỏ hơn
- Giảm số iteration trong training
- Tắt các tính năng visualization không cần thiết

## 🤝 Đóng góp

Mọi góp ý và feedback đều được chào đón!

## 📄 License

MIT License 