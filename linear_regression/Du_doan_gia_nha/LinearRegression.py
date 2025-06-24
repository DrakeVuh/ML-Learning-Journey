import numpy as np

class MultipleLinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Khởi tạo mô hình Linear Regression
        
        Parameters:
        -----------
        learning_rate : float
            Tốc độ học của mô hình
        n_iterations : int
            Số lần lặp trong quá trình huấn luyện
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.history = {'loss': []}  # Lưu lịch sử loss để vẽ đồ thị
        
    def fit(self, X, y):
        """
        Huấn luyện mô hình sử dụng gradient descent
        
        Parameters:
        -----------
        X : numpy.ndarray
            Ma trận đặc trưng đầu vào
        y : numpy.ndarray
            Vector giá trị đầu ra
        """
        n_samples, n_features = X.shape
        
        # Khởi tạo weights và bias
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Tính prediction
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Tính loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            self.history['loss'].append(loss)
            
            # Tính gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # In thông tin mỗi 100 lần lặp
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.4f}")
        
    def predict(self, X):
        """
        Dự đoán giá trị đầu ra cho dữ liệu mới
        
        Parameters:
        -----------
        X : numpy.ndarray
            Ma trận đặc trưng đầu vào
            
        Returns:
        --------
        numpy.ndarray
            Vector giá trị dự đoán
        """
        return np.dot(X, self.weights) + self.bias
        
    def score(self, X, y):
        """
        Tính R2 score của mô hình
        
        Parameters:
        -----------
        X : numpy.ndarray
            Ma trận đặc trưng đầu vào
        y : numpy.ndarray
            Vector giá trị thực tế
            
        Returns:
        --------
        float
            R2 score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2
