#!/usr/bin/env python
# Created by "Thieu" at 10:00, 15/02/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np


class ANFIS:
    def __init__(self, num_rules=3, learning_rate=0.01, epochs=100):
        self.num_rules = num_rules  # Số lượng luật mờ
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Khởi tạo các tham số cho Membership Functions (MF)
        self.means = np.random.rand(num_rules, 2)
        self.sigmas = np.random.rand(num_rules, 2)

        # Khởi tạo tham số tuyến tính của mô hình Takagi-Sugeno
        self.p = np.random.rand(num_rules, 2)  # Hỗ trợ multi-output
        self.q = np.random.rand(num_rules, 2)
        self.r = np.random.rand(num_rules, 2)

    def gaussian(self, x, mean, sigma):
        return np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    def forward(self, X):
        # Tính toán mức độ thuộc về của đầu vào với các tập mờ
        mu = np.array([[self.gaussian(X[:, i], self.means[j, i], self.sigmas[j, i])
                        for j in range(self.num_rules)] for i in range(2)])

        # Tính toán độ kích hoạt của luật
        firing_strengths = np.prod(mu, axis=0)

        # Chuẩn hóa các giá trị kích hoạt
        weights = firing_strengths / np.sum(firing_strengths, axis=0, keepdims=True)

        # Tính toán đầu ra của mô hình
        f = np.array([self.p[j] * X[:, 0, np.newaxis] + self.q[j] * X[:, 1, np.newaxis] + self.r[j]
                      for j in range(self.num_rules)])

        # Sửa lỗi broadcasting
        weights = weights[:, :, np.newaxis]  # Chuyển weights thành (3, 100, 1) để phù hợp với f (3, 100, 2)
        output = np.sum(weights * f, axis=0)
        return output, weights.squeeze()

    def train(self, X, y):
        y = np.atleast_2d(y)  # Đảm bảo y là dạng ma trận
        for epoch in range(self.epochs):
            output, weights = self.forward(X)
            error = y - output

            # Cập nhật tham số tuyến tính (p, q, r) bằng Least Squares Estimation (LSE)
            for j in range(self.num_rules):
                self.p[j] += self.learning_rate * np.sum(weights[j][:, np.newaxis] * error * X[:, 0, np.newaxis],
                                                         axis=0)
                self.q[j] += self.learning_rate * np.sum(weights[j][:, np.newaxis] * error * X[:, 1, np.newaxis],
                                                         axis=0)
                self.r[j] += self.learning_rate * np.sum(weights[j][:, np.newaxis] * error, axis=0)

            # Cập nhật tham số của hàm thành viên bằng Gradient Descent
            for j in range(self.num_rules):
                for i in range(2):
                    weight_j = weights[j][:, np.newaxis]  # Điều chỉnh kích thước weights[j] để phù hợp với error
                    grad_mean = np.sum(
                        weight_j * error * (X[:, i, np.newaxis] - self.means[j, i]) / (self.sigmas[j, i] ** 2), axis=0)
                    grad_sigma = np.sum(
                        weight_j * error * ((X[:, i, np.newaxis] - self.means[j, i]) ** 2) / (self.sigmas[j, i] ** 3),
                        axis=0)
                    self.means[j, i] -= self.learning_rate * grad_mean.mean()
                    self.sigmas[j, i] -= self.learning_rate * grad_sigma.mean()

            # Hiển thị lỗi sau mỗi epoch
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, MSE: {np.mean(error ** 2)}")

    def predict(self, X):
        output, _ = self.forward(X)
        return output


# Dữ liệu mẫu (bài toán phi tuyến y = [x1^2 + x2^2, x1 + x2])
np.random.seed(0)
X_train = np.random.rand(100, 2) * 2 - 1  # Giá trị x1, x2 từ -1 đến 1
y_train = np.column_stack((X_train[:, 0] ** 2 + X_train[:, 1] ** 2, X_train[:, 0] + X_train[:, 1]))  # Multi-output

# Huấn luyện ANFIS
anfis = ANFIS(num_rules=3, learning_rate=0.01, epochs=100)
anfis.train(X_train, y_train)

# Dự đoán trên tập kiểm tra
X_test = np.array([[0.5, 0.5], [-0.5, -0.5], [0.2, -0.3]])
y_pred = anfis.predict(X_test)
print("Predictions:\n", y_pred)



