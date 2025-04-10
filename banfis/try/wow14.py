#!/usr/bin/env python
# Created by "Thieu" at 10:02, 15/02/2025 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import torch
import torch.nn as nn
import torch.optim as optim


class ANFIS(nn.Module):
    def __init__(self, num_rules=3, learning_rate=0.01, epochs=100):
        super(ANFIS, self).__init__()
        self.num_rules = num_rules
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Khởi tạo các tham số cho Membership Functions (MF)
        self.means = nn.Parameter(torch.rand(num_rules, 2))
        self.sigmas = nn.Parameter(torch.rand(num_rules, 2))

        # Khởi tạo tham số tuyến tính của mô hình Takagi-Sugeno
        self.p = nn.Parameter(torch.rand(num_rules, 2))  # Hỗ trợ multi-output
        self.q = nn.Parameter(torch.rand(num_rules, 2))
        self.r = nn.Parameter(torch.rand(num_rules, 2))

        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def gaussian(self, x, mean, sigma):
        return torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2))

    def forward(self, X):
        # Tính toán mức độ thuộc về của đầu vào với các tập mờ
        mu = torch.stack([self.gaussian(X[:, i].unsqueeze(1), self.means[:, i], self.sigmas[:, i]) for i in range(2)], dim=1)

        # Tính toán độ kích hoạt của luật
        firing_strengths = torch.prod(mu, dim=1)

        # Chuẩn hóa các giá trị kích hoạt
        weights = firing_strengths / torch.sum(firing_strengths, dim=1, keepdim=True)

        # Tính toán đầu ra của mô hình
        f = torch.stack([self.p[j] * X[:, 0].unsqueeze(1) + self.q[j] * X[:, 1].unsqueeze(1) + self.r[j] for j in range(self.num_rules)], dim=1)

        weights = weights.unsqueeze(-1)  # Để phù hợp với f có shape (batch_size, num_rules, 2)
        output = torch.sum(weights * f, dim=1)
        return output, weights.squeeze()

    def train_model(self, X, y):
        y = y.view(y.shape[0], -1)  # Đảm bảo y có đúng shape
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).clone().detach()
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32).clone().detach()

        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            output, _ = self.forward(X)
            loss = self.loss_fn(output, y)
            loss.backward()
            self.optimizer.step()

            # Hiển thị lỗi sau mỗi epoch
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, MSE: {loss.item()}")

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32).clone().detach()
        output, _ = self.forward(X)
        return output.detach().numpy()


# Dữ liệu mẫu (bài toán phi tuyến y = [x1^2 + x2^2, x1 + x2])
torch.manual_seed(0)
X_train = torch.rand(100, 2) * 2 - 1  # Giá trị x1, x2 từ -1 đến 1
y_train = torch.column_stack((X_train[:, 0] ** 2 + X_train[:, 1] ** 2, X_train[:, 0] + X_train[:, 1]))  # Multi-output

# Huấn luyện ANFIS
anfis = ANFIS(num_rules=3, learning_rate=0.01, epochs=100)
anfis.train_model(X_train, y_train)

# Dự đoán trên tập kiểm tra
X_test = torch.tensor([[0.5, 0.5], [-0.5, -0.5], [0.2, -0.3]])
y_pred = anfis.predict(X_test)
print("Predictions:\n", y_pred)
