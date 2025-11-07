import pandas as pd
import torch
import torch.optim as optim
import inspect_both

df = pd.read_csv("problem.csv")
points = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
true_labels = torch.tensor(df["line"].values, dtype=torch.int64)

n = points.shape[0]
s = torch.randn(n, requires_grad=True)
m1 = torch.randn(1, requires_grad=True)
c1 = torch.randn(1, requires_grad=True)
m2 = torch.randn(1, requires_grad=True)
c2 = torch.randn(1, requires_grad=True)

optimizer = optim.Adam([s, m1, c1, m2, c2], lr=0.01)

for _ in range(5000):
    optimizer.zero_grad()
    bin = torch.sigmoid(s)
    x = points[:, 0]
    y = points[:, 1]
    d1 = (y - (m1 * x + c1)) ** 2
    d2 = (y - (m2 * x + c2)) ** 2
    loss = torch.sum(bin * d1 + (1 - bin) * d2)
    loss += torch.abs(torch.sum(bin) - n / 2)
    loss.backward()
    optimizer.step()

    if _ % 10 == 0:
        print(f"Loss: {loss.item():.4f}")

        predicted = (torch.sigmoid(s) >= 0.5).int().numpy()

        solution = df.copy()
        solution["line"] = predicted
        solution.to_csv("solution.csv", index=False)

        accuracy = (predicted == true_labels.numpy()).mean() * 100
        print(f"Accuracy: {accuracy:.2f}%")
        inspect_both.plot_please(_)
