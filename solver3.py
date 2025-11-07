import pandas as pd
import numpy as np
import torch
import inspect_both

df = pd.read_csv("problem.csv")
points = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
true_labels = df["line"].values
n = len(points)

m1 = torch.randn(1, requires_grad=True)
c1 = torch.randn(1, requires_grad=True)
m2 = torch.randn(1, requires_grad=True)
c2 = torch.randn(1, requires_grad=True)

optimizer = torch.optim.Adam([m1, c1, m2, c2], lr=0.01)

for _ in range(1000):
    optimizer.zero_grad()
    x = points[:, 0]
    y = points[:, 1]

    d1 = (y - (m1 * x + c1)) ** 2
    d2 = (y - (m2 * x + c2)) ** 2

    assign = (d1 > d2).float()
    loss = torch.sum((1 - assign) * d1 + assign * d2)
    loss.backward()
    optimizer.step()

    if _ % 10 == 0:
        pred = assign.int().numpy()
        solution = df.copy()
        solution["line"] = pred
        solution.to_csv("solution.csv", index=False)

        accuracy = (pred == true_labels).mean() * 100
        print(f"Step {_} — Accuracy: {accuracy:.2f}%")
        inspect_both.plot_please(_,"dump3")
