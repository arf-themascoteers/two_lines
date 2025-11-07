import pandas as pd
import numpy as np
import torch
import inspect_both

df = pd.read_csv("problem.csv")
points = torch.tensor(df[["x", "y"]].values, dtype=torch.float32)
true_labels = df["line"].values
n = len(points)

m1 = torch.randn(1)
c1 = torch.randn(1)
m2 = torch.randn(1)
c2 = torch.randn(1)
z = torch.randn(n)

for step in range(100):
    s = torch.sigmoid(z)
    x = points[:, 0]
    y = points[:, 1]

    d1 = (y - (m1 * x + c1))**2
    d2 = (y - (m2 * x + c2))**2

    w1 = s / (s + 1e-6)
    w2 = (1 - s) / ((1 - s) + 1e-6)

    w1 = w1 / (w1.sum() + 1e-8)
    w2 = w2 / (w2.sum() + 1e-8)

    X = torch.stack([x, torch.ones_like(x)], dim=1)

    sol1 = torch.linalg.lstsq((w1.unsqueeze(1) * X), (w1 * y).unsqueeze(1)).solution
    m1, c1 = sol1[:, 0]

    sol2 = torch.linalg.lstsq((w2.unsqueeze(1) * X), (w2 * y).unsqueeze(1)).solution
    m2, c2 = sol2[:, 0]

    d1 = (y - (m1 * x + c1))**2
    d2 = (y - (m2 * x + c2))**2

    z = - (d1 - d2)

    if step % 5 == 0:
        pred = (torch.sigmoid(z) >= 0.5).int().numpy()
        solution = df.copy()
        solution["line"] = pred
        solution.to_csv("solution.csv", index=False)

        accuracy = (pred == true_labels).mean() * 100
        print(f"Step {step} — Accuracy: {accuracy:.2f}%")
        inspect_both.plot_please(step,"dump2")
