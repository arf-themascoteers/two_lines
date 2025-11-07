import pandas as pd
import numpy as np
import inspect_both

df = pd.read_csv("problem.csv")
points = df[["x", "y"]].values
true_labels = df["line"].values
n = len(points)

def fit_line(p1, p2):
    if np.abs(p1[0] - p2[0]) < 1e-6:
        m = 1e6
        c = p1[1] - m * p1[0]
    else:
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        c = p1[1] - m * p1[0]
    return m, c

def compute_inliers(points, m, c, threshold):
    x, y = points[:, 0], points[:, 1]
    dists = np.abs(y - (m * x + c)) / np.sqrt(m**2 + 1)
    return np.where(dists < threshold)[0]

best_inliers1 = []
best_m1, best_c1 = None, None
threshold = 0.01

for step in range(1000):
    i, j = np.random.choice(n, 2, replace=False)
    m, c = fit_line(points[i], points[j])
    inliers = compute_inliers(points, m, c, threshold)
    if len(inliers) > len(best_inliers1):
        best_inliers1 = inliers
        best_m1, best_c1 = m, c

    if step % 10 == 0:
        mask = np.ones(n, dtype=bool)
        mask[best_inliers1] = False
        rest_points = points[mask]

        best_inliers2 = []
        best_m2, best_c2 = None, None
        for _ in range(300):
            a, b = np.random.choice(len(rest_points), 2, replace=False)
            m2, c2 = fit_line(rest_points[a], rest_points[b])
            in2 = compute_inliers(rest_points, m2, c2, threshold)
            if len(in2) > len(best_inliers2):
                best_inliers2 = in2
                best_m2, best_c2 = m2, c2

        idx1 = best_inliers1
        idx2 = np.arange(n)[mask][best_inliers2]

        predicted = np.zeros(n, dtype=int)
        predicted[idx2] = 1

        solution = df.copy()
        solution["line"] = predicted
        solution.to_csv("solution.csv", index=False)

        accuracy = (predicted == true_labels).mean() * 100
        print(f"Step {step} — Accuracy: {accuracy:.2f}%")
        inspect_both.plot_please(step,"dump1")
