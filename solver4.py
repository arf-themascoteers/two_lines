import pandas as pd
import numpy as np
import cv2
import inspect_both

df = pd.read_csv("problem.csv")
true_labels = df["line"].values
n = len(df)

canvas = np.zeros((512, 512), dtype=np.uint8)
scale = 256

for _, row in df.iterrows():
    cx = int(scale + row.x * scale)
    cy = int(scale - row.y * scale)
    if 0 <= cx < 512 and 0 <= cy < 512:
        cv2.circle(canvas, (cx, cy), radius=2, color=255, thickness=-1)

lines = cv2.HoughLines(canvas, rho=1, theta=np.pi / 180, threshold=20)

if lines is None or len(lines) < 2:
    print("Not enough lines found")
    exit()

l1 = lines[0][0]
l2 = lines[1][0]

def distance_to_line(x, y, rho, theta):
    return abs(x * np.cos(theta) + y * np.sin(theta) - rho)

points = df[["x", "y"]].values
dist1 = np.array([distance_to_line(x, y, l1[0], l1[1]) for x, y in points])
dist2 = np.array([distance_to_line(x, y, l2[0], l2[1]) for x, y in points])

predicted = (dist2 < dist1).astype(int)

solution = df.copy()
solution["line"] = predicted
solution.to_csv("solution.csv", index=False)

accuracy = (predicted == true_labels).mean() * 100
print(f"Accuracy: {accuracy:.2f}%")
inspect_both.plot_please("hough",".")
