import pandas as pd
import matplotlib.pyplot as plt

#df = pd.read_csv("problem.csv")
df = pd.read_csv("solution.csv")
x_min, x_max = df["x"].min(), df["x"].max()
y_min, y_max = df["y"].min(), df["y"].max()

print(f"x range: {x_min:.4f} to {x_max:.4f}")
print(f"y range: {y_min:.4f} to {y_max:.4f}")

plt.figure(figsize=(6, 6))
plt.scatter(df[df.line == 0].x, df[df.line == 0].y, color='red', label='Line 0', s=10)
plt.scatter(df[df.line == 1].x, df[df.line == 1].y, color='green', label='Line 1', s=10)

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(True)
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Generated Points by Line Label")
plt.show()


