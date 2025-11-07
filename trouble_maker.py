import numpy as np
import pandas as pd

def generate_line_points(slope, intercept, num_points, noise_std=0.05):
    x = np.random.uniform(-1, 1, size=num_points)
    y = slope * x + intercept
    x += np.random.normal(0, noise_std, size=num_points)
    y += np.random.normal(0, noise_std, size=num_points)
    return x, y

def main():
    np.random.seed(42)

    m1 = np.random.uniform(-2, 2)
    c1 = np.random.uniform(-0.5, 0.5)
    x1, y1 = generate_line_points(m1, c1, 100)
    df1 = pd.DataFrame({
        "pid": np.arange(1, 101),
        "x": x1,
        "y": y1,
        "line": 0
    })

    m2 = np.random.uniform(-2, 2)
    c2 = np.random.uniform(-0.5, 0.5)
    x2, y2 = generate_line_points(m2, c2, 100)
    df2 = pd.DataFrame({
        "pid": np.arange(101, 201),
        "x": x2,
        "y": y2,
        "line": 1
    })

    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv("problem.csv", index=False)

if __name__ == "__main__":
    main()
