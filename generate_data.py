import numpy as np
import pandas as pd

np.random.seed(42)

data = []
scores = []

for _ in range(rows):
    cgpa = round(np.random.uniform(5, 10), 2)
    dsa = np.random.randint(1, 6)
    projects = np.random.randint(0, 6)
    internship = np.random.randint(0, 2)
    communication = np.random.randint(1, 6)
    aptitude = np.random.randint(40, 100)
    certifications = np.random.randint(0, 6)
    consistency = round(np.random.uniform(4, 10), 1)
    score = np.random.randint(40, 100)

    weighted_score = (
        cgpa * 0.25 +
        dsa * 1.5 +
        projects * 0.8 +
        internship * 1.0 +
        communication * 1.0 +
        aptitude * 0.05 +
        certifications * 0.5 +
        consistency * 0.7 +
        score * 0.05
    )

    weighted_score += np.random.normal(0, 1)

    scores.append(weighted_score)

    data.append([
        cgpa, dsa, projects, internship,
        communication, aptitude, certifications,
        consistency, score, weighted_score
    ])

# Convert to DataFrame
df = pd.DataFrame(data, columns=[
    "cgpa", "dsa", "projects", "internship",
    "communication", "aptitude", "certifications",
    "consistency", "score", "weighted_score"
])

# 🔥 Dynamic threshold (IMPORTANT)
threshold = df["weighted_score"].median()

# Assign status
df["status"] = (df["weighted_score"] > threshold).astype(int)

# Drop weighted_score (not needed for model)
df = df.drop("weighted_score", axis=1)

df.head()
# ✅ SAVE INSIDE data folder
df.to_csv("data/btech_cse_advanced.csv", index=False)

print("✅ Dataset created inside data folder!")
