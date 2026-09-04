import pandas as pd

df = pd.read_csv("ESMC-6000/si_proteins 1.csv")

expected = set(range(1, 315))
found = set(df["number"].astype(int))

print("Number of rows:", len(df))
print("Number of unique IDs:", len(found))
print("Missing IDs:", sorted(expected - found))
print("Extra IDs:", sorted(found - expected))
print("Duplicate IDs:", df[df["number"].duplicated()]["number"].tolist())

print("\nSequence length range:")
print(df["protein"].str.len().min(), "-", df["protein"].str.len().max())
