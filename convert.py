import pandas as pd

# Load the CSV with correct delimiter
df = pd.read_csv("Dataset BMKG.csv", delimiter=';')

# Convert to JSON
json_data = df.to_json(orient="records", lines=False)

# Save to file
with open("Dataset_BMKG.json", "w") as f:
    f.write(json_data)