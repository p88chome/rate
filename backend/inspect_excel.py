import pandas as pd

file_path = r"d:\rate\Demo資料_利率合理性分析.xlsx"

try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())
    print("\nData Types:")
    print(df.dtypes)
except Exception as e:
    print(f"Error reading file: {e}")
