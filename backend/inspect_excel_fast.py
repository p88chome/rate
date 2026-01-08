import pandas as pd
try:
    df = pd.read_excel(r"d:\rate\Demo資料_利率合理性分析.xlsx", nrows=0)
    print("Columns:", df.columns.tolist())
except Exception as e:
    print(e)
