import pandas as pd
try:
    df = pd.read_excel('demo_data.xlsx', sheet_name='Demo資料')
    print("Columns:", df.columns.tolist())
    print("First 3 rows:", df.head(3).to_dict())
    print("Unique values in '產品' col if exists:", df['產品名稱'].unique() if '產品名稱' in df.columns else "No 產品 column")
    print("Unique values in '備註' col if exists:", df['備註'].unique() if '備註' in df.columns else "No 備註 column")
    print("Unique values in '身分' col if exists:", df['身分'].unique() if '身分' in df.columns else "No 身分 column")
except Exception as e:
    print(e)
