# Pandas数据分析
import pandas as pd

# 读取Excel文件
def read_excel_file(file_path):
    return pd.read_excel(file_path)

# 数据预处理
def preprocess_data(df):
    # 处理缺失值
    df.fillna(method='ffill', inplace=True)
    # 删除重复值
    df.drop_duplicates(inplace=True)
    # 重置索引
    df.reset_index(drop=True, inplace=True)
    return df

# 计算基本统计数据
def basic_statistics(df):
    # 仅选择与股价相关的列
    price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    return df[price_columns].describe()

# 计算移动平均线
def moving_average(df, window):
    return df['Close'].rolling(window=window).mean()

def main():
    file_path = r'C:\Users\13729\Desktop\AAPL.xlsx'
    df = read_excel_file(file_path)
    # 数据预处理
    df = preprocess_data(df)
    # 显示基本统计数据
    print("基本统计数据：\n", basic_statistics(df))
    # 计算5日移动平均线，并添加到数据框中
    df['5_day_MA'] = moving_average(df, 5)
    # 计算30日移动平均线，并添加到数据框中
    df['30_day_MA'] = moving_average(df, 30)
    # 显示包含移动平均线的数据
    print("\n前10行数据（包含移动平均线）：\n", df.head(232))

if __name__ == '__main__':
    main()
