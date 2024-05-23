import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import io
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)

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

# 准备数据
def prepare_data(df):
    # 使用前一天的收盘价作为特征
    df['Previous_Close'] = df['Close'].shift(1)
    df.dropna(inplace=True)
    X = df[['Previous_Close']]
    y = df['Close']
    return X, y

# 划分训练集和测试集
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# 训练模型
def train_model(X_train, y_train, model_type):
    if model_type == 'linear_regression':
        model = LinearRegression()
    elif model_type == 'knn':
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVR(kernel='linear', C=1e3)
    else:
        raise ValueError("Invalid model_type")
    model.fit(X_train, y_train)
    return model

# 评估模型
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def plot_model_performance(model_name, y_test, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values, label='True Values', color='blue', alpha=0.5)
    plt.plot(y_pred, label='Predicted Values', color='red', alpha=0.5)
    plt.title(f'{model_name} Prediction Performance')
    plt.xlabel('Test Data Index')
    plt.ylabel('Stock Price')
    plt.legend(loc='upper left')
    plt.show()

class StockAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Stock Analysis")
        self.geometry("800x600")

        self.file_path = None

        self.create_widgets()

    def create_widgets(self):
        tk.Grid.rowconfigure(self, 1, weight=1)
        tk.Grid.columnconfigure(self, 0, weight=1)
        tk.Grid.columnconfigure(self, 1, weight=1)
        tk.Grid.columnconfigure(self, 2, weight=1)

        self.file_button = tk.Button(self, text="导入Excel文件", command=self.import_excel)
        self.file_button.grid(row=0, column=0, padx=10, pady=10, sticky='w')

        self.analysis_button = tk.Button(self, text="分析数据", command=self.analyze_data, state=tk.DISABLED)
        self.analysis_button.grid(row=0, column=1, padx=10, pady=10)

        self.train_button = tk.Button(self, text="训练模型", command=self.train_models, state=tk.DISABLED)
        self.train_button.grid(row=0, column=2, padx=10, pady=10, sticky='e')

        self.output = ScrolledText(self, wrap=tk.WORD)
        self.output.grid(row=1, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

    def import_excel(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
        if self.file_path:
            self.df = read_excel_file(self.file_path)
            self.analysis_button.config(state=tk.NORMAL)
            messagebox.showinfo("成功", "Excel文件已成功导入")

    def analyze_data(self):
        if self.df is not None:
            self.df = preprocess_data(self.df)
            self.output.delete(1.0, tk.END)
            self.capture_print(lambda: basic_statistics(self.df))
            self.df['5_day_MA'] = moving_average(self.df, 5)
            self.df['30_day_MA'] = moving_average(self.df, 30)
            self.capture_print(lambda: print(self.df.head(232)))
            self.train_button.config(state=tk.NORMAL)

    def train_models(self):
        model_types = ['linear_regression', 'knn', 'decision_tree', 'random_forest', 'svm']
        if self.df is not None:
            X, y = prepare_data(self.df)
            X_train, X_test, y_train, y_test = split_data(X, y)
            for model_type in model_types:
                model = train_model(X_train, y_train, model_type)
                y_pred = model.predict(X_test)
                mse, r2 = evaluate_model(model, X_test, y_test)
                self.capture_print(lambda: print(f"{model_type.capitalize()} Model:\n  MSE: {mse:.2f}\n  R2: {r2:.2f}\n"))
                plot_model_performance(model_type.capitalize(), y_test, y_pred)

    def capture_print(self, func):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        func()
        output_str = sys.stdout.getvalue()
        sys.stdout = old_stdout
        self.output.insert(tk.END, output_str)

    def on_closing(self):
        if messagebox.askokcancel("退出", "你确定要退出吗？"):
            self.destroy()

if __name__ == "__main__":
    app = StockAnalysisGUI()
    app.mainloop()