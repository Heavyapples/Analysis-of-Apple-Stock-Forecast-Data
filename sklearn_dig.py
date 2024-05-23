# sklearn数据挖掘+matplotlib可视化
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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

def main():
    file_path = r'C:\Users\13729\Desktop\AAPL.xlsx'
    df = pd.read_excel(file_path)
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model_types = ['linear_regression', 'knn', 'decision_tree', 'random_forest', 'svm']
    predictions = []

    for model_type in model_types:
        print(f"Training and evaluating {model_type} model...")
        model = train_model(X_train, y_train, model_type)
        y_pred = model.predict(X_test)
        predictions.append(y_pred)

        mse, r2 = evaluate_model(model, X_test, y_test)
        print(f"{model_type.capitalize()} Model:")
        print(f"  Mean Squared Error (MSE): {mse:.2f}")
        print(f"  R-squared (R2): {r2:.2f}\n")

    for model_name, y_pred in zip(model_types, predictions):
        plot_model_performance(model_name.capitalize(), y_test, y_pred)

if __name__ == '__main__':
    main()