#Google COlab link: https://colab.research.google.com/drive/1MCqOD8mr673TaPBKwVsbZsjKK-5zPIqO
###**INTRODUCTION**

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

data=pd.read_csv('/content/IMDB Movies copy.csv')

data.head()

###**MODULE 1**

data.describe()

data.info()

data = data.dropna(subset=['Revenue'])

data['Meta_Score'].fillna(data['Meta_Score'].mean(), inplace=True)

features = ['Year_of_Release', 'Certificate', 'Runtime', 'Genre', 'Rating', 'Meta_Score',
            'Director', 'Star_1', 'Star_2', 'Star_3', 'Star_4', 'Total_Votes']
X = data[features]
y = data['Revenue']

categorical_cols = ['Certificate', 'Genre', 'Director', 'Star_1', 'Star_2', 'Star_3', 'Star_4']
numerical_cols = ['Year_of_Release', 'Runtime', 'Rating', 'Meta_Score', 'Total_Votes']

###**MODULE 2**

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

###**MODULE 3**

model_scores = {}

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42),
    "MLP Regressor": MLPRegressor(random_state=42, max_iter=1000)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    model_scores[name] = score
    print(f"{name} R² Score: {score:.4f}")

###**MODULE 4**

ann = Sequential()
ann.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
ann.add(Dense(units=64, activation='relu'))
ann.add(Dense(units=1))

ann.compile(optimizer='adam', loss='mean_squared_error')
history = ann.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

y_pred_ann = ann.predict(X_test)
ann_score = r2_score(y_test, y_pred_ann)
model_scores["Artificial Neural Network"] = ann_score
print(f"Artificial Neural Network R² Score: {ann_score:.4f}")

best_model = max(model_scores, key=model_scores.get)
print(f"\nBest Performing Model: {best_model} with R² Score: {model_scores[best_model]:.4f}")

###**MODULE 5**

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

# Optional: Sort models by R²
sorted_models = dict(sorted(model_scores.items(), key=lambda item: item[1], reverse=True))

# Plot model performance
def plot_model_scores(scores):
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(scores.keys()), y=list(scores.values()))
    plt.xticks(rotation=45)
    plt.ylabel("R² Score")
    plt.title("Model Comparison")
    plt.tight_layout()

plot_model_scores(sorted_models)

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted Revenue"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel("Actual Revenue")
    plt.ylabel("Predicted Revenue")
    plt.title(title)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.tight_layout()
    plt.show()

plot_actual_vs_predicted(y_test, y_pred_ann.ravel(), "Actual vs Predicted Revenue")

def plot_ann_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("ANN Training Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_ann_loss(history)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse

def show_evaluation_table(y_true, y_pred, model_name):
    r2, mae, rmse = evaluate_model(y_true, y_pred)
    print(f"{model_name} Evaluation")
    print({
        "R² Score": round(r2, 4),
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2)
    })

show_evaluation_table(y_test, y_pred_ann.ravel(), "Artificial Neural Network")