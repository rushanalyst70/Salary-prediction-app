import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# CSV file पढ़ें
df = pd.read_csv("SLR_marks.csv")

print(df.describe())
print(df.tail())
print("Shape:", df.shape)
print(df.head())

# Numeric columns में IQR से outliers पहचानें और cap करें
num_cols = df.select_dtypes(include="number").columns

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {outlier_count} outliers before handling")

    df[col] = df[col].clip(lower=lower, upper=upper)

    remaining = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"{col}: {remaining} outliers after handling")

# Missing values भरें
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        mode = df[col].mode()
        if not mode.empty:
            df[col] = df[col].fillna(mode[0])

# Features और target चुनें
X = df[["Hours_Studied"]]
y = df["Exam_Marks"]

# Training और testing data में बाँटें
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model train करें
model = LinearRegression()
model.fit(X_train, y_train)

# Model को pickle file में save करें
joblib.dump(model, "salary_training_model.pkl")
print("Model saved as salary_training_model.pkl")

# Saved pickle file को load करने का उदाहरण:
# loaded_model = joblib.load("salary_training_model.pkl")
# predictions = loaded_model.predict(X_test)
# print(predictions)