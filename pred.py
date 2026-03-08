import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Step 1: Create dataset
data = {
    "study_hours": [1,2,3,4,5,6,7,8,9,10],
    "attendance": [50,55,60,65,70,75,80,85,90,95],
    "previous_score": [40,42,45,50,55,60,65,70,75,80],
    "assignment_score": [30,35,40,45,50,55,60,65,70,75],
    "pass_fail": [0,0,0,0,1,1,1,1,1,1]
}

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

print("Dataset:\n", df)

# Step 3: Split input and target
X = df.drop("pass_fail", axis=1)
y = df["pass_fail"]

# Step 4: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Create pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Step 6: Hyperparameter grid
param_grid = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__solver": ["lbfgs"]
}

# Step 7: GridSearchCV
grid = GridSearchCV(pipeline, param_grid, cv=5)

grid.fit(X_train, y_train)

# Step 8: Best model
print("\nBest Model:")
print(grid.best_estimator_)

# Step 9: Prediction
predictions = grid.predict(X_test)

print("\nPredictions:", predictions)
print("Actual:", list(y_test))