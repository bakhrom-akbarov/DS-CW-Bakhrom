import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def convert_bool_to_string(df):
    for column in df.columns:
        if df[column].dtype == bool or (df[column].apply(lambda x: isinstance(x, bool)).any()):
            df[column] = df[column].astype(str)
    return df

# Load training features and labels
train_features_path = './training_set_values.csv'
train_labels_path = './training_set_labels.csv'
train_features = pd.read_csv(train_features_path)
train_labels = pd.read_csv(train_labels_path)
train_data = pd.merge(train_features, train_labels, on='id')

# Convert booleans to strings
train_data = convert_bool_to_string(train_data)

# Loading test data
test_data_path = './test_set_values.csv'
test_data = pd.read_csv(test_data_path)
test_data = convert_bool_to_string(test_data)

# Define features and target
X = train_data.drop(['id', 'status_group'], axis=1)
y = train_data['status_group']

# Preprocessing setup
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(transformers=[
    ('num', KNNImputer(n_neighbors=5), num_cols),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), cat_cols)
])

# Model setup
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=0))
])

# Model training and evaluation
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_valid)
accuracy = accuracy_score(y_valid, preds)
f1 = f1_score(y_valid, preds, average='macro')

# Print evaluation results
print('Validation Accuracy:', accuracy)
print('Validation F1 Score:', f1)
print("Confusion Matrix:\n", confusion_matrix(y_valid, preds))
print("\nFull Classification Report:\n", classification_report(y_valid, preds))

# Test predictions for submission
test_preds = model.predict(test_data.drop('id', axis=1))
submission = pd.DataFrame({'id': test_data['id'], 'status_group': test_preds})
submission.to_csv('./submission.csv', index=False)
print("Submission file created successfully!")
