import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load training features and labels
train_features_path = './training_set_values.csv'
train_labels_path = './training_set_labels.csv'
train_features = pd.read_csv(train_features_path)
train_labels = pd.read_csv(train_labels_path)

# Merge the training data and labels
train_data = pd.merge(train_features, train_labels, on='id')

# Load test data
test_data_path = './test_set_values.csv'
test_data = pd.read_csv(test_data_path)

# Convert boolean columns to string type if mixed with strings
for column in train_data.columns:
    if train_data[column].dtype == 'object':
        if train_data[column].apply(lambda x: isinstance(x, bool)).any():
            train_data[column] = train_data[column].astype(str)
            test_data[column] = test_data[column].astype(str)

# Define features and target
X = train_data.drop(['id', 'status_group'], axis=1)  # Exclude 'id' and the target column
y = train_data['status_group']  # Target variable

# Preprocessing for numerical data
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
encoder = OneHotEncoder(handle_unknown='ignore')

numerical_transformer = Pipeline(steps=[
    ('imputer', num_imputer)
])

categorical_transformer = Pipeline(steps=[
    ('imputer', cat_imputer),
    ('onehot', encoder)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# Use KNNImputer for numerical data
num_imputer = KNNImputer(n_neighbors=5)
numerical_transformer = Pipeline(steps=[
    ('imputer', num_imputer)
])

# Define the model
model = DecisionTreeClassifier(random_state=0)

# Create and evaluate the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Fit the model
pipeline.fit(X_train, y_train)

# Predict on validation set
preds = pipeline.predict(X_valid)

# Evaluate the model
score = accuracy_score(y_valid, preds)
print('Validation Accuracy:', score)

# Prepare the test data (apply the same preprocessing)
test_preds = pipeline.predict(test_data.drop('id', axis=1))

# Create submission DataFrame
submission = pd.DataFrame({'id': test_data['id'], 'status_group': test_preds})

# Export to CSV
submission.to_csv('./submission.csv', index=False)
print("Submission file created successfully!")
