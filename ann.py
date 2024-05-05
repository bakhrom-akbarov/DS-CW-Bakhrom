import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.pipeline import Pipeline

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
X = train_data.drop(['id', 'status_group'], axis=1)
y = train_data['status_group']

# Encoding the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Preprocessing
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
encoder = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_imputer, num_cols),
        ('cat', encoder, cat_cols)
    ])

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)
test_data_processed = preprocessor.transform(test_data.drop('id', axis=1))

# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_processed.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y_categorical, train_size=0.8, test_size=0.2, random_state=0)

# Fit the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
_, score = model.evaluate(X_valid, y_valid)
print('Validation Accuracy:', score)

# Predict on test set
test_preds = model.predict(test_data_processed)
test_preds_labels = label_encoder.inverse_transform(test_preds.argmax(axis=1))

# Create submission DataFrame
submission = pd.DataFrame({'id': test_data['id'], 'status_group': test_preds_labels})

# Export to CSV
submission.to_csv('./submission.csv', index=False)
print("Submission file created successfully!")
