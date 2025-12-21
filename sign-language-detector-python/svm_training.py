import pickle
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
print("Loading data from data.pickle...")
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

print(f"Data loaded successfully!")
print(f"Total samples: {len(data)}")
print(f"Number of features: {data.shape[1]}")
print(f"Number of classes: {len(np.unique(labels))}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels,
    random_state=42
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train SVM model
print("\nTraining SVM model...")
model = svm.SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
model.fit(X_train, y_train)

print("Training completed!")

# Test the model
print("\nTesting the model...")
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Save the model
print("\nSaving the model...")
with open('svm_model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model saved as 'svm_model.p'")
print("\nTraining complete!")