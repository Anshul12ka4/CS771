#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import tensorflow as tf  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences

# First dataset: Emoticon Dataset
# Load datasets
t_em_df = pd.read_csv("datasets/train/train_emoticon.csv")
v_em_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
test_em_df = pd.read_csv("datasets/test/test_emoticon.csv")

# Prepare training and validation data
t_em_X = t_em_df['input_emoticon'].tolist()  
t_em_Y = t_em_df['label'].tolist()  
v_em_X = v_em_df['input_emoticon'].tolist()  
v_em_Y = v_em_df['label'].tolist()  

# Prepare test input
test_em_X = test_em_df['input_emoticon'].tolist()

# Split training and validation inputs into DataFrames for encoding
t_em_X_split = pd.DataFrame([list(i) for i in t_em_X])
v_em_X_split = pd.DataFrame([list(j) for j in v_em_X])
test_em_X_split = pd.DataFrame([list(i) for i in test_em_X])  # Prepare test input

# OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit encoder on training data
X_em_t = encoder.fit_transform(t_em_X_split)
X_em_v = encoder.transform(v_em_X_split)

# Take 80% of emoticon training data
X_em_t, _, t_em_Y, _ = train_test_split(X_em_t, t_em_Y, train_size=0.8, random_state=42)

# Model Training Function
def model_training(X_train, y_train, X_valid, y_valid, model, per):
    val_acc = []
    for p in per:
        subset_size = int(p * len(X_train))
        X_subset = X_train[:subset_size]
        y_subset = y_train[:subset_size]
        
        model.fit(X_subset, y_subset)
        y_pred = model.predict(X_valid)
        acc = accuracy_score(y_valid, y_pred)
        val_acc.append(acc)
        print(f"{p*100}% of data: Accuracy = {acc:.4f}")
    return val_acc

# Train the Logistic Regression model
model_lr = LogisticRegression(max_iter=1000)
print("Logistic Regression")
per = [1.0]
lr_accuracies = model_training(X_em_t, t_em_Y, X_em_v, v_em_Y, model_lr, per)

# Prepare test input for prediction
X_em_test = encoder.transform(test_em_X_split)  # Transform test input using the same encoder

# Make predictions on the test dataset
pred_emoticon = model_lr.predict(X_em_test)

# Save predictions to a text file
np.savetxt('pred_emoticon.txt', pred_emoticon, fmt='%d')

print("Predictions saved to pred_emoticon.txt.")


# Second dataset: Deep Feature Dataset
deep_f = np.load("datasets/train/train_feature.npz")
X_d = deep_f['features']
y_d = deep_f['label']
X_d_flat = X_d.reshape(X_d.shape[0], -1)

# Load validation data
val_f = np.load("datasets/valid/valid_feature.npz")
X_val = val_f['features']
y_val = val_f['label']
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# Load test data
test_f = np.load("datasets/test/test_feature.npz")
X_test = test_f['features']
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Shuffle training data
X_d_flat, y_d = shuffle(X_d_flat, y_d, random_state=42)

# Take 60% of deep feature training data
X_d_flat, _, y_d, _ = train_test_split(X_d_flat, y_d, train_size=0.6, random_state=42)

# Instantiate Random Forest model
model_rf = RandomForestClassifier(n_estimators=200)

# Train Random Forest using 60% of the training data
model_rf.fit(X_d_flat, y_d)

# Validate the model
y_val_pred = model_rf.predict(X_val_flat)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy = {val_accuracy:.4f}")

# Make predictions on the test set
predictions_rf = model_rf.predict(X_test_flat)

# Save predictions to a .txt file
np.savetxt('pred_deepfeat.txt', predictions_rf, fmt='%d')

print("Predictions saved to pred_deepfeat.txt.")


# Third dataset: Text Sequence Dataset
# Load and preprocess the data
train_data = pd.read_csv("datasets/train/train_text_seq.csv")
val_data = pd.read_csv("datasets/valid/valid_text_seq.csv")
test_data = pd.read_csv("datasets/test/test_text_seq.csv")

def preprocess_data(data):
    X = np.array([list(map(int, list(s))) for s in data['input_str']])
    return X

# Prepare data
X_train = preprocess_data(train_data)
y_train = train_data['label'].values
X_val = preprocess_data(val_data)
y_val = val_data['label'].values
X_test = preprocess_data(test_data)

# Padding sequences to ensure uniform length
max_len = max([len(seq) for seq in X_train])
X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_val = pad_sequences(X_val, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)

# Define the BiLSTM model
model = Sequential([
    Embedding(input_dim=10, output_dim=64),
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.4),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Train the model (with epochs displayed)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, callbacks=[early_stopping, reduce_lr], verbose=1)

# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Predict on the test set
y_test_pred = model.predict(X_test)
y_test_pred_labels = np.argmax(y_test_pred, axis=1)

# Save predictions to a .txt file
np.savetxt('pred_textseq.txt', y_test_pred_labels, fmt='%d')

print("Predictions saved to pred_textseq.txt.")




# In[6]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the feature data
t_f = np.load("datasets/train/train_feature.npz", allow_pickle=True)
t_f_X = t_f['features']

# Load emoticon and text sequence datasets
t_em_df = pd.read_csv("datasets/train/train_emoticon.csv")
t_em_X = t_em_df['input_emoticon'].tolist()
t_em_Y = t_em_df['label'].tolist()

t_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
t_seq_X = t_seq_df['input_str'].tolist()

# Ensure all datasets have the same number of samples (5664)
t_f_X = t_f_X[:5664]  # Feature dataset reduced to 5664 samples
t_em_X = t_em_X[:5664]
t_em_Y = t_em_Y[:5664]
t_seq_X = t_seq_X[:5664]

# Combine training data
t_em_vect = CountVectorizer(analyzer='char', max_features=1000)
t_em_X_vect = t_em_vect.fit_transform(t_em_X).toarray()

t_seq_vect = TfidfVectorizer(max_features=1000)
t_seq_X_vect = t_seq_vect.fit_transform(t_seq_X).toarray()

# Flatten the feature dataset
t_f_X_flat = t_f_X.reshape(t_f_X.shape[0], -1).astype(np.float32)

# Combine all datasets
X_com = np.hstack((t_em_X_vect, t_seq_X_vect, t_f_X_flat))

# Scale the features
scal = StandardScaler()
X_com_scal = scal.fit_transform(X_com)

# Prepare the target variable
y_com = np.array(t_em_Y)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_com_scal, y_com)

# Load validation datasets (ensure consistent sizes with training data)
val_em_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
val_em_X = val_em_df['input_emoticon'].tolist()
val_em_Y = val_em_df['label'].tolist()

val_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
val_seq_X = val_seq_df['input_str'].tolist()

val_f = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
val_f_X = val_f['features']

# Ensure validation dataset sizes match (5664 samples)
val_em_X = val_em_X[:5664]
val_em_Y = val_em_Y[:5664]
val_seq_X = val_seq_X[:5664]
val_f_X = val_f_X[:5664]

# Combine validation data
val_em_vect = t_em_vect.transform(val_em_X).toarray()
val_seq_vect = t_seq_vect.transform(val_seq_X).toarray()
val_f_X_flat = val_f_X.reshape(val_f_X.shape[0], -1).astype(np.float32)

X_val_com = np.hstack((val_em_vect, val_seq_vect, val_f_X_flat))
X_val_com_scal = scal.transform(X_val_com)

# Validate the model
val_predictions = rf_model.predict(X_val_com_scal)
val_accuracy = accuracy_score(val_em_Y, val_predictions)
print(f"Validation Accuracy for Random Forest: {val_accuracy:.4f}")

# Make predictions on the test set (ensure consistent sizes)
test_em_df = pd.read_csv("datasets/test/test_emoticon.csv")
test_em_X = test_em_df['input_emoticon'].tolist()

test_seq_df = pd.read_csv("datasets/test/test_text_seq.csv")
test_seq_X = test_seq_df['input_str'].tolist()

test_f = np.load("datasets/test/test_feature.npz", allow_pickle=True)
test_f_X = test_f['features']

# Ensure test dataset sizes match (reduce to 5664 if necessary)
test_em_X = test_em_X[:5664]
test_seq_X = test_seq_X[:5664]
test_f_X = test_f_X[:5664]

# Combine test data
test_em_vect = t_em_vect.transform(test_em_X).toarray()
test_seq_vect = t_seq_vect.transform(test_seq_X).toarray()
test_f_X_flat = test_f_X.reshape(test_f_X.shape[0], -1).astype(np.float32)

X_test_com = np.hstack((test_em_vect, test_seq_vect, test_f_X_flat))
X_test_com_scal = scal.transform(X_test_com)

# Make predictions on the test set
test_predictions = rf_model.predict(X_test_com_scal)

# Save the output to a text file
with open('pred_combined.txt', 'w') as f:
    for label in test_predictions:
        f.write(f"{label}\n")

print("Predictions saved to pred_combined.txt.")


# In[ ]:




