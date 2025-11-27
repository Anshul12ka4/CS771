#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch
import torch.nn as nn
from torchvision import models

# Define the path for the ResNet-50 weights
weights_path = "resnet50-0676ba61.pth"

# Load ResNet-50 model
model = models.resnet50()  # Initialize ResNet-50
state_dict = torch.load(weights_path)  # Load weights
model.load_state_dict(state_dict)  # Apply weights

# Modify the model to remove the final fully connected layer
model = nn.Sequential(*list(model.children())[:-1])  # Exclude the FC layer
model.eval()  # Set model to evaluation mode

print("ResNet-50 model loaded and modified successfully.")


# In[10]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[11]:


import torch
from torchvision import transforms  # Import the transforms module

# Define the transformation (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x).permute(2, 0, 1)),  # Convert numpy array to tensor (C, H, W)
    transforms.Resize((224, 224)),  # Resizing the image to 224x224 (as required by ResNet)
    transforms.ConvertImageDtype(torch.float32),  # Convert image to float32 tensor
    transforms.Normalize(                         # Normalize using ImageNet mean and std
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

print("Transformations defined successfully.")


# In[12]:


# Feature extraction function
def extract_features(images):
    num_images = len(images)
    features = None  # To initialize the features storage later

    with torch.no_grad():
        for idx, img in enumerate(images):
            img_tensor = transform(img).unsqueeze(0)  # Apply transformation and add batch dimension
            feat = model(img_tensor).view(1, -1)  # Flatten the output to 1D
            
            # Initialize features array only once, based on the output size of the model
            if features is None:
                features = np.zeros((num_images, feat.size(1)))
            
            features[idx] = feat.numpy()  # Assign the flattened features to the pre-allocated array

    return features


# In[13]:


import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Compute initial prototypes
def compute_prototypes(features, labels):
    """
    Group features by class and compute the mean feature vector for each class.
    """
    prototypes = {}
    for cls in set(labels):  # Iterate through unique class labels
        indices = np.where(labels == cls)[0]  # Find indices of features belonging to the class
        cls_features = features[indices]  # Select the features for the class
        prototypes[cls] = cls_features.mean(axis=0)  # Compute and store the mean feature vector
    return prototypes

# Classify a feature based on its distance to class prototypes
def classify_sample(sample_feature, prototypes):
    """
    Determine the nearest prototype for a given sample feature.
    """
    min_distance = float("inf")
    predicted_class = None
    for cls, prototype in prototypes.items():
        distance = np.linalg.norm(sample_feature - prototype)  # Compute Euclidean distance
        if distance < min_distance:
            min_distance = distance
            predicted_class = cls
    return predicted_class

# Update prototypes with new data points
def refine_prototypes(prototypes, new_features):
    """
    Assign new features to the closest prototype and update the prototypes.
    """
    updated_data = {cls: [] for cls in prototypes.keys()}  # Create storage for new class data
    
    # Assign new features to their closest prototype
    for feature in new_features:
        predicted_label = classify_sample(feature, prototypes)
        updated_data[predicted_label].append(feature)
    
    # Compute new prototypes
    refined_prototypes = {}
    for cls, features in updated_data.items():
        if features:  # Ensure the class has assigned features
            features_stack = np.vstack(features)  # Stack features to compute the mean
            refined_prototypes[cls] = features_stack.mean(axis=0)
        else:  # If no new features, retain the old prototype
            refined_prototypes[cls] = prototypes[cls]
    
    return refined_prototypes


# In[14]:


# Evaluate the accuracy of a classifier using prototypes
def calculate_accuracy(features, labels, prototypes):
    """
    Calculate classification accuracy by comparing predictions to true labels.
    """
    total_samples = len(features)
    correct_predictions = sum(
        1 for feature, label in zip(features, labels) 
        if classify_sample(feature, prototypes) == label
    )
    return correct_predictions / total_samples

# Dictionary to store evaluation results for each model and dataset
evaluation_results = {}
feature_storage = {}

# Extract and store features for a specific dataset
def fetch_or_generate_features(dataset_id, dataset_samples):
    """
    Retrieve features from cache if available, otherwise generate them.
    """
    if dataset_id not in feature_storage:
        # Extract features and store them in the cache
        extracted_features = extract_features(dataset_samples)
        feature_storage[dataset_id] = extracted_features
    return feature_storage[dataset_id]


# In[15]:


# Load training data and labels
train_dataset = torch.load('1_train_data.tar.pth')
train_images = train_dataset['data']
train_labels = train_dataset['targets']

# Extract features and calculate prototypes for the training dataset
train_features = extract_features(train_images)
prototypes_train = compute_prototypes(train_features, train_labels)

# Initialize a dictionary to store results for the first model
evaluation_summary = {}
evaluation_summary['Model_A'] = {}

# Load test data and labels
test_dataset = torch.load('1_eval_data.tar.pth')
test_images = test_dataset['data']
test_labels = test_dataset['targets']

# Extract or retrieve cached features for the test dataset
test_features = fetch_or_generate_features("Dataset_A1", test_images)

# Evaluate the classifier's accuracy on the test dataset
classifier_accuracy = calculate_accuracy(test_features, test_labels, prototypes_train)
evaluation_summary['Model_A']['Dataset_A1'] = classifier_accuracy * 100

# Output the accuracy
classifier_accuracy


# In[16]:


import torch
import pandas as pd

# Initialize prototypes with the first training dataset
initial_training_data = torch.load('1_train_data.tar.pth')
initial_images = initial_training_data['data']
initial_labels = initial_training_data['targets']

# Extract features and compute initial prototypes
initial_features = extract_features(initial_images)
current_prototypes = compute_prototypes(initial_features, initial_labels)

# Dictionary to store evaluation results
evaluation_data = {}

# Iterate through training datasets from 2 to 10
for train_index in range(2, 11):  # Adjust range based on the dataset count
    # Load training data for the current index
    training_data = torch.load(f'{train_index}_train_data.tar.pth')
    training_images = training_data['data']

    # Extract features from the training images
    features_to_update = extract_features(training_images)

    # Update prototypes by integrating features from the new dataset
    current_prototypes = refine_prototypes(current_prototypes, features_to_update)

    # Initialize evaluation results for the current model
    model_key = f'Model_{train_index}'
    evaluation_data[model_key] = {}

    # Evaluate the model on all datasets up to the current index
    for test_index in range(1, train_index + 1):
        # Load test data for the current index
        test_data = torch.load(f'{test_index}_eval_data.tar.pth')
        test_images = test_data['data']
        test_labels = test_data['targets']

        # Extract or retrieve features for the test dataset
        test_features = fetch_or_generate_features(f'Dataset_{test_index}', test_images)

        # Evaluate classification accuracy on the test dataset
        accuracy = calculate_accuracy(test_features, test_labels, current_prototypes)

        # Save accuracy results for the current test dataset
        evaluation_data[model_key][f'Dataset_{test_index}'] = accuracy * 100

# Convert eva\luation results to a DataFrame for analysis
evaluation_results_df = pd.DataFrame(evaluation_data).T
print(evaluation_results_df)


# In[ ]:


# Task-2 (Problem 1) 


# In[24]:


import torch
import pandas as pd

# Function to inspect the structure of the loaded data
def check_data_structure(data):
    print("Keys in loaded data:", data.keys())

# Initialize prototypes with the first training dataset (for the 10th dataset)
initial_training_data = torch.load('10_train_data.tar.pth')

# Inspect the structure of the training data
check_data_structure(initial_training_data)

# Assuming 'data' contains the images, but labels might be in a different key (e.g., 'targets', 'labels')
initial_images = initial_training_data['data']

# Try extracting the labels from different potential keys
try:
    initial_labels = initial_training_data['targets']  # Try 'targets' first
except KeyError:
    try:
        initial_labels = initial_training_data['labels']  # Try 'labels' if 'targets' is not found
    except KeyError:
        # If both 'targets' and 'labels' are not present, inspect the structure for debugging
        print("Could not find labels directly. Inspect the structure and adjust accordingly.")
        initial_labels = None

# Check if labels are found
if initial_labels is None:
    print("Please inspect the data structure: ", initial_training_data)
else:
    print(f"Labels extracted: {initial_labels.shape} entries.")

# Check if initial_labels is not None before proceeding
if initial_labels is not None:
    # Extract features from the initial dataset and compute initial prototypes
    initial_features = extract_features(initial_images)
    current_prototypes = compute_prototypes(initial_features, initial_labels)

    # Dictionary to store evaluation results
    evaluation_data = {}

    # Iterate through training datasets from 11 to 20
    for train_index in range(11, 21):  # Adjust range based on the dataset count
        print(f"Loading training data for dataset {train_index}...")
        # Load training data for the current index
        training_data = torch.load(f'{train_index}_train_data.tar.pth')
        training_images = training_data['data']
        
        # Try to extract labels for training data (if they exist)
        try:
            training_labels = training_data['targets']  # Try 'targets' first
        except KeyError:
            try:
                training_labels = training_data['labels']  # Try 'labels' if 'targets' is not found
            except KeyError:
                print(f"Could not find labels for dataset {train_index}.")
                training_labels = None
        
        # Check if labels are available for the current training dataset
        if training_labels is None:
            print(f"Skipping dataset {train_index} as no labels are available.")
            continue

        # Extract features from the training images
        features_to_update = extract_features(training_images)

        # Update prototypes by integrating features from the new dataset
        current_prototypes = refine_prototypes(current_prototypes, features_to_update)

        # Initialize evaluation results for the current model
        model_key = f'Model_{train_index}'
        evaluation_data[model_key] = {}

        # Evaluate the model on all datasets up to the current index
        for test_index in range(1, train_index + 1):  # Evaluate on all datasets from 1 to current train_index
            print(f"Loading test data for dataset {test_index}...")
            # Load test data for the current index
            test_data = torch.load(f'{test_index}_eval_data.tar.pth')
            test_images = test_data['data']

            # Try to extract labels for test data
            try:
                test_labels = test_data['targets']  # Try 'targets' first
            except KeyError:
                try:
                    test_labels = test_data['labels']  # Try 'labels' if 'targets' is not found
                except KeyError:
                    print(f"Could not find labels for dataset {test_index}.")
                    test_labels = None

            # Check if labels are available for the current test dataset
            if test_labels is None:
                print(f"Skipping test dataset {test_index} as no labels are available.")
                continue

            # Extract or retrieve features for the test dataset
            test_features = fetch_or_generate_features(f'Dataset_{test_index}', test_images)

            # Evaluate classification accuracy on the test dataset
            accuracy = calculate_accuracy(test_features, test_labels, current_prototypes)
            print(f"Accuracy for Dataset {test_index}: {accuracy}")

            # Save accuracy results for the current test dataset
            evaluation_data[model_key][f'Dataset_{test_index}'] = accuracy * 100

    # Convert evaluation results to a DataFrame for analysis
    evaluation_results_df = pd.DataFrame(evaluation_data).T
    print(evaluation_results_df)

else:
    print("Skipping processing due to missing labels in the dataset.")


# In[ ]:




