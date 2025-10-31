🧠 Deep Learning Projects: CNN (CIFAR-10) & RNN (Breast Cancer)

This repository contains two deep learning experiments demonstrating the power of Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) using real-world datasets.

📸 Project 1: Image Classification with CNN on CIFAR-10
🧩 Overview

This project uses a Convolutional Neural Network to classify images from the CIFAR-10 dataset, which includes 60,000 color images across 10 categories — such as airplanes, automobiles, birds, cats, and ships.

The model learns to automatically extract visual features like edges, shapes, and textures, eventually classifying images with good accuracy.

⚙️ Workflow Summary

Data Preparation – The CIFAR-10 dataset is loaded and normalized so that pixel values fall between 0 and 1.

Model Design – A sequential CNN model is created with convolutional, pooling, and fully connected layers to capture image patterns.

Training – The model is trained on 50,000 images and validated on 10,000 unseen test images.

Evaluation – Model accuracy, loss curves, confusion matrix, and ROC curves are plotted to analyze performance.

Prediction – The trained model can predict the class of any new image uploaded by the user.

Model Saving – The trained model is saved in HDF5 format for reuse or deployment.

🏁 Key Insights

CNNs automatically learn hierarchical visual patterns — from edges to objects.

After training, the model achieves strong performance on most CIFAR-10 classes.

Visualizations such as accuracy/loss graphs, confusion matrices, and ROC curves provide detailed insights into classification performance.

💉 Project 2: Cancer Classification with RNN (LSTM)
🧩 Overview

This project applies a Recurrent Neural Network (RNN) with LSTM (Long Short-Term Memory) layers to classify breast cancer data as benign or malignant using the Breast Cancer Wisconsin dataset from scikit-learn.

Although RNNs are often used for sequential data, this project reshapes tabular data to demonstrate how LSTM cells can identify temporal or structural dependencies within features.

⚙️ Workflow Summary

Data Loading – The Breast Cancer dataset is loaded using scikit-learn.

Preprocessing – Features are standardized using StandardScaler to ensure consistent input ranges.

Reshaping – Data is reshaped into a 3D format suitable for LSTM input.

Model Design – An LSTM layer followed by a dense output neuron predicts the binary cancer label.

Training – The model is trained and validated using an 80/20 split.

Evaluation – The model’s accuracy, F1 score, and ROC-AUC score are computed.

Visualization – ROC curves show strong discrimination between classes.

🏁 Key Insights

LSTM networks can model feature interactions effectively, even in non-sequential data.

The model achieves high accuracy and near-perfect ROC-AUC, indicating excellent classification ability.

The project demonstrates how deep learning can complement traditional ML models in structured data problems.

📊 Results Summary
Model	Dataset	Accuracy	F1-Score	ROC-AUC
CNN	CIFAR-10	~75–80%	—	~0.90
RNN (LSTM)	Breast Cancer	~97%	~0.96	~0.99
📈 Visual Outputs

Both projects include comprehensive visualizations:

Accuracy and Loss Curves – Compare training and validation performance.

Confusion Matrices – Show correct and incorrect predictions by class.

ROC Curves – Illustrate the trade-off between sensitivity and specificity.

🧰 Technologies Used

TensorFlow / Keras – Deep learning framework for building and training models

NumPy & Pandas – Data manipulation and numerical operations

Matplotlib – Visualization of training metrics and results

scikit-learn – Dataset loading and evaluation metrics

Pillow (PIL) – Image processing and saving

🚀 Key Learnings

CNNs excel at spatial feature extraction, making them ideal for image data.

RNNs (LSTMs) excel at sequence and feature dependency modeling.

Proper preprocessing and normalization are critical for stable training.

Visualization helps interpret and debug model performance effectively.

Deep learning models can be reused and deployed easily after saving.
