# Implementing-a-CNN-for-Digit-Classification
### Introduction
This project involves the implementation of a Convolutional Neural Network (CNN) to classify images from the Street View House Numbers (SVHN) dataset. The SVHN dataset consists of images of digits (0-9) captured from house number plates in urban settings. The goal of this project is to build a model that can accurately classify these digits using CNN.
________________________________________
### Dataset
Dataset: Street View House Numbers (SVHN)
The SVHN dataset provides over 600,000 labeled digit images, with each image being 32x32 pixels in RGB format. The labels correspond to the digits from 0 to 9, and the dataset is divided into training, testing, and additional sets.
________________________________________
### Project Workflow
##### 1. Importing Necessary Libraries
Several essential Python libraries were imported for this project, including:
•	NumPy: For numerical computations.
•	TensorFlow & Keras: For building and training the CNN model.
•	Matplotlib: For visualizing the results.
•	scipy.io.loadmat: For loading the SVHN dataset, which is in .mat format.
##### 2. Loading and Preparing the Dataset
The dataset was loaded using the loadmat() function from the SciPy library. This function allows us to extract the training and testing data.
•	X_train, X_test: These are the input image datasets.
•	y_train, y_test: These are the corresponding labels for the input data.
The data was checked for its shape and confirmed that each image has dimensions of 32x32 pixels with 3 color channels (RGB).
##### 3. Preprocessing the Data
•	Reshaping: The images were reshaped using the moveaxis function to shift the channels to the last axis, which is the format expected by Keras for input data.
•	Normalization: Pixel values were normalized by dividing by 255, scaling them to the range [0, 1]. This helps in faster convergence during training.
•	Label Correction: In the SVHN dataset, the digit '0' was represented as the label '10'. This was corrected to ensure labels are within the range [0, 9].
•	One-Hot Encoding: The labels were one-hot encoded. For example, the label for the digit '2' was converted into the vector [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].
##### 4. Model Architecture Design
The CNN model architecture included the following components:
•	Convolutional Layers: Three Conv2D layers with increasing filters (32, 64, 128) were used for feature extraction.
•	MaxPooling Layers: These layers reduced the spatial dimensions, thereby downsampling the feature maps.
•	Flattening Layer: This layer converted the 2D feature maps into a 1D vector.
•	Dense Layers: Two fully connected layers were used for classification, with a Softmax output layer to predict the digit classes (0-9).
##### 5. Model Compilation
The model was compiled using:
•	Optimizer: Adam optimizer was used for adjusting the learning rate dynamically.
•	Loss Function: Categorical Crossentropy was used as the loss function, which is well-suited for multi-class classification.
•	Metrics: Accuracy was chosen as the performance metric.
##### 6. Training the Model
The model was trained for 10 epochs with a batch size of 64. During training, the model's performance was validated using a 20% split of the training data.
##### 7. Model Evaluation
After training, the model was evaluated on the test dataset. The model achieved a good classification accuracy, demonstrating its ability to generalize well on unseen data.
•	Test Accuracy: The test accuracy was printed after evaluation.
##### 8. Predictions
The model was tested on the SVHN test set, and predictions were made using the argmax() function, which converted one-hot encoded vectors back into their respective digit labels.
##### 9. Confusion Matrix
A confusion matrix was generated to visualize how well the model performed in terms of correctly classifying the digits and where it got confused. This helped in identifying areas where the model needs improvement.
##### 10. Visualizing Predictions
Test samples were visualized, showing the true digit labels and the predicted labels from the model. This helped in verifying the model's output on individual test images.
________________________________________
### Conclusion
The CNN model built in this project successfully classified the SVHN dataset with high accuracy. By using key techniques like convolutional layers, pooling, and normalization, the model effectively extracted features and learned the patterns needed to distinguish between digits in various real-world scenarios.

