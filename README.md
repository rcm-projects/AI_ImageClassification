# Image Classification Using Ai Models

## Contents
* `data_processing.py`: handles CIFAR 10 loading, ResNet 18 feature extraction, and PCA.
* `models.py`: contains class definitions for Naive Bayes, Decision Tree, MLP, and VGG11.
* `main.py`: main execution script for training and evaluation.
* `*.pth`: saved PyTorch models.

# Models

<ins>Naive Bayes & Decision Tree:</ins></br>
The input data for these models consist of feature vectors extracted from ResNet-18 (resized to 224x224) and reduced to 50 dimensions using PCA2222.</br>
Naive Bayes: Two variants were implemented. A custom Gaussian Naive Bayes using Python and numpy to calculate class means and variances, and the Scikit-learn GaussianNB implementation.</br>
Decision Tree: A custom implementation using the Gini impurity criterion was developed recursively. A variant using Scikit-learn's DecisionTreeClassifier with max_depth=50 was also trained.

<ins>Multi-Layer Perceptron (MLP):</ins></br>
The MLP architecture processes the 50-dimensional feature vectors.</br>
Structure: Linear(50, 512) -> ReLU -> Linear(512, 512) -> BatchNorm(512) -> ReLU ->  Linear(512, 10).</br>
Training: Optimized using SGD with momentum 0.9 and CrossEntropyLoss over 20 epochs.</br>
Variants: Experiments were conducted by varying the hidden layer size (e.g., 256 vs 512) and depth. Deeper networks (adding a 3rd hidden layer) generally increased training time but led to slight overfitting on this small dataset.</br>

<ins>Convolutional Neural Network (CNN):</ins></br>
The CNN utilizes the VGG11 architecture trained on raw 32x32 RGB images.
•	Structure: It follows the VGG11 layout (8 Convolutional blocks followed by 3 Fully Connected layers) as specified in the assignment. Key features include 3x3 kernels, padding of 1, and MaxPooling layers.</br>
•	Training: SGD optimizer (momentum 0.9), CrossEntropyLoss, trained for 10 epochs.</br>
•	Variants: Experiments adjusting kernel sizes to 5x5 resulted in a loss of fine-grained detail detection, lowering accuracy slightly compared to the 3x3 standard.


# Output Results:

<img width="323" height="317" alt="Image" src="https://github.com/user-attachments/assets/b02b3e12-bcea-4573-9453-7c35e52bf062" /></br>

The best performing model, MLP, achieved the highest accuracy at 82.4%. This performance is attributed to the input data. The MLP was trained on high level feature vectors extracted by a pre trained ResNet18. Even though the MLP architecture is simple it benefited from the visual features it learned by ResNet on the ImageNet dataset.</br>

The CNN Performance (VGG11) achieved the lowest accuracy (57.9%). While CNNs usually excel at image classification, this specific model was trained from scratch using raw pixels on a very limited dataset (only 500 images per class). Deep architectures like VGG11 require significantly larger datasets to learn effective feature representations without overfitting. This highlights the effectiveness of Transfer Learning (used in the MLP) versus training from scratch when data is scarce.</br>

Naive Bayes: The Gaussian Naive Bayes models performed well at 79.1%. This suggests that the features extracted by ResNet 18 and reduced via PCA are well-distributed and linearly separable, adhering well to the Gaussian assumption.</br>

Decision Trees: Both decision tree variants struggled at approx. 59-61%, likely due to the high dimensionality of the data and the inability of axis aligned splits to capture complex relationships in the state space compared to the gradient based learning of the MLP.</br>

Consistency: The custom implementations of Naive Bayes and Decision Trees had similar results to the Scikit-Learn implementations, validating the correctness of the custom algorithms.</br>



# References

<a href="https://cave.cs.toronto.edu/kriz/learning-features-2009-TR.pdf">Learning Multiple Layers of Features from Tiny Images<a>, Alex Krizhevsky, 2009.
