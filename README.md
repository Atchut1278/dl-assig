
Project Structure
train.py: The script responsible for training the feedforward neural network on the SVHN dataset, with configurable model hyperparameters.

evaluate.py: This script loads the trained model, evaluates it on the test set, and produces performance metrics such as a confusion matrix.

models.py: Contains functions for building the neural network, allowing flexible configurations (e.g., number of layers, neurons per layer, activation functions).

utils.py: Includes helper functions for loading and preprocessing the SVHN dataset, such as normalization and splitting the data into training, validation, and test sets.

requirements.txt: Specifies the Python packages required for the project.

report.pdf: Contains a detailed summary of the findings, experiments, and recommendations based on the results.

Dataset
The SVHN dataset is automatically downloaded using TensorFlow Datasets (tensorflow_datasets), making it easy to access and load. It is preprocessed by:

Normalizing pixel values to the range [0, 1].
Splitting the dataset into training, validation, and test sets for efficient model evaluation.
Hyperparameter Tuning
The model provides flexible options for tuning several hyperparameters to optimize performance:

Number of Hidden Layers: 3, 4, or 5 layers.
Neurons per Layer: 32, 64, 128.
Optimizers: Options include SGD, Momentum, Nesterov, RMSprop, Adam, and Nadam.
Learning Rates: 1e-3 and 1e-4.
Batch Sizes: 16, 32, or 64.
Activation Functions: ReLU or Sigmoid.
Weight Decay (L2 Regularization): 0, 0.0005, 0.5.
Key Findings from Experiments
Optimal Hyperparameters
The model performed best with 4 hidden layers, 128 neurons per layer, and the Adam optimizer (learning rate of 1e-3).
ReLU activation outperformed Sigmoid, likely due to its ability to mitigate the vanishing gradient problem during training.
Effect of Weight Decay
L2 regularization with weight decay of 0.0005 improved generalization by reducing overfitting.
Larger weight decay values (e.g., 0.5) caused underfitting, which negatively affected model accuracy.
Loss Function Comparison
Cross-entropy loss outperformed mean squared error (MSE) due to the probabilistic nature of classification tasks. MSE doesn't handle categorical outputs well, leading to poorer performance.
Recommendations for MNIST
Based on the findings from the SVHN dataset, we recommend the following model configurations for MNIST classification, depending on the desired balance between performance and efficiency:

Best Performing Model:

Hidden Layers: 4
Neurons per Layer: 128
Optimizer: Adam (learning rate = 1e-3)
Regularization: L2 (0.0005)
Activation: ReLU
Expected Accuracy: ~99%
Efficient Model for Faster Training:

Hidden Layers: 3
Neurons per Layer: 64
Optimizer: RMSprop (learning rate = 1e-3)
Regularization: L2 (0.0005)
Activation: ReLU
Expected Accuracy: ~98.5%
Balanced Model for Generalization:

Hidden Layers: 5
Neurons per Layer: 128, 128, 64, 64, 32
Optimizer: Momentum-based SGD (learning rate = 1e-3)
Regularization: L2 (0.0005)
Activation: ReLU
Expected Accuracy: ~98%
Conclusion
From the experiments on the SVHN dataset, we learned the following:

The Adam optimizer combined with ReLU activation is consistently the best configuration for digit classification tasks, including MNIST.
A moderate level of L2 regularization is essential to avoid overfitting while still allowing for good generalization.
The cross-entropy loss function is more suitable than MSE for classification tasks, particularly for problems with probabilistic outputs.
The findings from the SVHN dataset can be effectively applied to MNIST and similar digit classification datasets. The training and evaluation processes are modular, well-documented, and reproducible.

Evaluation Results
The model achieved high accuracy on the validation set, and the confusion matrix demonstrated strong performance across all digit classes, suggesting that the model generalizes well across different digits.
