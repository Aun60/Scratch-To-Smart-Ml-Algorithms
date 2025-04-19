# Machine Learning Algorithms from Scratch

This mini-project explores the foundational machine learning algorithms by implementing them from scratch and evaluating them on well-known datasets. The goal is to deepen the understanding of how these algorithms work under the hood and assess their real-world performance.

## 🔍 Objective

To implement, visualize, and evaluate classic machine learning algorithms including the Perceptron, Backpropagation for neural networks, Naive Bayes classifier using real datasets like MNIST, CIFAR-10, 20 Newsgroups, Fashion-MNIST, and Breast Cancer.

---

## 📂 Project Structure

### 1. Perceptron Learning Algorithm
- **Dataset**: MNIST (filtered to digits 0 and 1)
- **Goal**: Binary classification using the Perceptron algorithm
- **Steps**:
  - Data preprocessing and label filtering
  - Implementation of the perceptron learning rule
  - Dimensionality reduction via PCA for 2D visualization
  - Visualization of decision boundary
  - **Discussion**: Limitations of perceptron in high-dimensional data spaces

MNIST DATASET: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
---

### 2. Backpropagation on a Multilayer Neural Network
- **Dataset**: CIFAR-10
- **Goal**: Classify images into 10 classes using a two-layer neural network (from scratch)
- **Features**:
  - Custom forward and backward propagation
  - Support for multiple activation functions: ReLU, Sigmoid, Tanh
  - Weight initialization and gradient descent-based optimization
- **Evaluation**:
  - Accuracy per activation function
  - Training time analysis
  - Loss curve plots

CIFAT-10 DATASET: https://www.kaggle.com/datasets/krishnaharjai/cifar-10
---

### 3. Naive Bayes Classifier
- **Dataset**: 20 Newsgroups
- **Goal**: Text classification using Naive Bayes (from scratch)
- **Steps**:
  - Text preprocessing: lowercasing, stopword removal, TF-IDF vectorization
  - Implementation of Multinomial Naive Bayes
  - Evaluation on different subsets of newsgroups
  - **Analysis**:
    - Confusion matrix for frequently confused categories
    - Accuracy per class

20 NEWSGROUPS DATASET:  https://www.kaggle.com/datasets/snap/20-newsgroups
---

## 🧰 Libraries Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- NLTK / spaCy (for text preprocessing)

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/ml-algorithms-from-scratch.git
cd ml-algorithms-from-scratch

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install required libraries
pip install -r requirements.txt
