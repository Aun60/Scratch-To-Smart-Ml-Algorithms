# Machine Learning Algorithms from Scratch

This mini-project explores the foundational machine learning algorithms by implementing them from scratch and evaluating them on well-known datasets. The goal is to deepen the understanding of how these algorithms work under the hood and assess their real-world performance.

## 🔍 Objective

To implement, visualize, and evaluate classic machine learning algorithms including the Perceptron, Backpropagation for neural networks, Naive Bayes classifier, PCA, and Hierarchical Clustering using real datasets like MNIST, CIFAR-10, 20 Newsgroups, Fashion-MNIST, and Breast Cancer.

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

---

### 4. Principal Component Analysis (PCA)
- **Dataset**: Fashion-MNIST
- **Goal**: Dimensionality reduction and image reconstruction using PCA
- **Tasks**:
  - Reduce to 2 principal components
  - Visualize cluster separation in 2D space
  - Reconstruct images from reduced dimensions
  - **Evaluation**:
    - Class separability in 2D
    - Reconstruction error vs number of components
    - Explained variance ratio plot

---

### 5. Hierarchical Clustering (Agglomerative and Divisive)
- **Dataset**: Breast Cancer
- **Goal**: Cluster the dataset using hierarchical clustering
- **Techniques**:
  - Agglomerative and divisive clustering
  - Various linkage methods: single, complete, average
  - Visualization via dendrogram and PCA projection
- **Evaluation**:
  - Silhouette scores for cluster quality
  - Comparison of clustering effectiveness
  - Insights on inter-cluster distances and structure

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
