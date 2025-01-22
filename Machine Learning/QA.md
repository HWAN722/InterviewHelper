# Question & Answer

# Machine Learning

## 1. Difference between XGBoost, Lightgbm and GDBT

### XGBoost

XGBoost stands for Extreme Gradient Boosting. It is an advanced version of gradient boosting that incorporates additional features to improve performance and efficiency. These include:

- **Regularization**: L1 and L2 regularization to prevent overfitting.
- **Parallelization**: Parallel tree construction to reduce training time.
- **Sparsity Aware**: XGBoost can handle sparse data directly without needing preprocessing like imputation.
- **Tree Pruning**: It uses max depth for stopping tree growth, whereas traditional GBDT stops growth when further splits add no gain.

**Performance**: XGBoost is highly optimized and generally performs well on most structured/tabular data.

### LightGBM:

LightGBM (Light Gradient Boosting Machine) is a highly efficient gradient boosting framework designed for speed and scalability. It introduces innovations like:

- **Histogram-based learning**: LightGBM uses a histogram to discretize continuous features, reducing computation time for large datasets.
- **Leaf-wise growth**: Instead of growing level-wise (as in traditional GBDT), LightGBM grows trees leaf-wise, which can lead to better accuracy, though it may overfit on smaller datasets.
- **Categorical feature handling**: LightGBM directly handles categorical features without needing one-hot encoding, making it faster and more memory-efficient.

**Performance**: LightGBM is generally faster than XGBoost and performs well on large datasets, especially when there are many features.

### GBDT:

Gradient Boosting Decision Trees (GBDT) is the general framework behind both XGBoost and LightGBM. It constructs an ensemble of decision trees by iteratively fitting new trees to the residuals (errors) of the existing model.

**Limitations**: While GBDT is flexible, it's less optimized in terms of speed and memory efficiency compared to the specialized implementations like XGBoost and LightGBM.

## 2. Difference between LSTM, GRU and RNN

### RNN (Recurrent Neural Network):

- Standard RNNs suffer from the **vanishing gradient problem**, which occurs when the gradients used for updating weights in backpropagation shrink exponentially, making it difficult for the network to learn long-term dependencies.
- RNNs are simple in structure: at each time step, they take an input and the previous hidden state as input to the next time step.
- RNNs are suitable for sequence prediction tasks, but their main weakness is their inability to learn long-term dependencies effectively.

### LSTM (Long Short-Term Memory):

LSTMs were designed to combat the vanishing gradient problem by introducing a memory cell and gating mechanisms (input, forget, and output gates) to control the flow of information.

**Key Features**:
- Forget gate: Decides which information from the previous hidden state should be discarded.
- Input gate: Controls how much new information should be added to the cell state.
- Output gate: Determines the final output based on the memory cell.
LSTMs are particularly useful for tasks where capturing long-term dependencies is critical, such as in speech recognition, text generation, and time series forecasting.

### GRU (Gated Recurrent Unit):

GRU is a simpler version of LSTM, with fewer gates. It combines the forget and input gates into a single update gate, and it also lacks the separate output gate.

**Key Features**:
- Update gate: Decides how much of the previous memory and new information should be carried forward.
- Reset gate: Controls how much of the past information is forgotten.
- GRU is computationally more efficient than LSTM and can perform similarly on many tasks, making it a good choice when resource efficiency is critical.


## 3. Difference between generative model and discriminant model

### Generative Models:

These models learn the joint probability distribution *P*(*X*,*Y*) of the input data *X* and the output labels *Y*. Essentially, they try to learn how the data is generated.
- **Examples**: Naive Bayes, Gaussian Mixture Models (GMM), Hidden Markov Models (HMM), Generative Adversarial Networks (GAN).
- **Advantages**: Generative models can generate new data and are useful when you need to model the data distribution itself. They can be used for data augmentation.
- **Disadvantages**: They are often more complex and require more data to train effectively.

### Discriminative Models:
These models focus on modeling the conditional probability *P*(*X*∣*Y*), which means they learn the boundary between different classes in the feature space. They do not try to model how data is generated, only how to classify or predict labels given inputs.

- **Examples**: Logistic Regression, Support Vector Machines (SVM), Decision Trees, Neural Networks.
- **Advantages**: Discriminative models tend to perform better on classification tasks since they directly model the decision boundary.
- **Disadvantages**: They do not have the ability to generate new samples from the distribution, and they require more labeled data for training.

## 4. Difference between Newton method and gradient descent method

### Gradient Descent:

Gradient Descent is an iterative optimization technique that aims to minimize a function by following the negative gradient of the function with respect to the parameters.

**Formula**:
   
   $$\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla_{\theta} J(\theta)$$

where $\eta$ is the learning rate and $\nabla_{\theta} J(\theta)$ is the gradient of the loss function.

**Advantages**: Simple to implement and computationally efficient.

**Disadvantages**: Slow convergence, especially in high-dimensional spaces. It can also get stuck in local minima.

### Newton's Method:

Newton's Method uses both the gradient and the second derivative (Hessian matrix) to find the minimum of the function. This allows it to converge faster than gradient descent in many cases because it considers the curvature of the loss function.

**Formula**:

$$\theta_{\text{new}} = \theta_{\text{old}} - H^{-1} \nabla_{\theta} J(\theta)$$

where *H* is the Hessian matrix.

**Advantages**: Faster convergence, especially near the optimal point.

**Disadvantages**: Computing the Hessian matrix is computationally expensive, especially for large datasets. It can also be unstable if the Hessian is not positive-definite.

## 5. Difference Between Hard Margin and Soft Margin in SVM

### Hard Margin SVM:

In the case of linearly separable data, hard margin SVM creates a decision boundary that perfectly separates the classes with no errors or misclassifications. The margin between the support vectors (data points closest to the hyperplane) is maximized.

**Limitation**: Hard margin SVM is sensitive to outliers and noise in the data, as it requires perfect separation.

### Soft Margin SVM:

Soft margin SVM allows for some misclassification by introducing a penalty term for each misclassified point. The objective is to find a balance between a wide margin and minimizing classification errors.

**Formula**: The objective function is modified with a regularization term *C*, which controls the trade-off between margin width and classification errors.

**Advantage**: More robust to noise and outliers, making it applicable to non-linearly separable data.

## 6. Difference Between Parametric and Non-parametric Models

### Parametric Models:

Parametric models assume a specific functional form for the data distribution and estimate a fixed number of parameters.

**Examples**: Linear Regression, Logistic Regression, Naive Bayes, Gaussian Mixture Models.

**Advantages**: Computationally efficient, requires less data for training.

**Disadvantages**: May perform poorly if the data does not match the assumed distribution.

### Non-parametric Models:

Non-parametric models do not assume a specific distribution and can adapt to a wide range of data structures.

**Examples**: k-Nearest Neighbors (k-NN), Decision Trees, Random Forests, Kernel Density Estimation.

**Advantages**: More flexible, can fit complex data patterns.

**Disadvantages**: Requires more data to avoid overfitting and can be computationally expensive.

## 7. Difference and Connection Between Word2Vec Methods

Word2Vec methods aim to generate word embeddings, which are dense, low-dimensional vectors that capture the semantic meaning of words. The two primary models are:

- **Continuous Bag of Words (CBOW)**: CBOW tries to predict the target word from the surrounding context words. It takes a fixed-size window of context words and uses the average of their embeddings to predict the target word.
- **Skip-Gram**: Skip-Gram works in the opposite direction of CBOW. It takes the target word and tries to predict the surrounding context words. Skip-Gram performs better on rare words.

**Connection**: Both methods are used for learning word representations in a continuous vector space, but CBOW is faster to train, while Skip-Gram generally produces higher-quality embeddings for infrequent words.

## 8. What are boost methods

Boosting is an ensemble learning technique where multiple weak models (usually decision trees) are combined to form a strong model. The idea is to focus on the mistakes made by previous models in each round and adjust the weights accordingly.
Common boosting algorithms include:
- AdaBoost (Adaptive Boosting): Weights the misclassified instances more heavily in each round.
- Gradient Boosting: Builds trees sequentially to correct residual errors, using gradient descent to optimize the loss function.
- XGBoost and LightGBM: Optimized gradient boosting algorithms that improve training speed, handle missing values, and reduce overfitting.

## 9. What are the solutions to category imbalance?

**Resampling Techniques**:
- Oversampling: Increasing the number of examples in the minority class, often using techniques like SMOTE (Synthetic Minority Over-sampling Technique).
- Undersampling: Reducing the number of examples in the majority class.

**Cost-sensitive Learning**: Adjusting the loss function to penalize misclassifications of the minority class more heavily.

**Anomaly Detection**: Treating the minority class as anomalies, using models like Isolation Forest.

**Ensemble Methods**: Using balanced random forests or EasyEnsemble to handle imbalanced classes.

## 10. Is there any loss of effect in expansion?

**Model Expansion**:
Risk of Overfitting: Expanding the model can lead to overfitting if the model complexity exceeds what is needed for the data. In this case, the model may capture noise instead of the underlying data distribution.

**Feature Expansion**:
Dimensionality Curse: Increasing the number of features (especially irrelevant ones) can lead to a curse of dimensionality, where the model struggles to generalize well due to too many features.

# Deep Learning

## 1. What do transformer use, BN or LN?
Transformer uses **Layer Normalization**.

### Batch Normalization
Batch normalization is to normalize a batch of data. Specifically, for each feature in a batch, the mean and variance are calculated, and these statistics are used to normalize the features.

**Features and limitations:**
- Dependent on batch size: the effect of BN depends on a large batch size to a great extent, and it does not perform well in small batch training.
- Challenge of sequence data: BN may be unstable due to sequence length and position dependence when processing sequence data.
- Inconsistency between training and reasoning: BN uses different statistics in training and reasoning, which may lead to different model behaviors.

### Layer Normalization
Layer normalization is to normalize the feature dimension of each sample. Specifically, for each sample, the mean and variance of all its features are calculated, and then the features are normalized by these statistics.

**Features and limitations:**
- Independent of batch size: the calculation of LN only depends on the characteristics of a single sample, so it is insensitive to batch size, especially suitable for small batch training or variable length input.
- Suitable for sequence model: In Transformer, the input is usually sequence data, and LN can effectively deal with the dependence of each position in the sequence.
- Simplified training: Compared with BN, LN does not need to maintain batch statistics during training and reasoning, which reduces the complexity of implementation.

## 2. Difference between Layer Normalization of GPT3 and LLAMA
GPT3 uses Post-Layer Normalization. The calculation of self-attention or feedforward neural network is carried out first, and then the Layer Normalization is carried out. This structure is helpful to stabilize the training process and improve the model performance.

Llama uses Pre-Layer Normalization. Layer Normalization is performed first, and then the calculation of self-attention or feedforward neural network is performed. This structure helps to improve the generalization ability and robustness of the model.

## 3. Commonly used activation function for LLM

### GeLU(Gaussian Error Linear Unit)
GELU is currently the most widely used activation function in transformer-based LLMs. It was popularized by the BERT model and has since become a standard choice in many architectures, including GPT series.

### ReLU(Rectified Linear Unit)
ReLU is one of the most popular activation functions in deep learning due to its simplicity and effectiveness. While it was foundational in earlier neural network architectures, its usage in LLMs has been somewhat supplanted by GELU.

### Swish
Swish is an activation function proposed by Google researchers and has been considered as an alternative to ReLU and GELU.

| Model               | Activation Function | Notes                                 |
|---------------------|---------------------|---------------------------------------|
| Original Transformer| ReLU                | Introduced in "Attention is All You Need" |
| BERT                | GELU                | Smooth non-linearity enhances performance |
| GPT-3               | GELU                | Consistent with evolved transformer architectures |
| LLaMA               | GELU                | Aligns with state-of-the-art LLM practices |

## 4. How to handle repetition in LLM

### 1. Adjusting Decoding Strategies
- **Temperature Scaling**: A parameter that controls the randomness of the model's predictions. Lower temperatures make the model more deterministic, while higher temperatures increase randomness.
- **Top-k Sampling**: Limits the model to consider only the top k most probable next tokens.
- **Top-p (Nucleus) Sampling**: Selects tokens from the smallest possible set whose cumulative probability exceeds a threshold *p*.

### 2. Training Enhancements
- **Diverse Training Data**: Ensuring the training corpus includes a wide variety of language uses and minimizes repetitive patterns.
- **Data Augmentation**: Introducing variations of sentences and phrases in the training data.

## 5. Formula of Multihead attention, and the reason of divided by square root d_k

### Multihead Attention Formula

Multihead attention is a mechanism that allows the model to focus on different parts of the input sequence simultaneously. It is a key component of the Transformer architecture. The formula for scaled dot-product attention, which is used in multihead attention, is as follows:

Given queries $$ Q $$, keys $$ K $$, and values $$ V $$, the attention output is computed as:

$$ 
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V 
$$

Where:
- $$ Q $$ is the matrix of queries.
- $$ K $$ is the matrix of keys.
- $$ V $$ is the matrix of values.
- $$ d_k $$ is the dimension of the keys (and queries).

### Reason for Dividing by $$\sqrt{d_k}$$

The division by $$\sqrt{d_k}$$ is a scaling factor that is crucial for the stability of the attention mechanism. Here's why it's used:

- 1. **Preventing Large Dot-Product Values**: Without scaling, the dot products $$ QK^T $$ can become very large, especially when the dimensionality $$ d_k $$ is large. This can push the softmax function into regions where it has extremely small gradients, which can slow down learning and make optimization difficult.

- 2. **Stabilizing Gradients**: By scaling the dot products by $$\sqrt{d_k}$$, the values are kept in a range that is more suitable for the softmax function, which helps maintain stable gradients during training.

- 3. **Empirical Performance**: Empirically, it has been observed that this scaling improves the performance of the model, leading to faster convergence and better results.
