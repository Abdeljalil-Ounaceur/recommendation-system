# Technical Documentation: Level 1 - Custom Matrix Factorization

## Overview

This document provides detailed technical information about the first implementation of the recommendation system using custom matrix factorization on the MovieLens 100K dataset.

## Mathematical Foundation

### Matrix Factorization Model

The recommendation problem is formulated as a matrix completion task where we have a sparse user-item rating matrix R ∈ ℝ^(m×n) where:
- m = number of users (943)
- n = number of items (1681)
- R_ij = rating of user i for item j (1-5 scale)

The goal is to decompose R into two low-rank matrices:
```
R ≈ U × V^T
```

Where:
- U ∈ ℝ^(m×k) is the user embedding matrix
- V ∈ ℝ^(n×k) is the item embedding matrix  
- k is the embedding dimension (100)

### Prediction Function

For a user i and item j, the predicted rating is:
```
r̂_ij = u_i^T × v_j
```

Where:
- u_i is the embedding vector for user i
- v_j is the embedding vector for item j

### Loss Function

We use Mean Squared Error (MSE) as the loss function:
```
L = (1/|Ω|) × Σ_{(i,j)∈Ω} (r_ij - r̂_ij)²
```

Where:
- Ω is the set of observed ratings
- r_ij is the actual rating
- r̂_ij is the predicted rating

## Implementation Details

### Data Preprocessing

1. **Data Loading**: Load MovieLens 100K dataset
   - `u.data`: User-item ratings (100,000 ratings)
   - `u.item`: Movie metadata with genres
   - `u.user`: User metadata

2. **Data Cleaning**:
   - Remove missing values
   - Merge rating and movie data
   - Create user and item ID mappings

3. **Train-Test Split**: 80-20 split with random state 42

### Model Architecture

#### Embedding Initialization
```python
# Random initialization with small values
user_embeddings = np.random.rand(num_users, embedding_dim) * 0.01
item_embeddings = np.random.rand(num_items, embedding_dim) * 0.01
```

#### Training Algorithm: Stochastic Gradient Descent

**Forward Pass**:
```python
def predict_rating(user_index, movie_index, user_embeddings, item_embeddings):
    return np.dot(user_embeddings[user_index], item_embeddings[movie_index])
```

**Loss Calculation**:
```python
def mse_loss(predicted_rating, actual_rating):
    return (predicted_rating - actual_rating)**2
```

**Gradient Computation**:
```python
def calculate_gradients(prediction, actual_rating, user_embedding, item_embedding):
    error = prediction - actual_rating
    user_grad = 2 * error * item_embedding
    item_grad = 2 * error * user_embedding
    return user_grad, item_grad
```

**Parameter Update**:
```python
def update_embeddings(user_embedding, item_embedding, user_grad, item_grad, learning_rate):
    new_user_embedding = user_embedding - learning_rate * user_grad
    new_item_embedding = item_embedding - learning_rate * item_grad
    return new_user_embedding, new_item_embedding
```

### Training Process

#### Hyperparameters
- **Learning Rate**: 0.01
- **Embedding Dimension**: 100
- **Number of Epochs**: 20 (with early stopping)
- **Early Stopping Patience**: 3 epochs
- **Batch Size**: 1 (stochastic gradient descent)

#### Training Loop
1. **Epoch Loop**: For each epoch (1-20)
2. **Data Shuffling**: Randomly shuffle training data
3. **Sample Loop**: For each (user, item, rating) in training data
   - Get user and item indices
   - Predict rating
   - Calculate loss
   - Compute gradients
   - Update embeddings
4. **Validation**: Evaluate on test set
5. **Early Stopping**: Stop if no improvement for 3 epochs

#### Early Stopping Strategy
- Monitor test RMSE
- Save best embeddings when RMSE improves
- Stop training if no improvement for 3 consecutive epochs

## Performance Analysis

### Training Metrics
- **Final Training RMSE**: 0.7170
- **Final Test RMSE**: 0.9388
- **Training Time**: ~16 epochs (early stopping)
- **Overfitting Gap**: 0.2218 (test - train RMSE)

### Model Convergence
```
Epoch 1:  Training Loss: 5.9049, Test RMSE: 1.1865
Epoch 2:  Training Loss: 1.1265, Test RMSE: 1.0306
Epoch 3:  Training Loss: 0.9830, Test RMSE: 1.0044
Epoch 4:  Training Loss: 0.9492, Test RMSE: 0.9921
Epoch 5:  Training Loss: 0.9320, Test RMSE: 0.9899
Epoch 6:  Training Loss: 0.9187, Test RMSE: 0.9842
Epoch 7:  Training Loss: 0.8963, Test RMSE: 0.9712
Epoch 8:  Training Loss: 0.8647, Test RMSE: 0.9653
Epoch 9:  Training Loss: 0.8286, Test RMSE: 0.9589
Epoch 10: Training Loss: 0.7842, Test RMSE: 0.9521
Epoch 11: Training Loss: 0.7290, Test RMSE: 0.9451
Epoch 12: Training Loss: 0.6678, Test RMSE: 0.9397
Epoch 13: Training Loss: 0.5995, Test RMSE: 0.9388 ← Best
Epoch 14: Training Loss: 0.5279, Test RMSE: 0.9408
Epoch 15: Training Loss: 0.4560, Test RMSE: 0.9438
Epoch 16: Training Loss: 0.3895, Test RMSE: 0.9498
```

### Dataset Statistics
- **Total Users**: 943
- **Total Movies**: 1,681
- **Total Ratings**: 100,000
- **Sparsity**: ~93.7% (very sparse matrix)
- **Rating Distribution**: 1-5 stars
- **Average Rating**: ~3.53

## Recommendation Generation

### Algorithm
1. **User Selection**: Choose target user ID
2. **Movie Filtering**: Identify movies not rated by user
3. **Rating Prediction**: Predict ratings for all unrated movies
4. **Sorting**: Sort movies by predicted rating (descending)
5. **Top-K Selection**: Return top N recommendations

### Example Output
For User ID 1, top 10 recommendations:
```
1. Close Shave, A (1995): 5.1964
2. Pather Panchali (1955): 4.9437
3. Leaving Las Vegas (1995): 4.9277
4. Secrets & Lies (1996): 4.8396
5. Duck Soup (1933): 4.8132
6. Ran (1985): 4.8019
7. Lawrence of Arabia (1962): 4.7896
8. City of Lost Children, The (1995): 4.7678
9. Wings of Desire (1987): 4.7400
10. Faust (1994): 4.7185
```

## Web Interface (Gradio)

### Architecture
- **Framework**: Gradio
- **Input**: User ID (number input)
- **Output**: List of recommended movies with predicted ratings
- **Features**: Error handling for invalid user IDs

### Code Structure
```python
def recommend_movies(user_id, top_n=10):
    # Get user index
    # Predict ratings for all movies
    # Sort by predicted rating
    # Return top N recommendations
```

## Model Persistence

### Saved Files
- `user_embeddings.npy`: User embedding matrix (943 × 100)
- `item_embeddings.npy`: Item embedding matrix (1681 × 100)
- `user_to_index.pkl`: User ID to index mapping
- `movie_to_index.pkl`: Movie ID to index mapping

### Loading Process
```python
user_embeddings = np.load('user_embeddings.npy')
item_embeddings = np.load('item_embeddings.npy')
with open('user_to_index.pkl', 'rb') as f:
    user_to_index = pickle.load(f)
with open('movie_to_index.pkl', 'rb') as f:
    movie_to_index = pickle.load(f)
```

## Limitations and Future Improvements

### Current Limitations
1. **Cold Start Problem**: Cannot handle new users or items
2. **No Regularization**: Risk of overfitting
3. **No Bias Terms**: Missing global, user, and item biases
4. **Simple Loss Function**: Only MSE, no ranking loss
5. **No Content Features**: Ignores movie genres and metadata
6. **Scalability**: Not optimized for large datasets

### Planned Improvements (Level 2)
1. **Add Regularization**: L2 regularization to prevent overfitting
2. **Bias Terms**: Include global, user, and item bias terms
3. **Content Features**: Incorporate movie genres and metadata
4. **Advanced Metrics**: Precision@K, Recall@K, NDCG
5. **Cross-validation**: More robust evaluation
6. **Hyperparameter Tuning**: Optimize learning rate and embedding dimension

## Code Quality and Best Practices

### Code Organization
- **Modular Functions**: Separate functions for prediction, loss, gradients
- **Clear Documentation**: Docstrings for all functions
- **Error Handling**: Validation for user/movie existence
- **Reproducibility**: Fixed random seeds

### Performance Optimizations
- **Vectorized Operations**: NumPy for efficient matrix operations
- **Early Stopping**: Prevents overfitting and reduces training time
- **Memory Efficient**: In-place updates for embeddings

### Testing and Validation
- **Train-Test Split**: Proper evaluation setup
- **Multiple Metrics**: RMSE for regression evaluation
- **Visualization**: Training curves and data exploration

## References

1. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. Computer, 42(8), 30-37.
2. MovieLens 100K Dataset: https://grouplens.org/datasets/movielens/100k/
3. Gradio Documentation: https://gradio.app/docs/
4. NumPy Documentation: https://numpy.org/doc/
