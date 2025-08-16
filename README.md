# Recommendation System Project

This repository contains a collection of recommendation system implementations, starting from a basic matrix factorization approach and evolving to more sophisticated methods.

## Project Structure

```
recommendation-system/
├── level-1-custom-matrix-factorization-ml-100k-dataset/  # First implementation
│   ├── custom-matrix-factorization.ipynb                 # Jupyter notebook with implementation
│   └── gradio/                                          # Web interface
│       ├── app.py                                       # Gradio web app
│       ├── user_embeddings.npy                          # Trained user embeddings
│       ├── item_embeddings.npy                          # Trained item embeddings
│       ├── user_to_index.pkl                            # User ID to index mapping
│       └── movie_to_index.pkl                           # Movie ID to index mapping
├── level-2-*                                            # Future: Advanced collaborative filtering
├── level-3-*                                            # Future: Deep learning approaches
├── level-4-*                                            # Future: Hybrid systems
└── README.md                                            # This file
```

## Level 1: Custom Matrix Factorization (Current Implementation)

### Overview
This is the first version of the recommendation system, implementing matrix factorization from scratch using the MovieLens 100K dataset. It demonstrates the fundamental concepts of collaborative filtering through user and item embeddings.

### Key Features
- **Custom Matrix Factorization**: Implemented from scratch using NumPy
- **Stochastic Gradient Descent**: Optimized training with early stopping
- **MovieLens 100K Dataset**: Uses the classic movie rating dataset
- **Web Interface**: Gradio-based web application for easy interaction
- **Performance Metrics**: RMSE evaluation on test set

### Technical Details

#### Model Architecture
- **Embedding Dimensions**: 100-dimensional user and item embeddings
- **Loss Function**: Mean Squared Error (MSE)
- **Optimization**: Stochastic Gradient Descent with learning rate 0.01
- **Training**: 16 epochs with early stopping (patience=3)

#### Performance
- **Test RMSE**: 0.9388
- **Training RMSE**: 0.7170
- **Dataset**: 943 users, 1681 movies, ~100K ratings

#### Implementation Highlights
1. **Data Preprocessing**: Handles missing values and creates user-item matrices
2. **Embedding Initialization**: Random initialization with small values
3. **Training Loop**: Implements SGD with early stopping to prevent overfitting
4. **Recommendation Generation**: Predicts ratings for unrated movies and sorts by predicted rating
5. **Web Interface**: User-friendly Gradio app for getting recommendations

### Usage

#### Running the Jupyter Notebook
```bash
cd level-1-custom-matrix-factorization-ml-100k-dataset
jupyter notebook custom-matrix-factorization.ipynb
```

#### Running the Web Interface
```bash
cd level-1-custom-matrix-factorization-ml-100k-dataset/gradio
python app.py
```

The web interface will be available at `http://127.0.0.1:7860` and provides a public URL for sharing.

### Files Description

#### `custom-matrix-factorization.ipynb`
- Complete implementation of matrix factorization from scratch
- Data exploration and visualization
- Model training with early stopping
- Performance evaluation
- Recommendation generation

#### `gradio/app.py`
- Web interface for the recommendation system
- Loads trained embeddings and mappings
- Provides user-friendly interface for getting recommendations

#### Model Files
- `user_embeddings.npy`: Trained user embedding matrix (943 × 100)
- `item_embeddings.npy`: Trained item embedding matrix (1681 × 100)
- `user_to_index.pkl`: Mapping from user IDs to matrix indices
- `movie_to_index.pkl`: Mapping from movie IDs to matrix indices

## Future Implementations (Planned)

### Level 2: Advanced Collaborative Filtering
- **SVD++ and NMF**: More sophisticated matrix factorization techniques
- **Neighborhood-based methods**: User-based and item-based collaborative filtering
- **Content-aware filtering**: Incorporating movie genres and metadata
- **Evaluation metrics**: Precision@K, Recall@K, NDCG

### Level 3: Deep Learning Approaches
- **Neural Collaborative Filtering (NCF)**: Deep learning for recommendation
- **Autoencoders**: Denoising autoencoders for collaborative filtering
- **Graph Neural Networks**: Modeling user-item interactions as a graph
- **Attention mechanisms**: Self-attention for recommendation

### Level 4: Hybrid Systems
- **Multi-modal recommendations**: Combining text, image, and interaction data
- **Sequential recommendations**: Modeling temporal patterns in user behavior
- **Contextual recommendations**: Incorporating time, location, and context
- **Real-time recommendations**: Online learning and incremental updates

### Level 5: Production-Ready Systems
- **Scalable architectures**: Distributed training and inference
- **A/B testing framework**: Evaluating recommendation algorithms
- **Personalization**: User-specific model adaptation
- **Explainable AI**: Interpretable recommendation explanations

## Dataset Information

### MovieLens 100K Dataset
- **Users**: 943
- **Movies**: 1,682
- **Ratings**: 100,000
- **Rating Scale**: 1-5 stars
- **Genres**: 18 different movie genres
- **Time Period**: 1997-1998

## Dependencies

### Core Dependencies
- `numpy==1.26.4`: Numerical computations
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning utilities
- `matplotlib`: Data visualization

### Web Interface
- `gradio`: Web application framework

### Optional Dependencies
- `torch`: For future deep learning implementations
- `scikit-surprise`: For comparison with established recommendation algorithms

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd recommendation-system
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the first implementation**:
   ```bash
   cd level-1-custom-matrix-factorization-ml-100k-dataset
   jupyter notebook custom-matrix-factorization.ipynb
   ```

4. **Launch the web interface**:
   ```bash
   cd gradio
   python app.py
   ```

## Contributing

This project is designed to showcase the evolution of recommendation systems. Each level builds upon the previous one, demonstrating more sophisticated approaches to the recommendation problem.

## License

[Add your license information here]

## Acknowledgments

- MovieLens dataset for providing the benchmark dataset
- The recommendation systems research community for foundational algorithms
- Gradio for the web interface framework
