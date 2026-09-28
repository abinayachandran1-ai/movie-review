# Movie Review Sentiment Analysis using Word Embeddings and Transformers

### Overview

This project explores automated sentiment classification of movie reviews by comparing traditional Word2Vec embeddings with pre-trained Transformer-based sentence embeddings.

The main objective is to understand how different text representation techniques affect the performance of sentiment classification models. Two embedding approaches and two classifiers are evaluated to identify the combination that provides the strongest sentiment classification performance.

### Problem Statement

In the entertainment industry, understanding audience sentiment from movie reviews can provide useful insights for marketing, content development, and audience analysis. Manually analyzing a large number of reviews is time-consuming and difficult to scale.

This project develops an AI-based sentiment analysis workflow that automatically classifies movie reviews as **positive** or **negative** and compares traditional word-level embeddings with contextual Transformer embeddings.

### Dataset

The dataset contains movie reviews with two columns:

| Column | Description |
|---|---|
| `review` | Textual movie review |
| `sentiment` | Sentiment label: `0` = Negative, `1` = Positive |

The notebook loads the movie review CSV dataset and performs basic data-quality checks. The loaded dataset contains **9,982 reviews and 2 columns** after duplicate handling.

#### Data Quality Checks

- Checked the first few records to understand the dataset structure.
- Checked for missing values.
- Checked for duplicate records.
- Removed duplicate rows where applicable.
- Reset the Data Frame index after cleaning.

### Methodology

The project follows these major stages:

1. **Environment Setup**
2. **Dataset Loading**
3. **Data Exploration and Cleaning**
4. **Text Preprocessing**
5. **Word2Vec Embedding Generation**
6. **Sentence Transformer Embedding Generation**
7. **Random Forest Classification**
8. **Neural Network Classification**
9. **Accuracy Evaluation**
10. **Comparison of Embedding and Classification Approaches**

### Text Representation Techniques

#### 1. Word2Vec

Word2Vec is used to learn numerical representations of individual words based on their contextual relationships in the review dataset.

- Embedding dimension: **300**
- Window size: **5**
- Minimum word count: **1**
- Workers: **6**

The word embeddings are aggregated to create a fixed-length representation for each complete movie review.

#### 2. Sentence Transformer

The project also uses the pre-trained:

`all-MiniLM-L6-v2`

Sentence Transformer to generate contextual sentence-level embeddings.

Unlike simple word-level averaging, Transformer embeddings capture contextual information from the complete sentence/review, allowing the representation to reflect relationships between words and their surrounding context.

### Classification Models

Two classification approaches are evaluated with both types of embeddings:

#### Random Forest

A Random Forest classifier is trained using:

- Word2Vec review embeddings
- Sentence Transformer review embeddings

#### Neural Network

A neural network classifier is trained using:

- Word2Vec review embeddings
- Sentence Transformer review embeddings

The notebook uses TensorFlow/Keras components such as `Sequential`, `Dense`, and `Dropout` for the neural network implementation.

### Model Performance

The evaluated combinations produced the following test accuracies:

| Embedding | Classifier | Test Accuracy |
|---|---|---:|
| Word2Vec | Random Forest | **65.35%** |
| Sentence Transformer | Random Forest | **72.81%** |
| Word2Vec | Neural Network | **73.36%** |
| Sentence Transformer | Neural Network | **78.72%** |

#### Best Performing Model

**Sentence Transformer + Neural Network — 78.72% test accuracy**

This combination achieved the highest test accuracy among the evaluated approaches.

### Key Findings

- Transformer-based sentence embeddings performed better than the Word2Vec-based representations when paired with the same classifier.
- The **Sentence Transformer + Random Forest** combination outperformed **Word2Vec + Random Forest**.
- The **Sentence Transformer + Neural Network** combination achieved the highest test accuracy.
- The results demonstrate the benefit of contextual sentence representations compared with simple word-level embedding aggregation.
- The difference between training and test performance indicates that the models may have some degree of **overfitting**, particularly in the higher-capacity neural-network approach.

### Technologies Used

- **Python**
- **Pandas** — data manipulation
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualization
- **Gensim** — Word2Vec embeddings
- **Sentence Transformers** — contextual sentence embeddings
- **Scikit-learn** — data splitting, Random Forest, and evaluation
- **TensorFlow / Keras** — neural network classification
- **PyTorch** — backend used by Sentence Transformers
- **Google Colab / Google Drive** — notebook execution and dataset access

### Required Python Libraries

The notebook uses the following main packages:

```text
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.6.1
scipy==1.13.1
gensim==4.3.3
sentence-transformers==3.4.1
accelerate==1.7.0
sentencepiece==0.2.0
tensorflow
torch
matplotlib
seaborn
```

### How to Run

#### 1. Clone or download the repository

```bash
git clone https://github.com/dhxrshan-r/Movie-Review-Sentiment-Analysis-using-Word-Embeddings-and-Transformers.git
cd <your-repository-folder>
```

#### 2. Install the required dependencies

```bash
pip install numpy==1.26.4 pandas==2.2.2 scikit-learn==1.6.1 scipy==1.13.1 gensim==4.3.3 sentence-transformers==3.4.1 accelerate==1.7.0 sentencepiece==0.2.0 tensorflow torch matplotlib seaborn
```

#### 3. Prepare the dataset

Place the movie review CSV file in the location expected by the notebook, or update the dataset path in the data-loading cell.

The notebook currently loads:

```python
reviews = pd.read_csv("/content/drive/MyDrive/GenAI_Class/Dataset/movie_review.csv")
```

If you are using Google Colab, mount Google Drive before running the dataset-loading cell.

#### 4. Run the notebook

Open:

```text
Movie_Reviews_Sentiment_Analysis.ipynb
```

Run the cells sequentially to reproduce the preprocessing, embedding generation, model training, and evaluation workflow.

### Project Structure

```text
.
├── Movie_Reviews_Sentiment_Analysis.ipynb
├── movie_review.csv
└── README.md
```

> The dataset file is not necessarily included in the repository. If it is stored separately, update the path in the notebook before execution.

### Conclusion

This analysis shows that the choice of text representation has a significant effect on sentiment classification performance. While Word2Vec provides useful word-level semantic representations, the contextual representations generated by **Sentence Transformer (`all-MiniLM-L6-v2`)** provide stronger performance for this task.

Among the four evaluated combinations, **Sentence Transformer embeddings combined with a Neural Network achieved the best test accuracy of 78.72%**.

The comparison highlights the practical advantage of contextual Transformer embeddings for understanding the semantic information present in movie reviews, while also indicating the importance of monitoring the gap between training and test performance to reduce overfitting.

### Future Improvements

Potential improvements include:

- Hyperparameter tuning for Random Forest and Neural Network models.
- Applying regularization and early stopping to reduce overfitting.
- Evaluating additional metrics such as precision, recall, and F1-score.
- Using cross-validation for more robust model comparison.
- Comparing additional Transformer-based embedding models.
- Performing more advanced text preprocessing and error analysis.
- Expanding the dataset to improve model generalization.
