# 🌟 Yelp Stars Predictor & Recommendation Engine

### Overview
A hybrid machine learning recommendation system that predicts user ratings for businesses on Yelp. This project goes beyond simple collaborative filtering by combining the **PySpark** library with **NLP Sentiment Analysis** and gradient boosting algorithms to solve the cold-start problem and deliver highly accurate rating predictions.

### Architecture
This system utilizes a hybrid approach:
* **Item-Based Collaborative Filtering:** Computes Pearson correlation to find similar businesses based on co-rated user histories.
* **Feature-Based Machine Learning:** Extracts user/business metrics and utilizes a **CatBoost Regressor** to predict ratings based on broader metadata.
* **Sentiment Analysis:** Analyzes the text of Yelp "tips" using a custom lexicon to gauge implicit user sentiment.
* **Ensemble Engine:** Intelligently weights and combines the CF and CatBoost predictions based on data availability and confidence.


### 📂 Project Structure

| File | Description |
| :--- | :--- |
| `recommender.py` | The core Python script that implements the hybrid recommendation system, handling data ingestion, model training, and final ensembling. |
| `keywords.py` | An NLP script that parses review text to identify and extract the most common positive and negative keywords for sentiment scoring. |
| `gridsearch.py` | A utility script used to perform hyperparameter tuning to find the optimal configuration for the CatBoost model. |
| `requirements.txt` | A list of external Python dependencies required to run the project. |

### 💻 Usage & Execution

Because this script utilizes PySpark and requires specific dataset paths as arguments, you must run it from the terminal with the correct file paths provided in order.

**1. Install Requirements**

Ensure you have the required libraries installed in your Python environment:
```bash
pip install -r requirements.txt
```

**2. Run the Scripts**

In the terminal run:
```bash
spark-submit recommender.py yelp_data yelp_val.csv comp_output.csv
```
### 🚀 Future Work: From Prediction to True Recommendation
Currently, the base engine is complete and accurately predicts ratings for user-business pairs. However, a true recommendation system serves curated content rather than just predicting hypothetical scores. 

The next phase of this project involves transforming these predictions into a **Top-N Recommendation System** with the following planned features:

* **Location-Based Filtering:** Automatically gather unvisited restaurants in a user's specific area, score them using the hybrid model, and serve a ranked "Top N" list.
* **Cuisine-Specific Curations:** Allow users to filter by cuisine tags (e.g., "Top 5 Mexican Restaurants you haven't tried").
* **The "Spontaneous" Feature:** Implement a randomizer that selects 1 or more highly-predicted restaurants (surpassing a specific rating threshold) to encourage fun, spontaneous dining experiences.

## 💾 Dataset Setup
Due to GitHub's file size limits, the 2.1 GB Yelp dataset is hosted externally. To run this project locally, please follow these steps:

1. Download the dataset folder from [Google Drive here](https://drive.google.com/drive/folders/1cmzAAnM6qsgNQ__5zXvu5po_w2WVnTmt?usp=drive_link).
2. Extract the files.
3. Place the downloaded files into a folder named `yelp_data` within the root directory of this project.

When you are finished, your project directory should look exactly like this:

```text
your-repository-name/
├── recommender.py
├── keywords.py
├── gridsearch.py
├── requirements.txt
├── yelp_val.csv
└── yelp_data/
    ├── business.json
    ├── tip.json
    ├── user.json
    └── yelp_train.csv
