from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import spacy
import pandas as pd
from keybert import KeyBERT
import json

# TODO: Analyse, understand and improve the below code

# Load NLP models
nlp = spacy.load("en_core_web_sm")
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=bert_model)

def feature_engineering(df):
    features = {}

    # TF-IDF - extract keywords and phrases based on counts
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(df["lemmatized_description"])
    features["tfidf_matrix"] = tfidf_matrix
    features["tfidf_terms"] = tfidf_vectorizer.get_feature_names_out()

    # NER-based features - extract named entities
    ner_df = df["cleaned_description"].apply(extract_ner_features).apply(pd.Series)

    # KeyBERT keywords - extract keywords and phrases based on semantic similarity
    df["top_keywords"] = df["lemmatized_description"].apply(lambda x: kw_model.extract_keywords(x, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5))
    df["top_keywords"] = df["top_keywords"].apply(lambda x: [kw[0] for kw in x])
    df["keyword_text"] = df["top_keywords"].apply(lambda x: " ".join(x))

    # Sentence embeddings - extract semantic embeddings for cluster analysis
    embeddings = bert_model.encode(df["lemmatized_description"], show_progress_bar=False)
    features["embeddings"] = embeddings

    # Clustering - apply KMeans clustering to the embeddings to group similar descriptions
    kmeans = KMeans(n_clusters=7, random_state=42)
    cluster_ids = kmeans.fit_predict(embeddings)
    # Get distances to allow for outlier detection or some sort of niche scoring
    distances = kmeans.transform(embeddings).min(axis=1)

    df["cluster_id"] = cluster_ids
    df["distance_to_centroid"] = distances

    # Add on cluster top words for each cluster for filtering/sorting downstream
    cluster_top_keywords = (
        df.groupby("cluster_id")["top_keywords"]
        .apply(lambda lists: pd.Series(lists.sum()).value_counts().head(5).index.tolist())
        .to_dict()
    )
    df["cluster_top_keywords"] = df["cluster_id"].map(cluster_top_keywords)

    df = pd.concat([df, ner_df], axis=1)
    features["df"] = df

    return features

def extract_ner_features(text):
    doc = nlp(text)
    ents = [(ent.label_, ent.text) for ent in doc.ents]

    # Collect presence and first-example features
    labels_of_interest = ["ORG", "MONEY", "DATE", "GPE"]
    counts = {label: 0 for label in labels_of_interest}
    firsts = {label: None for label in labels_of_interest}

    for label, value in ents:
        if label in counts:
            counts[label] += 1
            if firsts[label] is None:
                firsts[label] = value

    # Assemble final feature set
    features = {
        f"has_{label.lower()}": counts[label] > 0 for label in labels_of_interest
    }
    features.update({
        f"first_{label.lower()}": firsts[label] for label in labels_of_interest
    })
    features["num_entities"] = len(ents)
    features["first_10_entities_json"] = json.dumps(ents[:10])

    return features
