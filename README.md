# NLP Technical Assessment - Joseph Marchant

This pipeline transforms raw company descriptions into a structured dataset with semantic, statistical, and named entity features. It's designed to support downstream use cases such as clustering, opportunity scoring, and classification. For example, companies can be grouped into niche categories via semantic clustering, or similar firms can be retrieved using cosine similarity on sentence embeddings.

The key goal given the open style of assignment and limited time was to draw as many useful signals out of the data as possible, whilst keeping it all reproducible and insightful. I have detailed what I would do next given more time at the bottom of this README.md.

The notebooks give a bit more insight into the early steps and some of the decisions made.

---

## How to run

### Setup environment

Work done in Python 3.10.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run the full pipeline

```bash
python main.py --run all
```

### Run specific pipeline stages

```bash
python main.py --run ingest
python main.py --run preprocess
python main.py --run features
python main.py --run features_structure
python main.py --run visualise
```

## Project structure

```text
main project folder
│
├── data/                # Raw input data
├── output/              # Processed outputs and visualisations
├── src/                 # Processing code for various pipeline stages
├── main.py              # Pipeline entry point
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Pipeline stages

- Ingestion: Loads and standardises the raw data, resolving known issues
- Preprocessing: Cleans text, generates document-level stats, tokenizes, and lemmatizes (~30s)
- Feature Engineering: Extracts TF-IDF, keywords, named entities, and embeddings (~60s)
- Structuring: Formats ML-ready data and saves additional matrices
- Visualisation: Generates interpretive charts and cluster-level summaries (~10s)

## Future Work

Given more time, these are the main topics/goals I would have:
- Data enrichment: Extend the dataset using additional sources such as APIs and web scraping. This could include firmographic data (e.g. Companies House), market sentiment (e.g. Google News API), or platform reviews (e.g. Trustpilot, social media). Enriching company context would help sharpen feature signal and real-world relevance.
- Better validation: Deeper validation of each stage, especially cleaning and feature engineering, would allow for iterative refinement. For instance, inspecting masked patterns, checking for character encoding anomalies (e.g. alternate apostrophes), and improving generalisability.
- Relate to business context: Tighter linkage to real business use cases would improve downstream utility. For example, deriving high-level opportunity labels from cluster themes or keywords (e.g. "IPO", "M&A", "growth") could directly support investor workflows.
- Modelling: With labeled targets, applying models like classification (e.g. growth vs turnaround), outlier detection, or ranking would enable more actionable insights. This would also unlock feature importance analyses, revealing which signals matter most.
- Token-level and semantic features: Adding token-level stats (e.g. average token length, noun/verb ratio, vocabulary richness) and semantic summaries (e.g. most similar company, cluster density, embedding norms) would provide finer-grained insight.
- Codebase improvements: With more time, I’d add clearer docstrings, logging, modular configuration, and better runtime flexibility for toggling stages, parameters, and debug outputs.

