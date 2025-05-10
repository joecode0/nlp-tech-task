from langdetect import detect_langs
import re
import spacy

nlp = spacy.load("en_core_web_sm")

MASK_AND_EXTRACT_PATTERNS = {
    "url": r"\b(?:https?://|www\.)?\w[\w.-]*\.\w{2,10}(?:/\S*)?\b",
    "email": r"[\w\.\+\-]+@[\w\.-]+\.\w+",
    "money": r"\b((\$|€|£|USD|EUR|GBP)\s?\d+(?:[.,]?\d+)?(?:\s?(million|billion|bn|k|m))?|\d+(?:[.,]?\d+)?\s?(million|billion|bn|k|m))\b",
    "date": r"\b(?:\d{1,2}[\/\-.]){2}(?:\d{2,4})\b|\b(19|20)\d{2}\b"
}

JUST_MASK_PATTERNS = {
    "phone": r"\b(?:\+?\d{1,2}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?)?\d{3}[\s\-]?\d{4}\b",
    "percent": r"\b\d{1,3}(?:[.,]\d+)?\s?(%|percent)\b",
    "number": r"\b\d+(?:[.,]\d+)?\b"
}

SPECIAL_TOKENS = [
    "mask_url", "mask_email", "mask_money", "mask_date",
    "mask_phone", "mask_percent", "mask_number"
]

def mask_and_extract_all(text, patterns=MASK_AND_EXTRACT_PATTERNS):
    '''
    Mask sensitive information in the text and extract it into a dictionary.
    '''
    results = {}
    masked_text = text
    for key, pattern in patterns.items():
        matches = re.findall(pattern, masked_text, flags=re.IGNORECASE)
        matches = ["".join(m) if isinstance(m, tuple) else m for m in matches]
        results[f"masked_{key}_list"] = matches
        results[f"has_masked_{key}"] = len(matches) > 0
        masked_text = re.sub(pattern, f" mask_{key} ", masked_text, flags=re.IGNORECASE)

    results["masked_description"] = masked_text
    return results

def mask_other(text, patterns=JUST_MASK_PATTERNS):
    '''
    Just mask sensitive information in the text without extracting it.
    '''
    masked_text = text
    for key, pattern in patterns.items():
        masked_text = re.sub(pattern, f" mask_{key} ", masked_text, flags=re.IGNORECASE)
    return masked_text


def ascii_ratio(text):
    ascii_count = sum(1 for c in text if ord(c) < 128)
    text_length = len(text)
    if text_length == 0:
        return 0
    else:
        return ascii_count / text_length

def symbol_ratio(text):
    symbol_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
    text_length = len(text)
    if text_length == 0:
        return 0
    else:
        return symbol_count / text_length

def digit_ratio(text):
    digit_count = sum(1 for c in text if c.isdigit())
    text_length = len(text)
    if text_length == 0:
        return 0
    else:
        return digit_count / text_length

def detect_lang(text):
    try:
        if len(text.strip()) < 20:
            return "short"
        langs = detect_langs(text)
        top = langs[0]
        if top.lang == "en" and top.prob > 0.80:
            return "en"
        elif top.prob > 0.80:
            return top.lang
        else:
            return "uncertain"
    except:
        return "error"

def lemmatize_text(text):
    '''
    Lemmatize the text using spaCy. Have chosen to keep stop words due to open-ended nature of the project.
    '''
    doc = nlp(text)
    lemmas = [
        token.text if token.text in SPECIAL_TOKENS else token.lemma_
        for token in doc if not token.is_punct and not token.is_space
    ]
    return " ".join(lemmas)

def advanced_doc_stats(df):
    '''
    Adds some more advanced document stats based on the cleaned and lemmatized descriptions
    '''
    # Create spacy doc objects for getting advanced stats
    df["cleaned_doc"] = df["cleaned_description"].apply(lambda x: nlp(x))

    # Calculate advanced stats
    df["stopword_ratio"] = df["cleaned_doc"].apply(stopword_ratio)
    df["unique_word_ratio"] = df["cleaned_doc"].apply(unique_word_ratio)
    df["noun_verb_ratio"] = df["cleaned_doc"].apply(noun_verb_ratio)
    df["sentence_count"] = df["cleaned_doc"].apply(sentence_count)
    df["avg_word_length"] = df["cleaned_doc"].apply(avg_word_length)

    return df

def stopword_ratio(doc):
    if len(doc) == 0:
        return 0.0
    return sum(1 for tok in doc if tok.is_stop) / len(doc)

def unique_word_ratio(doc):
    tokens = [tok.lemma_ for tok in doc if not tok.is_punct and not tok.is_space]
    if len(tokens) == 0:
        return 0.0
    return len(set(tokens)) / len(tokens)

def noun_verb_ratio(doc):
    num_nouns = sum(1 for tok in doc if tok.pos_ == "NOUN")
    num_verbs = sum(1 for tok in doc if tok.pos_ == "VERB")
    return num_nouns / (num_verbs + 1)  # Avoid div-by-zero

def sentence_count(doc):
    return len(list(doc.sents))

def avg_word_length(doc):
    words = [tok.text for tok in doc if tok.is_alpha]
    if len(words) == 0:
        return 0.0
    return sum(len(w) for w in words) / len(words)
