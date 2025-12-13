import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Keywords for leave reason categorization
CATEGORY_KEYWORDS = {
    'Medical': [
        'sick', 'illness', 'hospital', 'doctor', 'medical', 'health', 
        'fever', 'pain', 'surgery', 'treatment', 'appointment', 'checkup',
        'injury', 'disease', 'medication', 'clinic', 'emergency room', 
        'diagnosis', 'prescription', 'recovery', 'flu', 'cold', 'headache',
        'stomach', 'infection', 'covid', 'corona', 'virus', 'pneumonia'
    ],
    'Emergency': [
        'emergency', 'urgent', 'critical', 'accident', 'death', 'funeral',
        'crisis', 'immediate', 'sudden', 'unexpected', 'family emergency',
        'hospitalization', 'critical condition', 'serious', 'life-threatening',
        'casualty', 'disaster', 'tragedy', 'bereavement'
    ],
    'Personal': [
        'personal', 'family', 'wedding', 'marriage', 'celebration', 'function',
        'ceremony', 'event', 'child', 'parent', 'relative', 'home', 'moving',
        'relocation', 'travel', 'vacation', 'trip', 'visit', 'attend',
        'birthday', 'anniversary', 'festival', 'holiday', 'religious',
        'cultural', 'social', 'commitment', 'obligation', 'responsibility'
    ]
}

def preprocess_text(text):
    """
    Clean and preprocess text for NLP analysis.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def categorize_reason(reason_text):
    """
    Categorize leave reason based on keywords.
    
    Returns:
        str: 'Medical', 'Emergency', 'Personal', or 'Other'
    """
    processed_text = preprocess_text(reason_text)
    
    # Count keyword matches for each category
    category_scores = {}
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            if keyword in processed_text:
                score += 1
        category_scores[category] = score
    
    # Find category with highest score
    max_score = max(category_scores.values())
    
    if max_score == 0:
        return 'Other'
    
    # Return category with highest score
    for category, score in category_scores.items():
        if score == max_score:
            return category
    
    return 'Other'

def extract_nlp_features(reason_text, tfidf_vectorizer=None):
    """
    Extract NLP features from leave reason text.
    
    Args:
        reason_text: str, the leave reason
        tfidf_vectorizer: fitted TfidfVectorizer (optional)
    
    Returns:
        dict: {
            'category': str,
            'vector': numpy array (if vectorizer provided)
        }
    """
    # Categorize the reason
    category = categorize_reason(reason_text)
    
    result = {
        'category': category
    }
    
    # Generate TF-IDF vector if vectorizer is provided
    if tfidf_vectorizer is not None:
        processed_text = preprocess_text(reason_text)
        tfidf_vector = tfidf_vectorizer.transform([processed_text])
        result['vector'] = tfidf_vector.toarray()[0]
    else:
        result['vector'] = np.zeros(50)  # Default vector size
    
    return result

def create_tfidf_vectorizer(corpus, max_features=50):
    """
    Create and fit a TF-IDF vectorizer on the given corpus.
    
    Args:
        corpus: list of text documents
        max_features: maximum number of features
    
    Returns:
        fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        stop_words='english'
    )
    
    # Preprocess corpus
    processed_corpus = [preprocess_text(text) for text in corpus]
    
    # Fit vectorizer
    vectorizer.fit(processed_corpus)
    
    return vectorizer

def get_reason_embedding(reason_text, vectorizer):
    """
    Get TF-IDF embedding for a leave reason.
    
    Args:
        reason_text: str
        vectorizer: fitted TfidfVectorizer
    
    Returns:
        numpy array
    """
    processed_text = preprocess_text(reason_text)
    vector = vectorizer.transform([processed_text])
    return vector.toarray()[0]