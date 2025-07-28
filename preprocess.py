import cupy as cp
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

def load_and_vectorize(texts):
    vectorizer = TfidfVectorizer(stop_words='english')
    X_cpu = vectorizer.fit_transform(texts).toarray()
    X_cpu = normalize(X_cpu)
    return cp.asarray(X_cpu)
