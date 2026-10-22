import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tempfile
import os

from sentence_transformers import SentenceTransformer


print("Loading SentenceTransformer...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading sentiment classifier...")

classifier = joblib.load(
    "sentiment_classifier.pkl"
)

print("Models loaded successfully.")
