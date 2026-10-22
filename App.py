import gradio as gr
import pandas as pd
import numpy as np
import joblib
import tempfile
import os

from sentence_transformers import SentenceTransformer

print("Starting App.py...")

print("Loading SentenceTransformer...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2"
)

print("Loading sentiment classifier...")

classifier = joblib.load(
    "sentiment_classifier.pkl"
)

print("Models loaded successfully.")

def predict_sentiment(review):

    if review is None or review.strip() == "":
        return {
            "Negative": 0.0,
            "Neutral": 0.0,
            "Positive": 0.0
        }

    embedding = embedding_model.encode(
        [review],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    probabilities = classifier.predict_proba(
        embedding
    )[0]

    return {
        sentiment: float(probability)
        for sentiment, probability
        in zip(
            classifier.classes_,
            probabilities
        )
    }

def predict_batch(
    reviews,
    batch_size=32
):

    clean_reviews = [
        str(review)
        if pd.notna(review)
        else ""
        for review in reviews
    ]

    embeddings = embedding_model.encode(
        clean_reviews,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    predictions = classifier.predict(
        embeddings
    )

    probabilities = classifier.predict_proba(
        embeddings
    )

    return predictions, probabilities


def load_columns(file):

    if file is None:
        return gr.Dropdown(
            choices=[],
            value=None
        )

    df_upload = pd.read_csv(
        file.name
    )

    columns = df_upload.columns.tolist()

    possible_columns = [
        column
        for column in columns
        if (
            "review" in column.lower()
            or "comment" in column.lower()
            or "text" in column.lower()
        )
    ]

    default_column = (
        possible_columns[0]
        if possible_columns
        else columns[0]
    )

    return gr.Dropdown(
        choices=columns,
        value=default_column
    )

def analyse_dataset(
    file,
    review_column
):

    if file is None:
        return (
            None,
            None,
            "Please upload a CSV file."
        )

    if review_column is None:
        return (
            None,
            None,
            "Please select the review column."
        )

    try:

        df_upload = pd.read_csv(
            file.name
        )

        reviews = (
            df_upload[review_column]
            .fillna("")
            .astype(str)
            .tolist()
        )

        predictions, probabilities = predict_batch(
            reviews
        )

        df_upload["sentiment"] = predictions

        for i, sentiment in enumerate(
            classifier.classes_
        ):

            df_upload[
                f"{sentiment.lower()}_probability"
            ] = probabilities[:, i]

        positive_count = np.sum(
            predictions == "Positive"
        )

        neutral_count = np.sum(
            predictions == "Neutral"
        )

        negative_count = np.sum(
            predictions == "Negative"
        )

        preview = df_upload[
            [
                review_column,
                "sentiment"
            ]
        ].head(15)

        output_path = (
            tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".csv"
            ).name
        )

        df_upload.to_csv(
            output_path,
            index=False
        )

        status = f"""
### Analysis Complete ✨

**Total Reviews:** {len(df_upload):,}

🟢 **Positive:** {positive_count:,}

🟡 **Neutral:** {neutral_count:,}

🔴 **Negative:** {negative_count:,}
"""

        return (
            preview,
            output_path,
            status
        )

    except Exception as e:

        return (
            None,
            None,
            f"Error: {str(e)}"
        )

with gr.Blocks() as app:

    gr.Markdown(
        "# GlowSense Beauty Review Analyzer"
    )

    with gr.Tab(
        "Single Review"
    ):

        review_input = gr.Textbox(
            label="Customer Review"
        )

        button = gr.Button(
            "Analyse"
        )

        output = gr.Label(
            num_top_classes=3
        )

        button.click(
            predict_sentiment,
            review_input,
            output
        )


    with gr.Tab(
        "Dataset Analysis"
    ):

        upload = gr.File(
            file_types=[".csv"]
        )

        column = gr.Dropdown(
            choices=[]
        )

        upload.change(
            load_columns,
            upload,
            column
        )

        analyse = gr.Button(
            "Analyse Dataset"
        )

        file_output = gr.File()

        analyse.click(
            analyse_dataset,
            [
                upload,
                column
            ],
            file_output
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
