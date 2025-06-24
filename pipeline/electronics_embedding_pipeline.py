#!/usr/bin/env python3

"""
Kubeflow pipeline for electronics parts country of origin prediction
"""

import kfp
from kfp import dsl
from kfp.dsl import component, pipeline, Input, Output, Dataset, Model, Metrics


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["pandas", "numpy", "scikit-learn", "python-dotenv", "boto3"],
)
def load_and_prepare_data(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    data_filename: str,
    processed_data: Output[Dataset],
    label_encoder: Output[Model],
    labels: Output[Dataset],
    descriptions: Output[Dataset],
):
    """Load and prepare data for the embedding pipeline"""
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    import pickle
    import boto3
    import os

    # Setup MinIO client
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False,
    )

    # Download data from MinIO
    local_data_path = "/tmp/input_data.csv"
    s3_client.download_file(bucket_name, f"data/{data_filename}", local_data_path)

    # Load the synthetic electronics data
    df = pd.read_csv(local_data_path)

    print(f"Loaded {len(df)} records from MinIO bucket {bucket_name}")
    print(f"Columns: {list(df.columns)}")

    # Prepare labels
    label_encoder_obj = LabelEncoder()
    y = label_encoder_obj.fit_transform(df["Country_Of_Origin"])
    class_names = label_encoder_obj.classes_

    print(f"Number of countries: {len(class_names)}")
    print(f"Countries: {list(class_names)}")

    # Save processed data and encoder
    df.to_csv(processed_data.path, index=False)

    with open(label_encoder.path, "wb") as f:
        pickle.dump(label_encoder_obj, f)

    with open(labels.path, "wb") as f:
        pickle.dump(y, f)

    # Save descriptions for embedding generation
    descriptions_list = df["Part_Description"].tolist()
    with open(descriptions.path, "wb") as f:
        pickle.dump(descriptions_list, f)

    print("Data preparation complete")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["numpy", "python-dotenv", "requests"],
)
def generate_embeddings(
    descriptions: Input[Dataset],
    endpoint: str,
    embedding_model: str,
    api_key: str,
    embeddings: Output[Dataset],
):
    """Generate embeddings using vLLM BGE-Large model"""
    import pickle
    import numpy as np
    import requests
    import json

    # Load descriptions
    with open(descriptions.path, "rb") as f:
        descriptions_list = pickle.load(f)

    print(f"Generating embeddings for {len(descriptions_list)} descriptions...")

    # Create embeddings via API call
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    embeddings_list = []

    for desc in descriptions_list:
        payload = {"model": embedding_model, "input": desc}

        response = requests.post(
            f"{endpoint}/v1/embeddings", headers=headers, json=payload, verify=False
        )

        if response.status_code == 200:
            result = response.json()
            embedding = result["data"][0]["embedding"]
            embeddings_list.append(embedding)
        else:
            print(f"Error generating embedding: {response.status_code}")
            raise Exception(f"Failed to generate embedding for: {desc[:50]}...")

    if embeddings_list:
        X = np.array(embeddings_list)
        print(f"Generated embeddings with shape: {X.shape}")

        # Save embeddings
        with open(embeddings.path, "wb") as f:
            pickle.dump(X, f)

        print("Embeddings saved")
    else:
        raise Exception("Failed to generate embeddings")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["numpy", "scikit-learn", "joblib"],
)
def train_model(
    embeddings: Input[Dataset],
    labels: Input[Dataset],
    label_encoder: Input[Model],
    trained_model: Output[Model],
    test_results: Output[Dataset],
    metrics: Output[Metrics],
):
    """Train KNN classifier on embeddings"""
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    import joblib
    import json

    # Load embeddings and labels
    with open(embeddings.path, "rb") as f:
        X = pickle.load(f)

    with open(labels.path, "rb") as f:
        y = pickle.load(f)

    with open(label_encoder.path, "rb") as f:
        label_encoder_obj = pickle.load(f)

    class_names = label_encoder_obj.classes_

    print(f"Dataset info:")
    print(f"   - Total samples: {len(X)}")
    print(f"   - Embedding dimension: {X.shape[1]}")
    print(f"   - Number of countries: {len(class_names)}")

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Split data: {len(X_train)} train, {len(X_test)} test")

    # Train KNN classifier
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    print(f"Training KNN classifier with k={k}...")
    knn.fit(X_train, y_train)
    print("KNN training complete")

    # Evaluate model
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Performance:")
    print(f"   - Accuracy: {accuracy:.1%}")

    # Save model
    joblib.dump(knn, trained_model.path)

    # Save test data for evaluation
    test_data = {
        "X_test": X_test.tolist(),
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
        "accuracy": accuracy,
    }

    with open(test_results.path, "wb") as f:
        pickle.dump(test_data, f)

    # Log metrics
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("test_samples", len(X_test))
    metrics.log_metric("num_classes", len(class_names))

    print("Model training complete")


@component(
    base_image="registry.redhat.io/ubi8/python-39:latest",
    packages_to_install=["matplotlib", "seaborn", "scikit-learn", "numpy", "boto3"],
)
def evaluate_model(
    test_results: Input[Dataset],
    label_encoder: Input[Model],
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    bucket_name: str,
    confusion_matrix_plot: Output[Dataset],
):
    """Evaluate model and generate visualizations"""
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, classification_report
    import boto3

    # Load test results and label encoder
    with open(test_results.path, "rb") as f:
        test_data = pickle.load(f)

    with open(label_encoder.path, "rb") as f:
        label_encoder_obj = pickle.load(f)

    y_test = np.array(test_data["y_test"])
    y_pred = np.array(test_data["y_pred"])
    accuracy = test_data["accuracy"]

    class_names = label_encoder_obj.classes_

    print(f"Model Performance:")
    print(f"   - Accuracy: {accuracy:.1%}")

    # Show detailed results
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Setup MinIO client
    s3_client = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        verify=False,
    )

    # Create confusion matrix
    if len(class_names) > 1:
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title("Confusion Matrix - Country of Origin Prediction")
        plt.xlabel("Predicted Country")
        plt.ylabel("Actual Country")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save locally first
        local_plot_path = "/tmp/confusion_matrix.png"
        plt.savefig(local_plot_path, dpi=150, bbox_inches="tight")
        plt.savefig(confusion_matrix_plot.path, dpi=150, bbox_inches="tight")
        plt.close()

        # Upload to MinIO
        s3_client.upload_file(
            local_plot_path, bucket_name, "outputs/confusion_matrix.png"
        )

        print("Confusion matrix saved to MinIO")

    # Save evaluation report to MinIO
    eval_report = classification_report(y_test, y_pred, target_names=class_names)
    local_report_path = "/tmp/evaluation_report.txt"
    with open(local_report_path, "w") as f:
        f.write(f"Model Performance:\n")
        f.write(f"Accuracy: {accuracy:.1%}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(eval_report)

    s3_client.upload_file(
        local_report_path, bucket_name, "outputs/evaluation_report.txt"
    )

    print("Evaluation complete - results saved to MinIO outputs folder")


@pipeline(
    name="electronics-embedding-pipeline",
    description="Electronics parts country of origin prediction using BGE-Large embeddings",
)
def electronics_embedding_pipeline(
    minio_endpoint: str = "",
    minio_access_key: str = "",
    minio_secret_key: str = "",
    bucket_name: str = "pipeline",
    data_filename: str = "synthetic_electronics_parts_1k.csv",
    endpoint: str = "",
    embedding_model: str = "",
    api_key: str = "",
):
    """Main pipeline for electronics parts classification"""

    # Step 1: Load and prepare data
    load_task = load_and_prepare_data(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
        data_filename=data_filename,
    )

    # Step 2: Generate embeddings
    embed_task = generate_embeddings(
        descriptions=load_task.outputs["descriptions"],
        endpoint=endpoint,
        embedding_model=embedding_model,
        api_key=api_key,
    )

    # Step 3: Train model
    train_task = train_model(
        embeddings=embed_task.outputs["embeddings"],
        labels=load_task.outputs["labels"],
        label_encoder=load_task.outputs["label_encoder"],
    )

    # Step 4: Evaluate model
    eval_task = evaluate_model(
        test_results=train_task.outputs["test_results"],
        label_encoder=load_task.outputs["label_encoder"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        bucket_name=bucket_name,
    )


if __name__ == "__main__":
    # This allows the pipeline to be compiled
    pass
