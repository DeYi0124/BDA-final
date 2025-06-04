import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def cluster_and_save(input_path, output_path, n_clusters):
    df = pd.read_csv(input_path)
    features = df.drop(columns=["id"])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    output = pd.DataFrame({
        "id": df["id"],
        "label": labels
    })
    output.to_csv(output_path, index=False)

if __name__ == "__main__":
    cluster_and_save("public_data.csv", "public_submission.csv", 15)
    cluster_and_save("private_data.csv", "private_submission.csv", 23)
