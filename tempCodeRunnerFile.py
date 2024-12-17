from flask import Flask, request, jsonify, render_template
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests
import io

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # HTML page for uploads and links

@app.route('/upload', methods=['POST'])
def upload():
    dataset_url = request.form.get('dataset_url')  # URL of the dataset
    file = request.files.get('file')  # File uploaded by user

    # --- 1. Load Data ---
    try:
        if dataset_url:
            # Fetch data from the provided URL
            response = requests.get(dataset_url)
            if response.status_code == 200:
                data = np.loadtxt(io.StringIO(response.text), delimiter=',')
            else:
                return jsonify({'error': 'Failed to fetch dataset from the provided URL.'})
        elif file:
            # Load data from uploaded file
            data = np.loadtxt(file, delimiter=',')
        else:
            return jsonify({'error': 'No dataset provided. Upload a file or provide a dataset URL.'})
    except Exception as e:
        return jsonify({'error': f'Failed to load dataset: {str(e)}'})

    # --- 2. Visualize Original Images (First 5 Images) ---
    num_images = min(5, data.shape[0])
    original_images = []
    for i in range(num_images):
        try:
            img = data[i].reshape(28, 28)  # Assumes MNIST-like dataset
            img_buffer = BytesIO()
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            original_images.append(base64.b64encode(img_buffer.getvalue()).decode())
        except:
            break  # Skip image visualization if reshaping fails

    # --- 3. Apply PCA for Dimensionality Reduction ---
    original_features = data.shape[1]
    reduced_features = 2  # Number of components
    pca = PCA(n_components=reduced_features)
    reduced_data = pca.fit_transform(data)

    # --- 4. Generate Scatter Plot of Reduced Data ---
    scatter_buffer = BytesIO()
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=5, c='blue')
    plt.title("Reduced Dimensionality (PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(scatter_buffer, format='png')
    scatter_buffer.seek(0)
    scatter_image = base64.b64encode(scatter_buffer.getvalue()).decode()

    return jsonify({
        'original_images': original_images,
        'scatter_image': scatter_image,
        'original_features': original_features,
        'reduced_features': reduced_features
    })

if __name__ == "__main__":
    app.run(debug=True)
