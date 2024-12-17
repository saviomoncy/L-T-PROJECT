import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, flash
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Flask App Initialization
app = Flask(__name__)
app.secret_key = 'secret_key'  # Required for flash messages
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# PCA Function
def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return principal_components

# LDA Function
def apply_lda(X, y, n_components=2):
    lda = LDA(n_components=n_components)
    lda_components = lda.fit_transform(X, y)
    return lda_components

# Route for Upload and Visualization
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file exists
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check for valid file
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read and preprocess the dataset
            df = pd.read_csv(filepath)
            if df.shape[1] < 3:  # PCA and LDA require enough features
                flash("Dataset must have at least 2 features and 1 target column")
                return redirect(request.url)
            
            # Split features and target
            X = df.iloc[:, :-1]  # Features
            y = df.iloc[:, -1]   # Target column
            
            # Encode labels if necessary
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PCA and LDA
            pca_result = apply_pca(X_scaled, n_components=2)
            lda_result = None
            if len(np.unique(y_encoded)) > 1:  # LDA requires multiple classes
                lda_result = apply_lda(X_scaled, y_encoded, n_components=2)
            
            # Generate plots
            plt.figure(figsize=(12, 4))

            # Original Data (First 2 features)
            plt.subplot(1, 3, 1)
            plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_encoded, cmap='viridis', s=10)
            plt.title("Original Data (First 2 Features)")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")

            # PCA Reduced
            plt.subplot(1, 3, 2)
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_encoded, cmap='viridis', s=10)
            plt.title("PCA Reduced Data")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")

            # LDA Reduced (if available)
            if lda_result is not None:
                plt.subplot(1, 3, 3)
                plt.scatter(lda_result[:, 0], lda_result[:, 1], c=y_encoded, cmap='viridis', s=10)
                plt.title("LDA Reduced Data")
                plt.xlabel("LDA Component 1")
                plt.ylabel("LDA Component 2")
            else:
                plt.subplot(1, 3, 3)
                plt.text(0.5, 0.5, "LDA requires multiple classes", horizontalalignment='center')
                plt.axis('off')
                plt.title("LDA Not Applicable")

            plt.tight_layout()
            plot_path = os.path.join(UPLOAD_FOLDER, 'comparison_plot.png')
            plt.savefig(plot_path)
            plt.close()

            return render_template('result.html', image=plot_path)
        
        flash('Allowed file types: .csv')
        return redirect(request.url)

    return render_template('upload.html')

# Route to serve image files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
