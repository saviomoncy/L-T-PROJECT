document.getElementById('uploadForm').onsubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    const fileInput = document.getElementById('fileInput').files[0];
    const datasetUrl = document.getElementById('datasetUrl').value;

    // Append file or dataset URL to form data
    if (fileInput) {
        formData.append('file', fileInput);
    }
    if (datasetUrl) {
        formData.append('dataset_url', datasetUrl);
    }

    try {
        // Send data to the backend
        const response = await fetch('http://127.0.0.1:5000/upload', {
            method: 'POST',
            body: formData
        });


        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        // Display Original Images
        const originalDiv = document.getElementById('originalImages');
        originalDiv.innerHTML = "";
        data.original_images.forEach(img => {
            const imgTag = document.createElement('img');
            imgTag.src = 'data:image/png;base64,' + img;
            imgTag.style.margin = '10px';
            imgTag.width = 100;
            originalDiv.appendChild(imgTag);
        });

        // Display Reduced Scatter Plot
        document.getElementById('scatterPlot').src = 'data:image/png;base64,' + data.scatter_image;

        // Display Feature Information
        document.getElementById('featureInfo').innerText =
            `Original Features: ${data.original_features}, Reduced Features: ${data.reduced_features}`;
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing your request.');
    }
};
