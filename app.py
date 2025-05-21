import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
import os

# Set page configuration
st.set_page_config(
    page_title="Image Classification",
    page_icon="🖼️",
    layout="wide"
)

# Function to load and preprocess image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Function to get image as base64 for CSS styling
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Add custom CSS with improved contrast
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?q=80&w=2070&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    background-color: rgba(255, 255, 255, 0.9);
    background-blend-mode: lighten;
}

.main-container {
    background-color: rgba(255, 255, 255, 0.95);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    margin-bottom: 20px;
}

/* Text elements */
h1, h2, h3, h4, h5, h6, p, li, .header, .prediction-result, 
.image-label, .confidence-label, .file-uploader-label {
    color: #333333 !important;
}

.header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}

.prediction-result {
    background: linear-gradient(to right, #56ab2f, #a8e063);
    color: white !important;
    padding: 15px;
    border-radius: 10px;
    margin-top: 15px;
    text-align: center;
    font-size: 18px;
}

.predicted-badge {
    position: absolute;
    top: 10px;
    right: 10px;
    background: #00c853;
    color: white;
    padding: 5px 10px;
    font-size: 12px;
    border-radius: 20px;
    font-weight: bold;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}

.image-label {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0,0,0,0.7);
    color: white !important;
    padding: 8px;
    text-align: center;
    font-weight: bold;
    font-size: 14px;
}

/* Image effects */
.zoom-in {
    transform: scale(1.2);
    transition: transform 0.5s ease;
    margin: 0 auto;
    height: 300px;
    width: 80%;
}

.zoom-out {
    opacity: 0;
    height: 0;
    overflow: hidden;
    transition: all 0.5s ease;
}

.image-container {
    position: relative;
    overflow: hidden;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    height: 200px;
    border: 2px solid #ffffff;
    margin-bottom: 20px;
}

/* Confidence scores */
.confidence-container {
    background: rgba(255, 255, 255, 0.95);
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-top: 30px;
    border: 1px solid #e0e0e0;
    animation: fadeIn 0.8s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.confidence-bar {
    margin-bottom: 10px;
    padding: 5px;
    border-radius: 5px;
    transition: all 0.3s ease;
}

.confidence-bar:hover {
    background-color: rgba(0,0,0,0.05);
}

.confidence-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
    font-weight: 500;
}

/* Other components */
.upload-section {
    background: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    border: 1px solid #e0e0e0;
}

.file-uploader {
    border: 2px dashed #667eea;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
    background-color: rgba(255, 255, 255, 0.9);
}

.file-uploader:hover {
    border-color: #764ba2;
}

.sidebar .sidebar-content {
    background-color: rgba(255, 255, 255, 0.95) !important;
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.instructions {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# Main container for content
with st.container():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="header">
        <h1 style="color: white !important;">🌍 Landscape Recognition Model</h1>
        <p style="color: white !important;">Discover the beauty of nature and urban landscapes through AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Description
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <p style="font-size: 18px; color: #333333;">This intelligent system classifies images into six categories: 
        <span style="font-weight: bold; color: #667eea;">Buildings</span>, 
        <span style="font-weight: bold; color: #56ab2f;">Forest</span>, 
        <span style="font-weight: bold; color: #00b4d8;">Glacier</span>, 
        <span style="font-weight: bold; color: #ff6d00;">Mountain</span>, 
        <span style="font-weight: bold; color: #0077b6;">Sea</span>, and 
        <span style="font-weight: bold; color: #9d4edd;">Street</span>.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for layout
    col1, col2 = st.columns([1, 2])

    # Sample images for each class
    class_images = {
        'buildings': 'Images/building.jpg',
        'forest': 'Images/forest.jpeg',
        'glacier': 'Images/glacier.jpg',
        'mountain': 'Images/mountain.jpg',
        'sea': 'Images/sea.jpg',
        'street': 'Images/street.jpg'
    }

    # Encode each image to base64
    basepath = os.path.dirname(__file__)
    for label, path in class_images.items():
        with open(os.path.join(basepath, path), "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            class_images[label] = encoded_string

    # Upload section in the first column
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📤 Upload Your Image")
        
        # Custom file uploader
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        image_file = st.file_uploader("Drag and drop or click to browse", type=['jpg', 'png', 'jpeg'], label_visibility="collapsed")
        st.markdown('<p style="color: #333333;">Supported formats: JPG, PNG, JPEG</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Load model
        @st.cache_resource
        def load_classification_model():
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(base_dir, 'first_model.h5')
                return load_model(model_path)
            except Exception as e:
                st.error(f"Error in loading model: {e}")
                return None

        model = load_classification_model()

        # Display uploaded image and make prediction
        prediction = None
        result = None
        if image_file is not None:
            try:
                # Display the uploaded image
                img = load_image(image_file)
                st.image(img, caption="Your Uploaded Image", use_column_width=True)

                # Preprocess the image
                image = img.resize((150, 150))
                image_arr = np.array(image.convert('RGB')) / 255.0
                image_arr = image_arr.reshape(1, 150, 150, 3)

                # Predict
                if model:
                    with st.spinner("🔍 Analyzing your image..."):
                        result = model.predict(image_arr)
                        ind = np.argmax(result)
                        classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
                        prediction = classes[ind]
                        confidence = float(result[0][ind]) * 100
                        st.markdown(f'<div class="prediction-result">🎯 Prediction: {prediction.upper()} ({confidence:.2f}%)</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error processing image: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    # Display class images and confidence scores in the second column
    with col2:
        st.subheader("🌄 Class Gallery")
        
        with st.container():
            if prediction:
                # Display only the predicted image centered
                cols = st.columns([1, 3, 1])  # Wider center column
                classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
                
                with cols[1]:  # Middle column
                    # Predicted image with zoom effect
                    st.markdown(f"""
                    <div class="image-container zoom-in">
                        <img src="data:image/jpeg;base64,{class_images[prediction]}" alt="{prediction}" 
                             style="width: 100%; height: 100%; object-fit: cover;">
                        <div class="predicted-badge">⭐ Predicted</div>
                        <div class="image-label">{prediction.capitalize()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display confidence scores below
                    st.markdown('<div class="confidence-container">', unsafe_allow_html=True)
                    st.subheader("📊 Prediction Confidence Scores")
                    
                    for i, class_name in enumerate(classes):
                        conf = float(result[0][i]) * 100
                        color = {
                            'buildings': '#667eea',
                            'forest': '#56ab2f',
                            'glacier': '#00b4d8',
                            'mountain': '#ff6d00',
                            'sea': '#0077b6',
                            'street': '#9d4edd'
                        }.get(class_name, '#666666')
                        
                        # Highlight the predicted class
                        bar_style = "background-color: rgba(86, 171, 47, 0.1); padding: 8px; border-radius: 8px;" if class_name == prediction else ""
                        
                        st.markdown(f"""
                        <div class="confidence-bar" style="{bar_style}">
                            <div class="confidence-label">
                                <span style="color: {color}; font-weight: {'bold' if class_name == prediction else 'normal'}">{class_name.capitalize()}</span>
                                <span style="font-weight: {'bold' if class_name == prediction else 'normal'}">{conf:.2f}%</span>
                            </div>
                            <progress value="{conf}" max="100" style="width: 100%; height: 10px; accent-color: {color};"></progress>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # If no prediction yet, show all images normally
                gallery_cols = st.columns(3)
                classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
                for i, class_name in enumerate(classes):
                    with gallery_cols[i % 3]:
                        st.markdown(f"""
                        <div class="image-container">
                            <img src="data:image/jpeg;base64,{class_images[class_name]}" alt="{class_name}" 
                                 style="width: 100%; height: 100%; object-fit: cover;">
                            <div class="image-label">{class_name.capitalize()}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Bottom instructions
    st.markdown("""
    <div class="instructions">
        <h3>📝 How to use:</h3>
        <ol>
            <li>Upload an image using the file uploader</li>
            <li>Wait for the AI to analyze and classify the image</li>
            <li>See the prediction result and confidence scores</li>
            <li>Upload a different image to try again!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # Close main-container

# Sidebar info
with st.sidebar:
    st.title("ℹ️ About")
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.95); padding: 15px; border-radius: 10px;">
        <p style="color: #333333;">This ML model uses a Convolutional Neural Network (CNN) trained on the Image Classification dataset from Kaggle.</p>
        <p style="color: #333333;">The model can identify six different categories of natural and urban scenes with high accuracy.</p>
        
        Model Details:
    
        Architecture: Deep CNN
        Input Size: 150x150x3
        Classes: 6
        Accuracy: 82%
        Total Parameters: 8 million
        Training Size: 14k images
        Test Size: 3k images
        
        Try Once!
    </div>
    """, unsafe_allow_html=True)