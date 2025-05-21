🌄 Multiclass Landscape Image Classifier
An interactive web application built with Python and Streamlit that classifies landscape images into six categories:
🏢 Buildings | 🌲 Forest | 🌊 Sea | 🧊 Glaciers | 🏙️ Street | 🏔️ Mountains

Trained on a dataset of over 14,000 images, the model achieves an 83% accuracy on a 3,000-image test set, powered by a deep learning network with 7 million trainable parameters.

🔍 Project Overview
This project is a computer vision pipeline for multiclass image classification using a convolutional neural network (CNN). It features:

Interactive Streamlit dashboard for easy testing and visualization

Image upload support for user-supplied images

Clear and clean UI with real-time inference results

📂 Categories
The model classifies images into the following six categories:

Buildings

Forest

Sea

Glaciers

Street

Mountains

🧠 Model Details
Model Type: Custom Convolutional Neural Network (CNN)

Parameters: ~7 million trainable parameters

Training Data: 14,000+ labeled images

Test Set: 3,000 images

Accuracy: 83% on test data

🚀 Getting Started
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/sprakashx15/lImage-Classification-Using-CNN.git
cd landscape-image-classifier
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the App
bash
Copy
Edit
streamlit run app.py
🖼️ Using the App
Upload an image using the sidebar

The model will classify the image in real-time

View confidence scores and class prediction instantly

Visual insights and summaries are displayed interactively

📊 Dashboard Features
Image upload and preview

Predicted class with confidence level

Real-time classification

Optional class distribution visualization

🧪 Model Evaluation
Accuracy: 83%

Loss Function: CrossEntropyLoss

Optimizer: Adam

Evaluation Metrics: Accuracy, Confusion Matrix, Precision, Recall

📁 Folder Structure
bash
Copy
Edit
.
├── app.py                # Streamlit web app
├── model/                # Trained model files
├── notebook             # Preprocessing and helper functions
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
📌 To-Do
 Add Grad-CAM visualizations

 Improve accuracy with transfer learning

 Deploy to cloud (e.g., Streamlit Cloud, Hugging Face Spaces)

🧑‍💻 Author
Shubham

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
