from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms
from utils import load_finetuned_model, generate_gradcam_visualization, image_to_base64
import numpy as np

app = Flask(__name__)

# UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

model_path =  "./models/EfficientNet_augmented_transfer_10Epochs.pth"
model = load_finetuned_model(model_path)
model.eval()

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

image_tensor = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # filename = secure_filename(file.filename)
        # file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # file.save(file_path)

        # Load and preprocess the image
        image = Image.open(file).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]
            predicted_score = "{:.2f}".format(probabilities[0, predicted.item()].item())

        # gradcam
        visualization = generate_gradcam_visualization(model, image_tensor)
        visualization_base64 = image_to_base64(visualization)
        actual_image = image_to_base64(np.array(image))

        return jsonify({'class': predicted_class, 
                        'score': predicted_score, 'visualization': visualization_base64, 'actualImage': actual_image})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)