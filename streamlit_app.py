import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import timm

# Load your trained model checkpoint
model_path = "D:\\python_ml\\mobilenetv3_large_100_checkpoint_fold0.pt"

# Define the MobileNetV3 model from timm with the desired precision
model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=12).to(dtype=torch.float32)

# Load the checkpoint
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

# Extract the state dictionary from the checkpoint
state_dict = checkpoint.state_dict() if isinstance(checkpoint, torch.nn.Module) else checkpoint

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Preprocessing steps (same as before)
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# List of class names
class_names = [
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]

# Create a function to make predictions
def predict(image):
    img = Image.open(image).convert('RGB')
    img = preprocess(img).unsqueeze(0).float()  # Convert to float32 explicitly
    with torch.no_grad():
        model.eval()
        prediction = model(img)
        predicted_class = torch.argmax(prediction).item()
        confidence = torch.softmax(prediction, dim=1)[0][predicted_class].item() * 100
        return predicted_class, confidence

# Streamlit app
st.title('Garbage Classification App')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions on the uploaded image
    if st.button('Classify'):
        label, confidence = predict(uploaded_file)
        class_label = class_names[label]
        st.write(f"Predicted Class: {class_label}")
        st.write(f"Accuracy: {confidence:.2f}%")