import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 6)  # 6 classes in your dataset
    )
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))  # Load your trained model
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Model prediction
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_probabilities = probabilities.squeeze().cpu().numpy()
    return predicted_class, predicted_probabilities

# Streamlit UI
def main():
    st.title("Malaysia Ringgit Classification App")

    st.write("Upload an image to classify it into one of the RM categories (RM1, RM5, RM10, RM20, RM50, RM100).")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("")

        # Load the model
        model = load_model()

        # Preprocess the image
        image_tensor = preprocess_image(image)

        # Predict the class and probabilities
        predicted_class, predicted_probabilities = predict(model, image_tensor)

        # Define class labels
        class_names = ['RM1', 'RM5', 'RM10', 'RM20', 'RM50', 'RM100']

        # Display the predicted class with the highest probability
        st.write(f"**Predicted Class:** {class_names[predicted_class]}")

        # Display the probabilities for all classes
        st.write("### Prediction Probabilities:")
        for i, prob in enumerate(predicted_probabilities):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
=======
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 6)  # 6 classes in your dataset
    )
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))  # Load your trained model
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Model prediction
def predict(model, image):
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_probabilities = probabilities.squeeze().cpu().numpy()
    return predicted_class, predicted_probabilities

# Streamlit UI
def main():
    st.title("Malaysia Ringgit Classification App")

    st.write("Upload an image to classify it into one of the RM categories (RM1, RM5, RM10, RM20, RM50, RM100).")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Open and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("")

        # Load the model
        model = load_model()

        # Preprocess the image
        image_tensor = preprocess_image(image)

        # Predict the class and probabilities
        predicted_class, predicted_probabilities = predict(model, image_tensor)

        # Define class labels
        class_names = ['RM1', 'RM5', 'RM10', 'RM20', 'RM50', 'RM100']

        # Display the predicted class with the highest probability
        st.write(f"**Predicted Class:** {class_names[predicted_class]}")

        # Display the probabilities for all classes
        st.write("### Prediction Probabilities:")
        for i, prob in enumerate(predicted_probabilities):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")

if __name__ == "__main__":
    main()
