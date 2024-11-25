import torch
# mypy: ignore-errors
import torchvision.transforms as transforms
from PIL import Image
from celeb_classifier.model import CelebModel
import torch.nn.functional as F

def load_model(model_path):
    """Load the trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CelebModel().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, device

def preprocess_image(image_path):
    """Preprocess the input image."""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),  # This converts PIL Image to tensor
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path)
    image_tensor = transform(image)  # Now image_tensor is already a tensor
    return image_tensor.unsqueeze(0)  # type: ignore # Add batch dimension using tensor method

def predict(model, image, device):
    """Make prediction on the input image."""
    with torch.no_grad():
        image = image.to(device)
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        pred = output.argmax(dim=1, keepdim=True)
        return pred.item(), probabilities[0]

def main():
    # Path to your trained model
    model_path = "./model.pth"
    
    # Load model
    model, device = load_model(model_path)
    
    # Path to input image
    image_path = "./test/image.png"
    
    # Preprocess image and get prediction
    image = preprocess_image(image_path)
    prediction, probabilities = predict(model, image, device)
    
    print(f"Predicted digit: {prediction}")
    print("\nProbabilities for each digit:")
    for digit, prob in enumerate(probabilities):
        print(f"{digit}: {prob.item():.4f}")

if __name__ == "__main__":
    main()