import torch
from torchvision import models, transforms
import requests
import base64
from io import BytesIO
from PIL import Image

# Define image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device)
model.eval()
embedding_model = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)


@torch.inference_mode()
def get_embedding(image_path):
    """Modified to work with PIL images"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    embedding = embedding_model(image_tensor)
    return embedding.cpu().view(-1).numpy()


def process_image(input_data: str, is_base64: bool) -> Image.Image:
    """Process image from URL or base64 string"""
    if is_base64:
        # Base64 decoding
        if "base64," in input_data:  # Handle data URI
            input_data = input_data.split("base64,")[1]
        img_data = base64.b64decode(input_data)
        return Image.open(BytesIO(img_data)).convert("RGB")
    else:
        # URL handling
        response = requests.get(input_data, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")


def get_embeddings(image_paths, batch_size=32):
    """Extracts embeddings for a list of image paths in batches."""
    embeddings_list = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = preprocess(img)
            images.append(img_tensor)
        # Stack images into a single tensor of shape (batch_size, C, H, W)
        batch_tensor = torch.stack(images)
        with torch.no_grad():
            batch_embeddings = embedding_model(batch_tensor)
        # Flatten the embeddings for each image
        batch_embeddings = batch_embeddings.view(batch_embeddings.size(0), -1)
        embeddings_list.append(batch_embeddings)
    # Concatenate all batches and return as a NumPy array
    embeddings_all = torch.cat(embeddings_list, dim=0)
    return embeddings_all.numpy()
