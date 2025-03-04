import torch
from torchvision import models, transforms
from PIL import Image

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load a pre-trained model (e.g., ResNet50)
model = models.resnet50(pretrained=True)
model.eval()

# Remove the final classification layer to use the network as a feature extractor
embedding_model = torch.nn.Sequential(*(list(model.children())[:-1]))

def get_embedding(image_path):
    """Extracts and returns the embedding for a single image."""
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        embedding = embedding_model(image_tensor)
    return embedding.view(-1).numpy()  # flatten the output

def get_embeddings(image_paths, batch_size=32):
    """Extracts embeddings for a list of image paths in batches."""
    embeddings_list = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i: i + batch_size]
        images = []
        for path in batch_paths:
            img = Image.open(path).convert('RGB')
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
