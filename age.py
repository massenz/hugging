
from PIL import Image
import requests
from transformers import pipeline, ViTFeatureExtractor, ViTForImageClassification


def load_image(url):
    """Loads an image from a URL or local disk"""
    # if the image is from the Web
    if url.startswith('http'):
        image = Image.open(requests.get(url, stream=True).raw)
    elif url.startswith('file'):
        # the image is local
        image = Image.open(url[7:])
    else:
        raise ValueError("Unsupported image source URL")
    return image


# Load an image from disk
im = load_image('file://marco.jpeg')

####
# Init model, transforms
model = ViTForImageClassification.from_pretrained('nateraw/vit-age-classifier')
transforms = ViTFeatureExtractor.from_pretrained('nateraw/vit-age-classifier')

# Transform our image and pass it through the model
inputs = transforms(im, return_tensors='pt')
output = model(**inputs)

# Predicted Class probabilities
proba = output.logits.softmax(1)

# Predicted Classes
preds = proba.argmax(1)

# Print the results
print(f"Age predicted: {model.config.id2label[preds.item()]}, "
      f"with probability: {proba.max().item() * 100:.2f}%")

####
# Using transformers pipeline
classifier = pipeline("image-classification",
                      model='nateraw/vit-age-classifier')
results = classifier(im)
maxProb = 0
detectedAge = ""
for result in results:
    if result['score'] > maxProb:
        maxProb = result['score']
        detectedAge = result['label']

print(f"Age predicted: {detectedAge}, with probability: {maxProb * 100:.2f}%")
