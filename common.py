from PIL import Image, ImageDraw, ImageOps
import random
import requests


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


def rand_color():
    """Generates a random color"""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return r, g, b


def model2pipeline_results(results, model):
    """Converts the results from a model to the format expected by the pipeline"""
    res = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        res.append(
            {"score": score, "label": model.config.id2label[label.item()],
             "box": {'xmin': box[0], 'ymin': box[1], 'xmax': box[2], 'ymax': box[3]}}
        )
    return res


def draw_boxes(image, results, threshold=0.9, font_size=50):
    """Draws boxes and detected category around the detected object

    :param image: the base image where the boxes will be drawn
    :param results: the results from the object detection pipeline, an array of
                    dictionaries, each containing the `label`, `score`, and `box`
    :param threshold: boxes with confidence scores below this threshold will not be drawn
    :param font_size: the font size for the label
    """
    draw = ImageDraw.Draw(image)
    for item in results:
        if item['score'] > threshold:
            box = [i for i in item['box'].values()]
            draw.rectangle(
                box, outline=rand_color(), width=6
            )
            draw.text(
                (box[0], box[1] - 10), f"{item['label']} ({item['score']:.2f})",
                font_size=font_size, fill='white'
            )


def apply_mask(image, mask):
    """Applies a mask to an image

    An `image_segmentation` pipeline returns a mask for the detected object, in the
    `mask` key of the `results` dictionary. This function applies the mask to the image
    and returns the modified image.
    """
    base_image = image.copy()
    mask_image = ImageOps.invert(mask)
    base_image.paste(mask_image, mask=mask_image)
    return base_image
