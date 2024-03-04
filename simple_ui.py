import gradio as gr
from PIL import Image, ImageDraw, ImageOps
from transformers import SegformerForSemanticSegmentation, pipeline


model = pipeline("image-segmentation",
                 model="nvidia/segformer-b0-finetuned-ade-512-512")


def segmentation(image, label):
    image = Image.fromarray(image)
    results = model(image)
    print(f"Found: {[result['label'] for result in results]}")
    for result in results:
        if result['label'] == label:
            base_image = image.copy()
            mask_image = result['mask']
            mask_image = ImageOps.invert(mask_image)
            base_image.paste(mask_image, mask=mask_image)
            return base_image


def run():
    image_input = gr.Image(label = "Image to segmentize")
    label = gr.Textbox(label = "Label to look for", placeholder = "Label")
    image_output = gr.Image(label = "Image with the mask applied")
    gr.Interface(segmentation,
                 [image_input, label],
                 image_output).launch()


if __name__ == "__main__":
    run()
