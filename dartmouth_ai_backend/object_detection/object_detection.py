from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch

import base64
import io
from pathlib import Path


def px_to_pt(px):
    return px * 0.75


class ObjectDetector:
    def __init__(self):
        self.__model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        self.__image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.__relative_font_size = 0.1

    def detect(self, image):
        image = Image.open(image)

        inputs = self.__image_processor(images=image, return_tensors="pt")
        outputs = self.__model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.__image_processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes)[0]

        objects = dict()

        draw = ImageDraw.Draw(image)
        font_path = str(Path(__file__).parent.parent.resolve() / "resources/AppleGaramond.ttf")

        for score, label, box in zip(results["scores"],
                                     results["labels"],
                                     results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_name = self.__model.config.id2label[label.item()]

            font_size_px = (box[3] - box[1]) * self.__relative_font_size
            font = ImageFont.truetype(font_path, size=px_to_pt(font_size_px))
            draw.rectangle(box, outline='green', width=2)
            draw.text(box[:2], label_name, fill='green', font=font)
            byte_buffer = io.BytesIO()
            image.save(byte_buffer, format=image.format)
            objects[label_name] = {
                'score': score.item(),
                'bbox': box
            }
            objects['annotated_image'] = base64.b64encode(byte_buffer.getvalue()).decode()

        return objects


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog='object_detection',
        description='Object detection in images',
        epilog='Detects various objects in a given image.'
        )

    parser.add_argument('image', metavar='img', type=str,
                        help='The name of the image file to process.')

    args = parser.parse_args()

    results = ObjectDetector().detect(args.image)

    print(results)
