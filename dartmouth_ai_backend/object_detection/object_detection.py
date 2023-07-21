from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw
import torch


class ObjectDetector:
    def __init__(self):
        self.__model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        self.__image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")

    def detect(self, image):
        image = Image.open(image)

        inputs = self.__image_processor(images=image, return_tensors="pt")
        outputs = self.__model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.__image_processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes)[0]

        objects = dict()

        for score, label, box in zip(results["scores"],
                                     results["labels"],
                                     results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            objects[self.__model.config.id2label[label.item()]] = {
                'score': score.item(),
                'bbox': box
            }
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
