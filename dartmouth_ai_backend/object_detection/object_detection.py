from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import torch

import base64
import io
from pathlib import Path


def px_to_pt(px):
    return px * 0.75


class ObjectDetector:
    def __init__(self, model_cache=None):
        self.__model = YolosForObjectDetection.from_pretrained(
            "hustvl/yolos-tiny", cache_dir=model_cache
        )
        self.__image_processor = YolosImageProcessor.from_pretrained(
            "hustvl/yolos-tiny", cache_dir=model_cache
        )
        self.__relative_font_size = 0.1

    def detect(self, image):
        image = Image.open(image)

        inputs = self.__image_processor(images=image, return_tensors="pt")
        outputs = self.__model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.__image_processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        objects = dict()

        draw = ImageDraw.Draw(image)
        font_path = str(
            Path(__file__).parent.parent.resolve() / "resources/AppleGaramond.ttf"
        )

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            label_name = self.__model.config.id2label[label.item()]

            font_size_px = (box[3] - box[1]) * self.__relative_font_size
            font = ImageFont.truetype(font_path, size=px_to_pt(font_size_px))
            draw.rectangle(box, outline="green", width=2)
            draw.text(box[:2], label_name, fill="green", font=font)
            byte_buffer = io.BytesIO()
            image.save(byte_buffer, format=image.format)
            objects[label_name] = {"score": score.item(), "bbox": box}
            objects["annotated_image"] = base64.b64encode(
                byte_buffer.getvalue()
            ).decode()

        return objects

    @staticmethod
    def how_to_cite(format="bibtex") -> str:
        if format != "bibtex":
            return NotImplemented
        return """@article{DBLP:journals/corr/abs-2106-00666,
    author    = {Yuxin Fang and Bencheng Liao and Xinggang Wang and Jiemin Fang and Jiyang Qi and Rui Wu and Jianwei Niu and Wenyu Liu},
    title     = {You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection},
    journal   = {CoRR},
    volume    = {abs/2106.00666},
    year      = {2021},
    url       = {https://arxiv.org/abs/2106.00666},
    eprinttype = {arXiv},
    eprint    = {2106.00666},
    timestamp = {Fri, 29 Apr 2022 19:49:16 +0200},
    biburl    = {https://dblp.org/rec/journals/corr/abs-2106-00666.bib},
    bibsource = {dblp computer science bibliography, https://dblp.org}
}"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="object_detection",
        description="Object detection in images",
        epilog="Detects various objects in a given image.",
    )

    parser.add_argument(
        "image", metavar="img", type=str, help="The name of the image file to process."
    )

    args = parser.parse_args()

    results = ObjectDetector().detect(args.image)

    print(results)
