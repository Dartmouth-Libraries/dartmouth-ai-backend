import os
import shutil
from pathlib import Path
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import seaborn as sns
from tqdm import tqdm


from dartmouth_ai_backend.language_detection import LanguageDetector
from dartmouth_ai_backend.named_entity_recognition import NamedEntityRecognizer
from dartmouth_ai_backend.object_detection import ObjectDetector
from dartmouth_ai_backend.sentiment_analysis import SentimentAnalyzer

# Instantiating these objects takes a significant amount of time, so only do this once
LANGUAGE_DETECTOR = LanguageDetector()
NAMED_ENTITY_RECOGNIZER = NamedEntityRecognizer()
OBJECT_DETECTOR = ObjectDetector()
SENTIMENT_ANALYZER = SentimentAnalyzer()


class Benchmark:
    def __init__(
        self,
        text_test_file,
        image_test_file,
        ner_file_root=None,
        ner_steps=10,
        ner_reps=6,
        ld_file_root=None,
        ld_steps=10,
        ld_reps=6,
        sa_file_root=None,
        sa_steps=10,
        sa_reps=6,
        od_file_root=None,
        od_steps=10,
        od_reps=6,
    ) -> None:
        self.text_test_file = Path(text_test_file)
        with open(self.text_test_file, "r") as f:
            full_text = f.read()
        self.image_test_file = Path(image_test_file)
        full_image = Image.open(self.image_test_file)

        # Named Entity Recognition
        if ner_file_root:
            self.ner_file_root = Path(ner_file_root)
        else:
            self.ner_file_root = Path("benchmark/benchmark_files/ner")
        self.ner_token_steps = np.linspace(10, len(full_text), ner_steps, dtype=int)
        self.ner_reps = ner_reps
        # Language Detection
        if ld_file_root:
            self.ld_file_root = Path(ld_file_root)
        else:
            self.ld_file_root = Path("benchmark/benchmark_files/ld")
        self.ld_token_steps = np.linspace(10, len(full_text), ld_steps, dtype=int)
        self.ld_reps = ld_reps
        # Sentiment Analysis
        if sa_file_root:
            self.sa_file_root = Path(sa_file_root)
        else:
            self.sa_file_root = Path("benchmark/benchmark_files/sa")
        self.sa_token_steps = np.linspace(10, len(full_text), sa_steps, dtype=int)
        self.sa_reps = sa_reps
        # Object Detection
        if od_file_root:
            self.od_file_root = Path(od_file_root)
        else:
            self.od_file_root = Path("benchmark/benchmark_files/od")
        self.od_size_steps = list(
            zip(
                np.linspace(10, full_image.size[0], od_steps, dtype=int),
                np.linspace(10, full_image.size[1], od_steps, dtype=int),
            )
        )
        self.od_reps = od_reps

    def _prepare_text_files(self, text, steps, root, base_name):
        # Clean folder first
        if root.exists():
            shutil.rmtree(root)
        os.makedirs(root)

        # Generate files of the specified length
        digits = len(str(steps.max()))
        for step in steps:
            with open(
                root / f"{base_name}_{step:0{digits}d}.txt",
                "w+",
            ) as f:
                f.write(text[:step])

    def _prepare_image_files(self, image: Image, steps, root, base_name):
        # Clean folder first
        if root.exists():
            shutil.rmtree(root)
        os.makedirs(root)

        width, height = steps[-1]
        digits = max(len(str(width)), len(str(height)))
        for step in steps:
            im = image.resize(step)
            im.save(
                root
                / f"{base_name}_{step[0]:0{digits}d}x{step[1]:0{digits}d}.{image.format.lower()}"
            )

    def prepare_files(self):
        # Text files
        with open(self.text_test_file, "r") as f:
            text = f.read()

        self._prepare_text_files(
            text=text,
            steps=self.ner_token_steps,
            root=self.ner_file_root,
            base_name=self.text_test_file.stem,
        )

        self._prepare_text_files(
            text=text,
            steps=self.ld_token_steps,
            root=self.ld_file_root,
            base_name=self.text_test_file.stem,
        )

        self._prepare_text_files(
            text=text,
            steps=self.sa_token_steps,
            root=self.sa_file_root,
            base_name=self.text_test_file.stem,
        )

        # Image files
        main_image = Image.open(self.image_test_file)
        self._prepare_image_files(
            image=main_image,
            steps=self.od_size_steps,
            root=self.od_file_root,
            base_name=self.image_test_file.stem,
        )

    def language_detection(self):
        files = sorted(self.ld_file_root.glob("*.txt"))
        return self._benchmark(files=files, task="ld", reps=self.ld_reps)

    def named_entity_recognition(self):
        files = sorted(self.ner_file_root.glob("*.txt"))
        return self._benchmark(files=files, task="ner", reps=self.ner_reps)

    def object_detection(self):
        files = sorted(self.od_file_root.glob("*.*"))
        return self._benchmark(files=files, task="od", reps=self.od_reps)

    def sentiment_analysis(self):
        files = sorted(self.sa_file_root.glob("*.txt"))
        return self._benchmark(files=files, task="sa", reps=self.sa_reps)

    def _benchmark(self, files, task, reps):
        n = []
        execution_time = []
        for file_name in tqdm(files):
            tqdm.write(f"Processing {file_name}")

            step = file_name.stem.split("_")[1]

            for rep in range(reps):
                match task:
                    case "ld":
                        with open(file_name, "r") as f:
                            text = f.read()
                        start = timer()
                        result = LANGUAGE_DETECTOR.detect(text)
                        stop = timer()
                    case "ner":
                        with open(file_name, "r") as f:
                            text = f.read()
                        start = timer()
                        result = NAMED_ENTITY_RECOGNIZER.recognize(text)
                        stop = timer()
                    case "od":
                        start = timer()
                        result = OBJECT_DETECTOR.detect(file_name)
                        stop = timer()
                    case "sa":
                        with open(file_name, "r") as f:
                            text = f.read()
                        start = timer()
                        result = SENTIMENT_ANALYZER.analyze(text)
                        stop = timer()

                execution_time.append(stop - start)
                if "x" in step:
                    width, height = step.split("x")
                    n.append((int(width), int(height)))
                else:
                    n.append(int(step))

        if isinstance(n[0], tuple):
            n = [width * height for (width, height) in n]
        results = pd.DataFrame(
            {
                "n": n,
                "execution_time_s": execution_time,
            }
        )
        results["task"] = task
        return results


if __name__ == "__main__":
    benchmark = Benchmark(
        text_test_file="./benchmark/benchmark_files/frankenstein.txt",
        image_test_file="./benchmark/benchmark_files/desk.jpg",
    )

    benchmark.prepare_files()

    # Object Detection
    od = benchmark.object_detection()
    od.to_csv("./benchmark/results/object_detection.csv", index=None)

    # Language Detection
    ld = benchmark.language_detection()
    ld.to_csv("./benchmark/results/language_detection.csv", index=None)

    # Named Entity Recognition
    ner = benchmark.named_entity_recognition()
    ner.to_csv("./benchmark/results/named_entity_recognition.csv", index=None)

    # Sentiment Analysis
    sa = benchmark.sentiment_analysis()
    sa.to_csv("./benchmark/results/sentiment_analysis.csv", index=None)

    # Combined results
    all_results = pd.concat([ld, ner, sa, od], ignore_index=True)
    all_results.to_csv("./benchmark/results/all_results.csv", index=None)
