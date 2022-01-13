from typing import Optional
from urllib import request
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import EfficientNetB3, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array


def predict_imagenet_labels(url: str, top: Optional[int],
                            min_confidence: Optional[float]) -> list[dict[str, float]]:
    if not top and not min_confidence:
        # Default behavior is to return the top-1 label.
        top = 1
        min_confidence = .0
    elif top and not min_confidence:
        # When only top filter given, disable confidence filter.
        min_confidence = .0
    elif not top and min_confidence:
        # When only confidence filter given, disable top filter.
        top = 1000  # all classes available

    input_tensor = load_remote_image(url, (300, 300))

    model = EfficientNetB3(weights=str(Path(__file__).parent / Path("noisystudent_b3.h5")))
    predictions = model.predict(input_tensor)

    formatted_results = []

    for pred in decode_predictions(predictions, top=top)[0]:
        confidence = float(pred[2])
        if confidence < min_confidence:
            break
        formatted_results.append({"label": pred[1], "confidence": confidence})

    return formatted_results


def load_remote_image(url: str, size: tuple[int, int]) -> np.ndarray:
    # Change user agent to prevent some 403 errors.
    req = request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    response = request.urlopen(req).read()
    image = Image.open(BytesIO(response)).resize(size)

    input_tensor = img_to_array(image)
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor


def main():
    from sys import argv
    url = argv[1]
    top = None
    min_confidence = None
    if len(argv) > 2:
        for arg in argv[2:]:
            if arg.startswith("top="):
                top = int(arg.strip("top="))
            elif arg.startswith("minconf="):
                min_confidence = float(arg.strip("minconf="))

    print(predict_imagenet_labels(url, top=top, min_confidence=min_confidence))


if __name__ == "__main__":
    main()
