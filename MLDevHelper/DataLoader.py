import cv2
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def load_and_transform_image(image_path, transformer):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed_image = transformer.forward(image)
    return transformed_image


def image_loader(
    df, batch_size, transformer, num_workers=None, shuffle=None, random_state=None
):
    indices = np.arange(len(df))
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)

    start_idx = 0
    while start_idx < len(df):
        batch_indices = indices[start_idx : start_idx + batch_size]

        # Use ThreadPoolExecutor for parallel image loading and transformation
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for idx in batch_indices:
                image_path, label = df.iloc[idx]
                futures.append(
                    executor.submit(load_and_transform_image, image_path, transformer)
                )

            images = [future.result() for future in futures]
            labels = [df.iloc[idx][1] for idx in batch_indices]

        yield {"images": images, "labels": labels}
        start_idx += batch_size
