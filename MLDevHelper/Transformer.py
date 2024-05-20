import numpy as np
import cv2
import matplotlib.pyplot as plt


class TransformerList:
    class Node:
        def __init__(self, method, params):
            self.val = None
            self.next = None
            self.method = method
            self.params = params

    def __init__(self):
        self.root = None
        # self.create_transformer_list(method_list)

    def insert_at_head(self, new_node):
        if self.root is None:
            self.root = new_node
        else:
            new_node.next = self.root
            self.root = new_node

    def normalize_image(self, mean=None, std=None):
        def normalize_image_helper(image, *args):
            normalized_image = image.astype(np.float32) / 255.0
            if mean is not None and std is not None:
                mean_val = np.array(args[0]).reshape(1, 1, len(mean))
                std_val = np.array(args[1]).reshape(1, 1, len(std))
                normalized_image = (normalized_image - mean_val) / std_val
            return normalized_image

        return normalize_image_helper, (mean, std)

    def resize_image(self, size):
        def resize_helper(image, *args):
            return cv2.resize(image, args[0], interpolation=cv2.INTER_LINEAR)

        return resize_helper, (size,)

    def color_jitter(self, brightness=0, contrast=0, saturation=0, hue=0):
        def color_jitter_helper(image, *args):
            if len(args) != 4:
                raise ValueError(
                    "Expected exactly 4 arguments (brightness, contrast, saturation, hue)"
                )

            brightness, contrast, saturation, hue = args

            if brightness != 0:
                delta = 255 * brightness
                image = np.clip(image + delta, 0, 255).astype(np.uint8)

            if contrast != 0:
                alpha = 1.0 + contrast
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                gray = (gray - 128) * alpha + 128
                image = np.clip(image * alpha, 0, 255).astype(np.uint8)

            if saturation != 0:
                alpha = 1.0 + saturation
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = np.clip(image * alpha, 0, 255).astype(np.uint8)

            if hue != 0:
                hue_factor = np.random.uniform(-hue, hue)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                image[:, :, 0] = (image[:, :, 0].astype(int) + hue_factor) % 180
                image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

            return image

        return color_jitter_helper, (brightness, contrast, saturation, hue)

    def random_rotate_image(self, angle, threshold=0.5):
        def random_rotate_image_helper(image, *args):
            if len(args) != 2:
                raise ValueError("Expected exactly 2 arguments (angle, threshold)")

            angle, threshold = args

            if np.random.rand() > threshold:
                height, width = image.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D(
                    (width / 2, height / 2), angle, 1
                )
                image = cv2.warpAffine(image, rotation_matrix, (width, height))

            return image

        return random_rotate_image_helper, (angle, threshold)

    def random_horizontal_flip(self, threshold=0.5):
        def random_horizontal_flip_helper(image, *args):
            if len(args) != 1:
                raise ValueError("Expected exactly 1 argument (threshold)")

            threshold = args[0]

            if np.random.rand() > threshold:
                image = np.fliplr(image)

            return image

        return random_horizontal_flip_helper, (threshold,)

    def random_vertical_flip(self, threshold=0.5):
        def random_vertical_flip_helper(image, *args):
            if len(args) != 1:
                raise ValueError("Expected exactly 1 argument (threshold)")

            threshold = args[0]

            if np.random.rand() > threshold:
                image = np.flipud(image)

            return image

        return random_vertical_flip_helper, (threshold,)

    def create_transformer_list(self, method_list):
        method_list.reverse()
        for method in method_list:
            fun, params = method
            new_node = self.Node(fun, params)
            self.insert_at_head(new_node)

    def forward(self, image):
        current_node = self.root
        current_node.val = current_node.method(image, *current_node.params)
        previous_image = current_node.val
        current_node = current_node.next

        while current_node:
            current_node.val = current_node.method(previous_image, *current_node.params)
            previous_image = current_node.val
            current_node = current_node.next

        return previous_image


# image = cv2.imread(r"C:\Users\mahes\DSA PROJ\anteater\anteater-0003.jpg")
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# transformer_list = TransformerList()

# transformer_list.create_transformer_list(
#     [
#         transformer_list.resize_image((200, 200)),
#         transformer_list.color_jitter(
#             brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
#         ),
#         transformer_list.random_rotate_image(angle=30),
#         transformer_list.random_horizontal_flip(threshold=0.5),
#         transformer_list.random_vertical_flip(threshold=0.5),
#     ]
# )

# transformed_image = transformer_list.forward(image)

# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Original Image")
# plt.subplot(1, 2, 2)
# plt.imshow(transformed_image)
# plt.title("Transformed Image")
# plt.show()
