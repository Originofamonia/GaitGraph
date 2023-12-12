import random
import cv2
from matplotlib import pyplot as plt
import albumentations as A

KEYPOINT_COLOR = (0, 255, 0)  # Green


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)

image = cv2.imread('images/keypoints_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

keypoints = [
    (100, 100),
    (720, 410),
    (1100, 400),
    (1700, 30),
    (300, 650),
    (1570, 590),
    (560, 800),
    (1300, 750),
    (900, 1000),
    (910, 780),
    (670, 670),
    (830, 670),
    (1000, 670),
    (1150, 670),
    (820, 900),
    (1000, 900),
]


vis_keypoints(image, keypoints)

transform = A.Compose(
    [A.HorizontalFlip(p=1)],
    keypoint_params=A.KeypointParams(format='xy')
)
transformed = transform(image=image, keypoints=keypoints)
vis_keypoints(transformed['image'], transformed['keypoints'])

transform = A.Compose(
    [A.VerticalFlip(p=1)],
    keypoint_params=A.KeypointParams(format='xy')
)
transformed = transform(image=image, keypoints=keypoints)
vis_keypoints(transformed['image'], transformed['keypoints'])

random.seed(7)
transform = A.Compose(
    [A.RandomCrop(width=768, height=768, p=1)],
    keypoint_params=A.KeypointParams(format='xy')
)
transformed = transform(image=image, keypoints=keypoints)
vis_keypoints(transformed['image'], transformed['keypoints'])


random.seed(7)
transform = A.Compose(
    [A.Rotate(p=0.5)],
    keypoint_params=A.KeypointParams(format='xy')
)
transformed = transform(image=image, keypoints=keypoints)
vis_keypoints(transformed['image'], transformed['keypoints'])


transform = A.Compose(
    [A.CenterCrop(height=512, width=512, p=1)],
    keypoint_params=A.KeypointParams(format='xy')
)
transformed = transform(image=image, keypoints=keypoints)
vis_keypoints(transformed['image'], transformed['keypoints'])



