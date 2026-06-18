import albumentations as A
import cv2
import random
import numpy as np

class HalfFill127(A.ImageOnlyTransform):
    def __init__(self, always_apply=False,p=0.1):
        super(HalfFill127, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        if random.random() < 0.5:
            img[:, :w//2]=127
        else:
            img[:, w//2:]=127
        return img

def test(image_path):
    image = cv2.imread(image_path)

    transform = A.Compose([HalfFill127(p=1.0)])

    augmented = transform(image=image)

    image_aug = augmented['image']
    cv2.imwrite("/workspace/FaceNist/ImageFolder/person_00/img_000_aug.png", image_aug)
    return image_aug

test('/workspace/FaceNist/ImageFolder/person_00/img_000.png')
