import cv2

image = cv2.imread("/workspace/FaceNist/ImageFolder/person_05/img_050.png")

image = cv2.resize(image, (112,112))

cv2.imwrite("/workspace/FaceNist/ImageFolder/img_test.png", image)