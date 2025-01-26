import cv2
import numpy as np

def correct_skew(image_path, output_path):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.rad2deg(theta) - 90
            angles.append(angle)
        median_angle = np.median(angles)
    else:
        print("Не удалось обнаружить линии. Убедитесь, что изображение имеет чёткий текст.")
        return

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(255, 255, 255))

    cv2.imwrite(output_path, rotated)
    print(f"image is saved to {output_path}")


number_image=6
input_image = "Input/input_image" + str(number_image) + ".jpeg"
output_image = "Output/aligned_document" + str(number_image) + ".jpeg"
correct_skew(input_image, output_image)
