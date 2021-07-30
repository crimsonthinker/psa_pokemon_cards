from trainers.modules.preprocessor import extract_front_content
import cv2


if __name__ == '__main__':
    image = cv2.imread("./data/2135041002/front.jpg")
    dd = extract_front_content(image)
    cv2.imwrite("test.png", dd)