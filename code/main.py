import process
import cv2

if __name__ == "__main__":
    print("Main Application")
    image_test_path = "./data/f19.jpg"
    image_test = cv2.imread(image_test_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    outout = process.extract(image_test)
    cv2.imwrite("image_test.jpg", outout)
