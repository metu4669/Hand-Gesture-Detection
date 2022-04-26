import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def get_img_contour_thresh(img):
    x, y, w, h = 0, 50, 300, 300
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh1 = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


def video_capture():
    i=0
    captured = cv2.VideoCapture('http://192.168.178.101:4747/video')
    while True:
        _, img = captured.read()

        # Flipping Image
        # img = cv2.flip(img, 0) # Vertical
        img = cv2.flip(img, 1) # Horizontal

        # Or you can make both
        # img = cv2.flip(img, -1)

        x, y, width, height = 0, 0, 200, 200
        hand_box = img[x:x+width, y:y+height]

        cv2.rectangle(img, (x, y), (x+width, y+height), (255, 0, 0), 1)

        image_x, image_y, _ = img.shape
        newimage = cv2.resize(img, (int(image_y/2), int(image_x/2)))

        hand_box_contour, contours, thresh = get_img_contour_thresh(hand_box)

        resized_thres = cv2.resize(thresh, (28, 28))

        # cv2.imshow("Image Panel", img)
        cv2.imshow("Hand Box", resized_thres)
        cv2.imshow("Hand Image", newimage)

        LP = cv2.waitKey(5) & 0XFF

        current_directory = os.getcwd()
        train_folder_name = "Train Data"
        train_folder = os.path.join(current_directory, train_folder_name)
        new_image_name = os.path.join(train_folder, str(i) + ".png")

        if LP == ord('s'):
            cv2.imwrite(new_image_name, resized_thres)
            i +=1
        if LP == ord('q'):
            break
    captured.release()
    cv2.destroyAllWindows()


def main():
    video_capture()


if __name__ == '__main__':
    main()
