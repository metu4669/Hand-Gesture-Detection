import tensorflow as tf
import numpy as np
import os
import cv2
from keras import backend as k
import win32api
import win32con


model = tf.keras.models.load_model('gesture_final_model.model')
Categories = ['5', '4', '1', '3', '2', '0']


def get_img_contour_thresh(img, m, b):
    x, y, w, h = 0, 100, 200, 200
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh1 = cv2.threshold(blur, m, b, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


def click(x, y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)


def hold(x, y):
    win32api.SetCursorPos((x, y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)


def video_capture():
    i = 0
    m = 175
    b = 255
    movable = False
    captured = cv2.VideoCapture(0)
    while True:
        _, img = captured.read()

        # Flipping Image
        # img = cv2.flip(img, 0) # Vertical
        img = cv2.flip(img, 1) # Horizontal

        # Or you can make both
        # img = cv2.flip(img, -1)

        x, y, width, height = 0, 100, 200, 200
        hand_box = img

        cv2.rectangle(img, (x, y), (x+width, y+height), (255, 0, 0), 1)

        image_x, image_y, _ = img.shape
        newimage = cv2.resize(img, (int(image_y), int(image_x)))

        hand_box_contour, contours, thresh = get_img_contour_thresh(hand_box, m, b)
        resized_thres = cv2.resize(thresh, (28, 28))

        LP = cv2.waitKey(5) & 0XFF
        new_image = np.array(resized_thres)
        new_image = new_image.astype('float32')
        new_image /= 255

        if k.image_data_format() == 'channels_first':
            new_image = new_image.reshape(1, 28, 28)
        else:
            new_image = new_image.reshape(28, 28)
        vv = new_image
        new_image = np.expand_dims(new_image, axis=0)
        ans = model.predict(new_image).argmax()
        result = str(Categories[ans])

        cv2.putText(newimage, "Deger : " + result, (0, 220),
                    cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

        c = max(contours, key = cv2.contourArea)

        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        xCoeff = 3
        yCoeff = 3
        nX = xCoeff*cX
        nY = yCoeff*cY

        print(result)
        if movable:
            if result == "5":
                win32api.SetCursorPos((nX, nY))
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, nX, nY, 0, 0)
            elif result == "0":
                hold(nX, nY)

        # draw the contour and center of the shape on the image
        # cv2.drawContours(hand_box, [c], -1, (255, 0, 0), 2)
        # cv2.circle(hand_box, (cX, cY), 7, (255, 255, 255), -1)
        # cv2.putText(hand_box, "X: " + str(cX) + " - Y: " + str(cY), (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Image Panel", thresh)
        cv2.imshow("Hand Box", img)

        if LP == ord('p'):
            if movable:
                movable = False
            else:
                movable = True
        if LP == ord('q'):
            break
        if LP == ord('s'):
            direc = os.path.join(os.getcwd(), "Drop")
            file = os.path.join(direc, str(i) + ".png")
            cv2.imwrite(file, newimage)
            i += 1
    captured.release()
    cv2.destroyAllWindows()


def main():
    video_capture()


if __name__ == '__main__':
    main()


















