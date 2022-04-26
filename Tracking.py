import tensorflow as tf
import numpy as np
import os
import cv2
from keras import backend as k
import win32api
import win32con


model = tf.keras.models.load_model('gesture_final_model.model')
Categories = ['5', '4', '1', '3', '2', '0']
IP = "http://192.168.82.101:4747/video"
IP = 0

# Neural Network Guess
def make_guess(thresh):
    resized = cv2.resize(thresh, (28, 28))
    new_image = np.array(resized)
    new_image = new_image.astype('float32')
    new_image /= 255
    if k.image_data_format() == 'channels_first':
        new_image = new_image.reshape(1, 28, 28)
    else:
        new_image = new_image.reshape(28, 28)
    new_image = np.expand_dims(new_image, axis=0)
    ans = model.predict(new_image).argmax()
    result = str(Categories[ans])
    return result


# Grab Contours Image
def get_img_contour_thresh(img, x, y, w, h):
    img = img[y:y + h, x:x + w]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh1 = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh1 = thresh1[y:y + h, x:x + w]
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    return img, contours, thresh1


# Flip Image Function
def flip_image(img, horizontal=False, vertical=False, both=False):
        if horizontal:
            img = cv2.flip(img, 1)
        if vertical:
            img = cv2.flip(img, 0)
        if both:
            img = cv2.flip(img, -1)
        return img


# Instant Tracking Motion
def tracking(ip):
    x_click = 200
    y_click = 200

    x, y, w, h = 100, 0, 300, 200

    captured = cv2.VideoCapture(0)
    while True:
        screen_black = np.zeros((w, h))
        _, image = captured.read()

        # --------------------------------------------------------------------------------------------------------------
        image = flip_image(image, horizontal=True, vertical=False, both=False)  # Flip Image
        _, contours, thresh = get_img_contour_thresh(image, x, y, w, h)

        # --------------------------------------------------------------------------------------------------------------
        # compute the center of the contour
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(max_contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                center_x, center_y = 0, 0

            # --------------------------------------------------------------------------------------------------------------
            # draw the contour and center of the shape on the image
            cv2.drawContours(screen_black, [max_contour], -1, (255, 0, 0), 2)
            # cv2.circle(screen_black, (center_x, center_y), 7, (255, 255, 255), -1)
            # cv2.putText(screen_black, "X: " + str(center_x) + " - Y: " + str(center_y),
            #            (center_x - 20, center_y - 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # --------------------------------------------------------------------------------------------------------------
        part_image = image[y:y + h, x:x + w]
        partial_edge = cv2.Canny(part_image, x_click, y_click)
        _, part_contour, part_thresh = get_img_contour_thresh(part_image, 0, 0,
                                                              partial_edge.shape[0], partial_edge.shape[1])

        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(max_contour)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                center_x, center_y = 0, 0

            # --------------------------------------------------------------------------------------------------------------
            # draw the contour and center of the shape on the image
            cv2.drawContours(screen_black, [max_contour], -1, (255, 0, 0), 2)
            # cv2.circle(screen_black, (center_x, center_y), 7, (255, 255, 255), -1)
            # cv2.putText(screen_black, "X: " + str(center_x) + " - Y: " + str(center_y),
            #            (center_x - 20, center_y - 20),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # blur = cv2.GaussianBlur(image,(5,5),0)
        # lapl = cv2.Laplacian(image, cv2.CV_64F, ksize=5)
        # log = cv2.Laplacian(blur, cv2.CV_64F, ksize=5)

        # --------------------------------------------------------------------------------------------------------------
        cv2.imshow("Partial Edge", partial_edge)
        # cv2.imshow("Black Panel", screen_black)
        cv2.imshow("Partial Image", part_image)
        cv2.imshow("Partial Thresh", part_thresh)
        print(make_guess(thresh))
        # --------------------------------------------------------------------------------------------------------------
        # Event Listening
        event_listening = cv2.waitKey(5) & 0XFF
        if event_listening == ord('q'):
            break

        if event_listening == ord('z'):
            x_click -= 50
        if event_listening == ord('x'):
            y_click -= 50
        if event_listening == ord('c'):
            x_click -= 50
            y_click -= 50

        if event_listening == ord('a'):
            x_click += 50
        if event_listening == ord('s'):
            y_click += 50
        if event_listening == ord('d'):
            x_click += 50
            y_click += 50

    captured.release()
    cv2.destroyAllWindows()


def main():
    tracking(IP)


if __name__ == '__main__':
    main()
