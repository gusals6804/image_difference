from skimage.measure import compare_ssim
import argparse
import imutils
import cv2

capture = cv2.VideoCapture(1)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, frame = capture.read()
previous_frame = frame
record = False

while True:
    imageA = frame
    imageB = previous_frame

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    print("SSIM: {}".format(score))

    thresh = cv2.threshold(diff, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if score < 0.7:
    #     for c in cnts:
    #         # compute the bounding box of the contour and then draw the
    #         # bounding box on both input images to represent where the two
    #         # images differ
    #         (x, y, w, h) = cv2.boundingRect(c)
    #         cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #         cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

    #cv2.imshow(frame)
    # cv2.imshow("VideoFrame", imageA)
    # cv2.imshow("VideoFrame11", imageB)

    frame_diff = cv2.absdiff(imageA, imageB)

    cv2.imshow('frame diff ', frame_diff)

    key = cv2.waitKey(33)

    if key == 27:
        break
    elif key == 26:
        print("캡쳐")
        cv2.imwrite("D:/" + str("test") + ".png", frame_diff)
    elif key == 24:
        print("녹화 시작")
        record = True
        video = cv2.VideoWriter("D:/" + str("test") + ".avi", fourcc, 15.0, (frame_diff.shape[1], frame_diff.shape[0]))
    elif key == 3:
        print("녹화 중지")
        record = False
        video.release()

    if record == True:
        print("녹화 중..")
        video.write(frame_diff)

    if cv2.waitKey(1) > 0:
        break

    previous_frame = frame.copy()
    ret, frame = capture.read()



capture.release()
cv2.destroyAllWindows()