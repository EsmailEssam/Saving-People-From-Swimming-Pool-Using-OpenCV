import cv2
import math
import os
import numpy as np
from ultralytics import YOLO
import playsound
from mutagen.mp3 import MP3
import time
import miniaudio


def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Calculate the angle in radians
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])

    # Convert from radian to degree
    angle = np.abs(radians * 180 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return angle


def get_cordnations(results, i, hand='left'):
    if hand == 'left':
        shoulder = [int(results[0].keypoints[i][6][0].item()), int(results[0].keypoints[i][6][1].item())]
        elbow = [int(results[0].keypoints[i][8][0].item()), int(results[0].keypoints[i][8][1].item())]
        wrist = [int(results[0].keypoints[i][10][0].item()), int(results[0].keypoints[i][10][1].item())]

    elif hand == 'right':
        shoulder = [int(results[0].keypoints[i][5][0].item()), int(results[0].keypoints[i][5][1].item())]
        elbow = [int(results[0].keypoints[i][7][0].item()), int(results[0].keypoints[i][7][1].item())]
        wrist = [int(results[0].keypoints[i][9][0].item()), int(results[0].keypoints[i][9][1].item())]

    return shoulder, elbow, wrist

file = r'data\assets_alarm.mp3'
audio = MP3(file)
length=audio.info.length

model = YOLO(r"data\yolov8n-pose.pt")


def video_detection(path_x):

    file = r'data\assets_alarm.mp3'
    audio = MP3(file)
    length = audio.info.length

    frame_check = 7  # a threshold to determine if the preson is drowning or not
    l_flag = [0] * 6  # a list of zeros with the length of the number of people
    r_flag = [0] * 6
    frame_count = 0

    video_capture = path_x
    #Create a Webcam Object
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #out=cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P','G'), 10, (frame_width, frame_height))

    model = YOLO(r"data\yolov8n-pose.pt")

    while True:
        ret, frame = cap.read()

        if ret:  # break if it didn't read any frame

            results = model.predict(frame, save=True)  # applay the model on the frame

            # Draw keypoints
            detection = results[0].plot()

            h, w, c = detection.shape

            for i in range(len(results[0].keypoints)):

                nose = [int(results[0].keypoints[i][0][0].item()), int(results[0].keypoints[i][0][1].item())]

                # check on the left hand
                if results[0].keypoints[i][6][0] and results[0].keypoints[i][8][0] and results[0].keypoints[i][10][0]:

                    # get the coordinates of left shoulder, left elbow, and left wrist
                    l_shoulder, l_elbow, l_wrist = get_cordnations(results, i, hand='left')

                    # calculate the angle between left shoulder, left elbow, and left wrist
                    l_ang = calc_angle(l_shoulder, l_elbow, l_wrist)

                    # write the angle
                    cv2.putText(detection, str(int(l_ang)), (l_elbow[0], l_elbow[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 255, 255), 1)

                    # Make sure the person raises his hand for help and the angle btween his arm is more than 150
                    if l_wrist[1] < l_elbow[1] < l_shoulder[1] and l_ang > 140:

                        l_flag[i] += 1
                        if l_flag[i] >= frame_check:
                            cv2.putText(detection, 'Warnning!!! someone need help', (nose[0], nose[1] - 100),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                            stream = miniaudio.stream_file(file)
                            with miniaudio.PlaybackDevice() as device:
                                device.start(stream)
                                time.sleep(length)
                            l_flag[i] = 0

                # check on the right hand
                if results[0].keypoints[i][5][0] and results[0].keypoints[i][7][0] and results[0].keypoints[i][9][0]:

                    # get the coordinates of right shoulder, right elbow, and right wrist
                    r_shoulder, r_elbow, r_wrist = get_cordnations(results, i, hand='right')

                    # calculate the angle between right shoulder, right elbow, and right wrist
                    r_ang = calc_angle(r_shoulder, r_elbow, r_wrist)

                    # write the angle
                    cv2.putText(detection, str(int(r_ang)), (r_elbow[0], r_elbow[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                                (255, 255, 255), 1)

                    # Make sure the person raises his hand for help and the angle btween his arm is more than 150
                    if r_wrist[1] < r_elbow[1] < r_shoulder[1] and r_ang > 140:

                        r_flag[i] += 1
                        if r_flag[i] >= frame_check:
                            cv2.putText(detection, 'Warnning!!! someone need help', (nose[0], nose[1] - 100),
                                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
                            stream = miniaudio.stream_file(file)
                            with miniaudio.PlaybackDevice() as device:
                                device.start(stream)
                                time.sleep(length)
                            r_flag[i] = 0
        yield detection
        #out.write(img)
        #cv2.imshow("image", img)
        #if cv2.waitKey(1) & 0xFF==ord('1'):
            #break
    #out.release()
cv2.destroyAllWindows()