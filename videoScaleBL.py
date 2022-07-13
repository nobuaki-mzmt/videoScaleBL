"""
Program to measure scale / body length from video files manually
videoScaleBL.py
N. Mizumoto
"""
# Parameters (modify before run)
input_dir       = r'C:\Users\nobua\python-soft'  # input directly
output_dir      = r'C:\Users\nobua\python-soft'  # output directly
num_ind         = 2                              # number of individuals
video_format    = ".mp4"
frame_interval  = 3000                           # interval for sampling frame

# Loading Libraries
import cv2
import numpy as np
import glob
import os
from keyboard import press
import math
import pickle
import pandas as pd

# Loading data
path = glob.glob(input_dir + "\*" + video_format)
file_nums = list(range((len(path))))

# Dataframe
df_column = ["name", "width", "height", "length", "fps", "frame", "scale"]
for i in range(num_ind):
    df_column.append("bodyLength" + str(i))
df = pd.DataFrame(np.arange(len(path)*(7+num_ind)). reshape(len(path), 7+num_ind),
                  columns=df_column)

# ------------------ main ------------------
for i in file_nums:
    cv2.namedWindow(winname='window')

    # ----- file info -----
    v = path[i]
    file_name = os.path.basename(v)
    name = file_name.split('.')[0]
    video = cv2.VideoCapture(v)
    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    print("name:{}, width:{}, height:{}, count:{}, fps:{}".format(name, width, height, count, fps))
    # ----------

    # region ----- 1. Extract frame -----
    def frame_check(event, x, y, flags, param):
        global frame_okay, frame_id
        frame_okay = False
        if event == cv2.EVENT_LBUTTONDOWN:
            frame_okay = True
            press('enter')
        if event == cv2.EVENT_RBUTTONDOWN:
            frame_okay = False
            frame_id   = frame_id + 3000
            press('enter')

    frame_id = 0
    while True:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = video.read()
        frame_copy = frame.copy()
        cv2.putText(frame_copy,
                    'frame number: ' + str(frame_id),
                    (10, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255),
                    2, cv2.LINE_AA)
        cv2.putText(frame_copy,
                    'Use this frame? Yes -> L click, No -> R click',
                    (10, 100), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255),
                    2, cv2.LINE_AA)
        cv2.imshow('window', frame_copy)
        cv2.setMouseCallback('window', frame_check)
        if frame_id > count:
            print("Error: end of frames. Maybe reduce frame_interval to sample more frames.")
            break
        if cv2.waitKey(0) & frame_okay:
            break
    # endregion ----------

    # region ----- 2. Scale -----
    def line_scale(event, x, y, flags, param):
        global sx0, sy0, sx1, sy1, scale, drawing, end
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            sx0, sy0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(img_copy, (sx0, sy0), (x, y), (0, 0, 255), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            sx1, sy1 = x, y
            cv2.line(img_copy, (sx0, sy0), (sx1, sy1), (0, 0, 255), 1)
            drawing = False
        if event == cv2.EVENT_RBUTTONDOWN:
            end = 1
        press('enter')

    def circle_scale(event, x, y, flags, param):
        global sx0, sy0, sx1, sy1, scale, drawing, end, img_output
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            sx0, sy0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.circle(img_copy, center=(int((x - sx0) / 2) + sx0, int((y - sy0) / 2) + sy0),
                           radius=int(math.sqrt((sx0 - x) ** 2 + (sy0 - y) ** 2) / 2),
                           color=(0, 0, 255), thickness=1)
        elif event == cv2.EVENT_LBUTTONUP:
            scale = math.sqrt((sx0 - x) ** 2 + (sy0 - y) ** 2)
            cv2.circle(img_copy, center=(int((x - sx0) / 2) + sx0, int((y - sy0) / 2) + sy0),
                       radius=int(scale / 2),
                       color=(0, 0, 255), thickness=1)
            cv2.circle(img_output, center=(int((x - sx0) / 2) + sx0, int((y - sy0) / 2) + sy0),
                       radius=int(scale / 2),
                       color=(0, 0, 255), thickness=1)

            drawing = False
        if event == cv2.EVENT_RBUTTONDOWN:
            end = 1
        press('enter')

    global img
    img = frame
    img_copy = img.copy()
    img_output = img.copy()
    end = 0
    drawing = False
    while True:
        cv2.imshow('window', img_copy)
        if drawing:
            img_copy = img.copy()
        cv2.putText(img_copy, 'Circle scaling. L DOWN -> L UP. R click to finish', (10, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                    (0, 0, 255),
                    2, cv2.LINE_AA)
        cv2.setMouseCallback('window', circle_scale)
        if cv2.waitKey(0) & end == 1:
            break
    # endregion

    # region ----- 3. Body length -----
    def obtainBodyLength(event, x, y, flags, param):
        global sx0, sy0, bl, drawing, end, img_output
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            sx0, sy0 = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(img_copy, (sx0, sy0), (x, y), (0, 0, 255), 1)
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.line(img_copy, (sx0, sy0), (x, y), (0, 0, 255), 1)
            cv2.line(img_output, (sx0, sy0), (x, y), (0, 0, 255), 1)
            bl = math.sqrt((x-sx0)**2 + (y-sy0)**2)
            drawing = False
        if event == cv2.EVENT_RBUTTONDOWN:
            end = 1
        press('enter')

    body_length = [0] * num_ind
    for ii in range(num_ind):
        bl, end = 0, 0
        img_copy = img.copy()
        while True:
            cv2.imshow('window', img_copy)
            if drawing:
                img_copy = img.copy()
            cv2.putText(img_copy, 'Body Length. L DOWN -> L UP. R click to finish', (10, 50), cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (0, 0, 255),
                        2, cv2.LINE_AA)
            cv2.setMouseCallback('window', obtainBodyLength)
            if cv2.waitKey(0) & end == 1:
                break
        img = img_copy
        body_length[ii] = bl
    # endregion

    # region ----- 4. Output -----#
    df.iloc[i:(i+1), 0:7] = [name, width, height, count, fps, frame_id, scale,]
    for ii in range(num_ind):
        df.iloc[i:(i+1), 7+ii] = body_length[ii]
    cv2.putText(img_output, name, (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(output_dir + "/" + name + ".jpg", img_output)

cv2.destroyAllWindows()

print(df)

with open(output_dir + '/res.pickle', mode='wb') as f:
    pickle.dump(df, f)

df.to_csv(output_dir + '/res.csv')
