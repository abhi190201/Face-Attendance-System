import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'img'
if not os.path.exists(path):
    os.makedirs(path)  # Create the directory if it does not exist
    print(f"Directory '{path}' created. Please add images to this directory.")
    exit(1)  # Exit the program after creating the directory
images = []
classnames = []
mylist = os.listdir(path)

# Load images and class names
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    if curimg is not None:  # Check if the image was loaded successfully
        images.append(curimg)
        classnames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Check if encoding was successful
            encodeList.append(encode[0])
    return encodeList

def markAttendance(name):
    # Check if the Attendance.csv file exists, if not create it
    if not os.path.exists('Attendance.csv'):
        with open('Attendance.csv', 'w') as f:
            f.write('Name,Time\n')  # Write header to the file

    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Find encodings for known images
encodeListKnow = findEncodings(images)

print(f'Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:  # Check if the frame was captured successfully
        print("Failed to capture image")
        break

    img5 = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img5 = cv2.cvtColor(img5, cv2.COLOR_BGR2RGB)

    faceCurFrames = face_recognition.face_locations(img5)
    encodeCurFrame = face_recognition.face_encodings(img5, faceCurFrames)

    for encodeface, faceloc in zip(encodeCurFrame, faceCurFrames):
        matches = face_recognition.compare_faces(encodeListKnow, encodeface)
        facedis = face_recognition.face_distance(encodeListKnow, encodeface)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key
        break

cap.release()
cv2.destroyAllWindows()
