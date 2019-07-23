import cv2

# read classifier

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


# get list of cameras

index = 0
arr = []
while True:
    cap = cv2.VideoCapture(index)
    if not cap.read()[0]:
        break
    else:
        arr.append(index)
    cap.release()
    index += 1

# open last available camera

last_cam_index = arr[-1]
cap = cv2.VideoCapture(last_cam_index)


# main loop

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    n = 0
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        n += 1
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    target = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    text = 'Faces: ' + str(n)
    cv2.putText(
        target,
        text, 
        (10,60), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        2,
        (0,255,255),
        3
    )




    cv2.imshow('Face Detection',target)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
