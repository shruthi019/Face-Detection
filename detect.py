import cv2

def classify(gray, img):    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roigray = gray[y:y+h, x:x+w]
        roicolor = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roigray, 1.2, 5)
        smiles = smile_cascade.detectMultiScale(roigray, 1.9, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roicolor, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roicolor, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return img

#Use absolute path name for the following, here it is relative 
face_cascade = cv2.CascadeClassifier('/haar_cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/haar_cascades/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('/haar_cascades/haarcascade_smile.xml')

cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cap = cv2.VideoCapture(0)

while(1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = classify(gray, frame)
    cv2.imshow('video', canvas)
    k = cv2.waitKey(1)
    if k == 27: #press Esc to stop
        break

cap.release()
cv2.destroyAllWindows()
