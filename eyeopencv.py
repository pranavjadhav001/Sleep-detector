import numpy as np

import cv2
eye_cascade = cv2.CascadeClassifier(r'C:\python36\missing_alert_script\haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
i = 1
while True:
    ret,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray)
    for(ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        
        roi_gray = img[ey:ey+eh,ex:ex+ew]

        cv2.imwrite('sample'+str(i)+'.jpg',roi_gray)

    cv2.imshow('img',img)
    i+=1
    if(cv2.waitKey(1) == ord('q')):
        break


cap.release()
cv2.destroyAllWindows()
    
    
