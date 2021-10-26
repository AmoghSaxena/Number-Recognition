import numpy as np
import cv2
import pickle

################################

width = 640
height = 480
threshold = 0.65

#####################################


cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)


pickle_in = open("model_trained.p","rb")
model = pickle.load(pickle_in)

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOrignal = cap.read()
    img = np.asarray(imgOrignal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processed Image",img)
    img = img.reshape(1,32,32,1)

    #Predict
    classIndex = int(model.predict_classes(img))
    prediction = model.predict(img)
    probVal = np.amax(prediction)
    print(classIndex,probVal)


    if probVal > threshold:
        cv2.putText(imgOrignal,str(classIndex)+"  "+str(probVal),(50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)


    cv2.imshow("Orignal Image",imgOrignal)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
