import numpy as np
import cv2
import os


from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

######################

path = 'myData'
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32,32,3)
batchSizeVal = 50
epochsVal = 10
stepsPerEpochVal = 2000

###################
images = []
classNo = []
myList = os.listdir(path)
print("Total Number of Classes Detected: ",len(myList))
noOfClasses = len(myList)
print("Importing Classes.....")
for x in range(0,noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x, end=" ")
print(" ")

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)

#### Spliting the Data ####

x_train,x_test,y_train,y_test = train_test_split(images,
                                                 classNo,
                                                 test_size=testRatio)
x_train,x_validation,y_train,y_validation = train_test_split(x_train,
                                                             y_train,
                                                             test_size=valRatio)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)

numOfSamples = []
for x in range(0,noOfClasses):
    #print(len(np.where(y_train==0)[0]))
    numOfSamples.append(len(np.where(y_train==0)[0]))
print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("No. Of Images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()
print(x_train[30].shape)


#### Pre Processing ####

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

#img = preProcessing(x_train[30])
#img = cv2.resize(img,(300,300))
#cv2.imshow("PreProcessed ",img)
#cv2.waitKey(0)


x_train = np.array(list(map(preProcessing,x_train)))
#print(x_train[30].shape)
x_test = np.array(list(map(preProcessing,x_test)))
x_validation = np.array(list(map(preProcessing,x_validation)))

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)


####Augmentation [Rotation Zoom Etc]####

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(x_train)

y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)


def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters, sizeOfFilter1, input_shape=(imageDimensions[0],
                                                               imageDimensions[1], 1), activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = myModel()
print(model.summary())



history = model.fit_generator(dataGen.flow(x_train, y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch = stepsPerEpochVal,
                                 epochs = epochsVal,
                                 validation_data = (x_validation,y_validation),
                                 shuffle=1)

#### PLOT THE RESULTS
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(x_test,y_test,verbose=0)
print("Test Score = ",score[0])
print("Test Accuracy ",score[1])

pickle_out = open("model_trainedteas.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()