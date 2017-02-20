import cv2
import numpy as np
from skimage import img_as_ubyte
from sklearn.datasets import fetch_mldata
from skimage.morphology import disk
from skimage.measure import label  
from skimage.measure import regionprops  
from sklearn.neighbors import KNeighborsClassifier
import os
import math

def reshape(bbox,w,h, img):
    number = np.zeros((28, 28));
    for i in range(0, h):
        for j in range(0, w):
            number[i, j] = img[bbox[0]+i-1, bbox[1]+j-1]
    return number
    


def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
 

def transform(mnistData):
    close_kernel = np.ones((5, 5), np.uint8)
    for i in range(0, len(mnistData)):
        number = mnistData[i].reshape(28, 28)
        th = cv2.inRange(number, 150, 255)
        closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, close_kernel)
        labeled = label(closing)
        regions = regionprops(labeled)
        if(len(regions) > 1):
            max_width = 0
            max_height = 0
            for region in regions:
                tempBbox = region.bbox
                tempWidth = tempBbox[3] - tempBbox[1]
                tempHeight = tempBbox[2] - tempBbox[0]
                if(max_width < tempWidth and max_height < tempHeight):
                    max_height = tempHeight
                    max_width = tempWidth
                    bbox = tempBbox
        else:
            bbox = regions[0].bbox
        img = np.zeros((28, 28))
        x = 0
        for w in range(bbox[0], bbox[2]):
            y = 0
            for h in range(bbox[1], bbox[3]):
                img[x, y] = number[w, h]
                y += 1
            x += 1
        mnistData[i] = img.reshape(1, 28*28)


import operator 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
 
def checkNumber(brojevi,knn_num,w,h,region,bbox):
    x=bbox[1]
    y=bbox[0]
    for clan in brojevi:
        #Ako se dati broj nalazi dijagonalno ispod trenutnog onda ga ignorisemo
        if(clan[0] == knn_num and clan[1] < x+7  and clan[2] < y+7 and clan[3] == w):
            return False
    brojevi.append((knn_num, x, y, w))

file = open('out.txt','w');

file.write('E2 190/2013 Darko Tacic\nfile sum\n')

str_elem_line = disk(2);
str_elem_number = disk(1);
mnist = fetch_mldata('MNIST original')
train = mnist.data

transform(train)    

train_labels = mnist.target
knn = KNeighborsClassifier(n_neighbors=1, algorithm='brute').fit(train, train_labels)
videoNumber = 10;




ROOT = os.path.dirname(os.path.abspath(__file__))

for videoIndex in range(0,videoNumber):
    
    videoName = 'video-' + str(videoIndex) + '.avi';
    VIDEO_PATH= os.path.join(ROOT, videoName)
    cap = cv2.VideoCapture(VIDEO_PATH);
    
    frameIndex = 0;
    suma = 0;
    brojevi = []
    
    while (cap.isOpened()):
        
        #citanje framea
        ret, frame = cap.read();
        print frameIndex;
        if(frameIndex%2 != 0):
            frameIndex += 1
            continue

        if ret == True:
                    grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
                    #pronalazenje linije Hough-transformacijom    
                    if(frameIndex == 0):
                        ret,thresh1 = cv2.threshold(grayScale,4,54,cv2.THRESH_BINARY)
                        erosion = cv2.erode(thresh1, str_elem_line, iterations=1)
                        byte = img_as_ubyte(erosion)
                        lines = cv2.HoughLinesP(byte, 1, np.pi/180, 100, minLineLength=1, maxLineGap=100)
                        #Koordinate prave
                        x1 = lines[0][0][0]
                        y1 = lines[0][0][1]    
                        x2 = lines[0][0][2]
                        y2 = lines[0][0][3]
                    #Izdvajanje regiona za brojeve
                    ret,thresh2 = cv2.threshold(grayScale,160,200,cv2.THRESH_BINARY)
                    thresh2_closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, str_elem_number)
                    labeled = label(thresh2_closing)
                    regions = regionprops(labeled)
                    
                    for region in regions:
                        bbox = region.bbox
                        h = bbox[2] - bbox[0]  # visina
                        w = bbox[3] - bbox[1]  # sirina
                        if(h<8):
                            continue;
                        #postavljanje regiona na 28x28
                        number = reshape(bbox,w,h,thresh2_closing)
                        #number = number.reshape(784, 784)
                        knn_num = int(knn.predict(number.reshape(1,28*28)))
                        #validNumber = checkNumber(brojevi,knn_num,w,h,region,bbox)
                        frameIndex += 1
        else:
            break
        
    for clan in brojevi:
        suma += clan[0]
    print suma;
        
        

cap.release()
cv2.destroyAllWindows()
file.close();