import numpy as np
import cv2
import mahotas as mh
import glob

def checkSegmentation(vLines, mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret, thres = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    labeled, nr_objects = mh.label(thres)

    labelAcc = np.zeros(nr_objects)
    labelArea = np.bincount(labeled.flatten())[1:]

    for i in range(len(vLines)-1):
        band = labeled[:,vLines[i]:vLines[i+1]]
        bins = np.bincount(band.flatten())
        if bins.shape[0] > 1:
            maxArea = np.max(bins[1:])
            maxLabel = np.argmax(bins[1:])

            acc = maxArea/float(labelArea[maxLabel])
            if acc > labelAcc[maxLabel]:
                labelAcc[maxLabel] = acc

    return nr_objects, labelAcc[labelAcc > 0.85].shape[0]


folder = "./"
folder_images = "Dataset/"
results = []

files = glob.glob("./results/*.txt")


for fileName in files:
    print("----------------------- " + fileName + " ----------------------- ")

    totalScore = 1000
    numberOfProcSegm = 0
    numberOfProcHA = 0

    try:
        ########################### Check segmentation ###########################
        totalChars = 0
        foundedChars = 0

        fileseg = open(fileName, 'r')
        content = fileseg.readlines()
        content = [x.strip() for x in content]

        i = 0
        while i < len(content):

            maskpath = content[i].replace('.jpg','C.jpg')
            mask = cv2.imread(folder_images + maskpath)

            ranges = np.asarray(content[i+1].split(' '), dtype = np.int)
            maskCharBand = mask[ranges[0]:ranges[2],ranges[1]:ranges[3]]

            segments = np.asarray(content[i+2].split(' '), dtype = np.int)

            total, founded = checkSegmentation(segments, maskCharBand)
            totalChars += total
            foundedChars += founded

            i += 3
            numberOfProcSegm += 1


        print()
        print("Processed for Segmentation %d,"%numberOfProcSegm)
        print("From %d characters not recognized %d."%(totalChars, totalChars - foundedChars))
        print()

    except Exception:
        print ("Some error")
