import os
import cv2
import random

pathvideo = '/data5/zhn/rain'
savedir = '/data5/zhn/rain/img'

shiftSize = [-50, -30, -10, 0, 10, 30, 50]
imgRoisize = [120, 140, 160, 180, 200]

def genImgSize():
    index = random.randint(1, 5) - 1
    return imgRoisize[index]

def genShift():
    index = random.randint(1, 7) - 1
    return shiftSize[index]



def saveGeimg(img, filename):
    
    saveDir = os.path.join(savedir, filename.split('.')[0])
    rows = img.shape[0]
    cols = img.shape[1]
    ii = 0
    for row in range(0, rows, 100):
        for col in range(0, cols, 100):
            realy = row + genShift()
            realx = col + genShift()
            realh = realy + genImgSize()
            realw = realx + genImgSize()
            
            if realx <= 0 or realy <= 0 or realh >= rows or realw >= cols:
                continue
            else:
                savename = saveDir + '_' + str(ii) + '_' + str(realx) + '_' + str(realy) + '.png'
                ii = ii + 1
                imgroi = img[realy:realh, realx:realw]
                cv2.imwrite(savename, imgroi)
                # print('')
            
    # print('')

def main():
    
    files = os.listdir(pathvideo)
    
    for file in files:
        ext = os.path.splitext(file)[1]
        if ext != '.mp4':
            continue
        filename = os.path.join(pathvideo, file)
        frameNum = 0
        video = cv2.VideoCapture(filename)
        
        while video.isOpened():
            retval, img = video.read()
            
            if retval == True:
                frameNum = frameNum + 1
                print('current frame:', frameNum)
                if frameNum % 450 == 0:
                    saveGeimg(img, file)
            else:
                break
        video.release()

if __name__ == '__main__':
    main()