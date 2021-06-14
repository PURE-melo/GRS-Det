import numpy as np
import cv2
import matplotlib.pyplot as plt


def gaussian(kernel, w):
    sigma = ((kernel-1) * 0.5 - 1) * 0.3 + 0.8
    s = 2*(sigma**2)
    dx = np.exp(-1*w*np.square(np.arange(kernel) - int(kernel / 2)) / s)
    return np.reshape(dx,(-1,1))

def gaussian_mask(bbox,imgshape):
    segmap = np.zeros((imgshape[0],imgshape[1]), dtype=np.float)
    for i, box in enumerate(bbox):
        rect = cv2.minAreaRect(box)
        angle = rect[-1]
        x1,y1,x2,y2 = np.min(box[:,0]), np.min(box[:,1]), np.max(box[:,0]), np.max(box[:,1])
        w = x2 - x1
        h = y2 - y1
        if w<h:
            flag=1
            max_w_h = h
            min_w_h = w
        else:
            flag=0
            max_w_h = w
            min_w_h = h
        if rect[1][0]>rect[1][1]:
            longside = rect[1][0]
            shortside = rect[1][1]
            angle = -angle
        else:
            longside = rect[1][1]
            shortside = rect[1][0]
            angle=-angle
            angle = angle+90

        centerx = w/2
        centery = h/2 
        value= 2.95*np.exp(-0.35*(max_w_h/min_w_h))*max_w_h/min_w_h
        if flag:
            dx = gaussian(w, value) 
            dy = gaussian(h, 0.1*min_w_h/max_w_h)
        else:
            dx = gaussian(w, 0.1*min_w_h/max_w_h)
            dy = gaussian(h, value)
        gau_map = np.multiply(dy,np.transpose(dx))
        rot_mat =  cv2.getRotationMatrix2D((w/2,h/2), angle+flag*90, 1)
        gau_map = cv2.warpAffine(gau_map, rot_mat, (w,h))
        gau_map = (gau_map - np.min(gau_map))/(np.max(gau_map)-np.min(gau_map))
        segmap[y1:y2, x1:x2] = np.maximum(segmap[y1:y2, x1:x2],gau_map)

    return segmap

if __name__ == "__main__":
    #box = [np.array([[134,52],[417,481],[335,535],[52,106]]),np.array([[298,22],[480,316],[439,341],[257,47]])]
    #box = [np.array([[420,8],[787,692],[621,781],[254,97]])]
    box = [np.array([[41,372],[349,452],[340,487],[32,407]]),np.array([[686,539],[1000,622],[993,656],[677,573]]),np.array([[381,465],[678,537],[670,570],[373,498]]),np.array([[526,97],[834,118],[831,154],[523,133]]),np.array([[192,70],[505,95],[503,131],[190,106]])]    
    segmap = gaussian_mask(box, (1000,1000))
    cmap='jet'
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.imshow(segmap,cmap)
    plt.show()
