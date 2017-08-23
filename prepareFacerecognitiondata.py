import numpy as np;
import matplotlib.pyplot as plt;
import cv2;
from PIL import Image;
from scipy import signal;
from scipy import ndimage;

im1 = cv2.imread("hw4Data/ima/img_37.jpg",0);
im2 = cv2.imread("hw4Data/ima/img_3157.jpg",0);
im3 = cv2.imread("hw4Data/ima/img_159.jpg",0);
im4 = cv2.imread("hw4Data/ima/img_503.jpg",0);
im5 = cv2.imread("hw4Data/ima/img_190.jpg",0);


def calculateEllipseArea(minorAxes,mayorAxes):
    return np.pi * (minorAxes*mayorAxes);

def drawEllipse(image,count):

    i_width, i_height = image.shape
    rows= i_width/32;
    columns =i_height/32;
    temp = [];
    print "count is : ", count
    x = int(imagenes[count][3]);
    y = int(imagenes[count][4]);
    cx = int(imagenes[count][1]);
    cy =int(imagenes[count][0]);
    angle =imagenes[count][2];
    max_radius = max(x,y);
    if (max_radius *2> 32):
        temp_x = x/2;
        temp_y =y/2;
        temp_cx=cx/2;
        temp_cy=cy/2;x<

        cv2.ellipse(imagesDecimated[count],(temp_x,temp_y),(temp_cx,temp_cy),angle,0,360,(255,0,0),1)
        plt.imshow(imagesDecimated[count],cmap=plt.cm.gray);
        plt.show();
        print count, " ",calculateEllipseArea(temp_x,temp_y);
    else:
        cv2.ellipse(image,(x,y),(cx,cy),angle,0,360,(255,0,0),1)
        plt.imshow(image,cmap=plt.cm.gray);
        plt.show();
        print count, " ",calculateEllipseArea(x,y);
    #for row in range(0,rows):
        #for col in range(0,columns):
            #temp = col +1;
            #cv2.rectangle(image,(row,col*32),(temp*32,32),1);

def gaussImages(images):
    for image in images:
        gaussImage = ndimage.filters.gaussian_filter(image,3);
        imagesGaussed.append(gaussImage);

def decimateImages(images):
    for image in images:
        imDecimated = cv2.pyrDown(image,2);
        imagesDecimated.append(imDecimated);
#IDea simplemente get the position with meshgrid (x,y) y dibujar con opencv las coordenadas, try and see
imagesGaussed = [];
imagesDecimated =[];
images = [im1,im2,im3,im4,im5];

dataIm1=[57.003700,38.577270,1.497448,125.690483,81.684807];
dataIm2=[70.960133,45.218661,1.495503,207.647565,94.103697];
dataIm3=[133.803563,78.778994,1.440883,148.385701,151.788563 ];
dataIm4=[69.777250,41.642620,-1.557946,197.738562,115.666369];
dataIm5=[123.137063,75.155654,-1.514282,146.893226,157.169602];

imagenes = [dataIm1,dataIm2,dataIm3,dataIm4,dataIm5];
x = np.linspace(1,32,num=32, dtype=int,);
y = np.linspace(1,32,num=32,dtype=int);
fixedSize= np.meshgrid(x,y,indexing='ij');
fixedSizeX,fixedSizeY = np.meshgrid(x,y,indexing='ij');
gaussImages(images);
decimateImages(imagesGaussed);

count =0;
for imagen in images:
    drawEllipse(imagen,count);
    count+=1;
