# -*- coding: utf-8 -*-
#The disparity for pixel (x1, y1) is (x2 − x1) if the images are rectified
#For each pixel
#Fundamental matrix
    #Fundamental Matrix encapsulates projective geometry
    #No need of internal parameters
    #No need of knowing relative Pose
    #Show that any pair of corresponding points  X0Fx1 = 0
import math as m;
import numpy as np;
import matplotlib.pylab as plt;
import cv2;
import sys;
from scipy import misc;
from scipy import signal;
from scipy import ndimage;
from PIL import Image

def error(left, right, x_0, y_0, d_0, epsilon=0.001):
    s = 0
    width = 5;
    for i in xrange(x_0-width, x_0+width):
        val1 = (left[i][y_0]+epsilon)/(right[i-d_0][y_0]+epsilon)
        val2 = (right[i-d_0][y_0]+epsilon)/(left[i][y_0]+epsilon)
        s += val1
        s += val2
    #print s;
    return s

def disparity():
    print "tata"
def leftRightTransformation():
    #R = Rotation Matrix
    #V = Translation Vector
    RpPr = Pl-T;
    print 'tete';
def differenceComputation(real,imag):
    alfa = 1000;
    diff = imag - real;
    edgeu = np.exp(-alfa*diff);
    plt.imshow(edgeu,cmap= plt.cm.gray);
    plt.show();
    plt.imsave('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/Convolution/Edge/Dif/Rel/rightEdge%d.png'%alfa,edgeu, cmap='gray');

def ratioComputation(real,imag):
	epsilon = 0.001;
	ratioUp = np.add(real,(0.001 * epsilon));
	ratioDown = np.add(imag,epsilon);
	ratio = np.divide(ratioUp,ratioDown);
	d = np.max(ratio);
	secondDivide = np.divide(ratio,d);
	edge = np.subtract(1,secondDivide);
	plt.imshow(edge,cmap= plt.cm.gray);
	plt.show();
	#plt.imsave('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/Convolution/Edge/Ratio/leftEdge%d.png'%epsilon,edge, cmap='gray');

def edgeComputation(real,imag):
    #FLOAT FLOAT FLOAT
	newIma = procesarMatriz(imag);
	newIma = newIma.astype(int);
	newRea = procesarMatriz(real);
	newRea = newRea.astype(int);
	ratioComputation(newRea,newIma);
	#differenceComputation(newRea,newIma);

def procesarMatriz(matrix):
	#Length of X and Y
	lengthx = matrix.shape[0];
	lengthy = matrix.shape[1];
	#New Image with shape -1 (3D->2D)
	newImage = np.zeros(shape=(lengthx,lengthy));

	for i in range(lengthx):
		for j in range(lengthy):
			#Checks if the image is already 2D (grayscale)
			if(len(matrix[i][j].shape)==0):
				#print "Entro3"
				newImage[i][j] = matrix[i][j];
				#print "MATRIX ", matrix[i][j];
				#print "NEW IMAG ",newImage[i][j]
			elif(len(matrix[i][j].shape)>0):
				#print "This is matrix ", matrix[i][j];
				newImage[i][j] = max(abs(matrix[i][j]));
	return newImage;

#ReadBothImages
def convolution(real,imag):
    #print left.shape;
    #opciones de convolve ndimage paila
        #Esto pasa por el orden de parámetros X,Y --> Y,X
        #Me di cuenta que signal2d<ndimage.convolve (más nitido,más rápido))
        #El orden da lo rapido en signal al parecer
        #Leer más sobre el convolve -- entender las opciones

    #TAKES THE WHOLE PSI INSTEAD OF PSI.REAL and PSI.IMAG
    uConvolveReal = signal.convolve2d(left,real);
    return uConvolveReal;
    #uConvolveReal = ndimage.convolve(left,real, mode='constant');
    #uConvolveImag = ndimage.convolve(left,imag, mode='constant');
    #uConvolveImag = signal.convolve2d(left,imag);
    #edgeComputation(uConvolveReal,uConvolveImag);
    #plt.imshow(uConvolveReal,cmap=plt.cm.gray);
    #plt.show();
def maxmin(image, convolved):
    #SHAPE
    print "Image ",image.shape;
    print "Image[0] ", image[0].shape;
    print "COnvolved ",len(convolved);
    max_array = np.zeros((len(image)+12,len(image[0])+12)) #padding
    for i in convolved:
        max_array = np.maximum(np.abs(i.imag - i.real),np.abs(max_array))
    return max_array

#NoScale
def morletWavePsi(Angulo,Sigma):
    angle = Angulo;#Angulo pi/2;
    sigma = Sigma;#Sigma;
    x = np.linspace(-(sigma*3),(sigma*3),num=(2*(sigma*3)+1));
    y = np.linspace(-18.0,18.0,num=(2*(sigma*3)+1));
    [X,Y] = np.meshgrid(x,y);
    #print Angulo;
    u2 = X ** 2 + Y **2;
    #Dot product
    ue0 = X*np.cos(angle) + Y*np.sin(angle);
    div1 = m.pi/ (2 *sigma);
    innerPart = np.exp(1j * div1 * ue0);
    outerPart = np.exp(-0.5 * u2 / (sigma ** 2));
    C2 = sum(sum(innerPart * outerPart)) / sum(outerPart.flatten(1));
    tmp = (innerPart - C2) * outerPart;
    multi = tmp * tmp.conj();
    C1 = 1 / sum(multi.flatten(1)) ** 0.5;
    psi = C1 * (innerPart - C2) * outerPart;
    #plt.imshow(psi.real,cmap=plt.cm.gray);
    #plt.show();
    convolved.append(convolution(psi,psi));

left  = cv2.imread('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/left.png',0);
#right = cv2.imread('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/right.png',0);
#testLeft = cv2.imread('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/Convolution/Dif/Rel/leftEdge100.png',0);
#testRight = cv2.imread('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/Convolution/Dif/Rel/rightEdge100.png',0);

leftFix  = cv2.imread('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/Convolution/Edge/Fix/leftEdgeFix.png',0);
rightFix = cv2.imread('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/Convolution/Edge/Fix/rightEdgeFix.png',0);

angulo = [0,m.pi/4,m.pi/2,3*m.pi/4];#m.pi/2
#Sigma = 2
sigma =2;
convolved = [];
#error();
for i in range(len(angulo)):
    psi = morletWavePsi(angulo[i], sigma);

max_array = maxmin(left, convolved);
#plt.figure(figsize=(10,10))
#plt.imshow(max_array,cmap="gray")
#plt.imsave('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/Convolution/Edge/Fix/rightEdgeFix.png',max_array, cmap='gray');
#plt.show();
#plt.savefig("right_contrast.jpg")
#broadens your horizontal by
disp_array = np.zeros(max_array.shape)
err_array = np.zeros(max_array.shape)+sys.maxint
for i in xrange(30,len(max_array)-30):
    for j in xrange(30,len(max_array[0])-30):
        for k in xrange(-5,15):
            loc_error = error(leftFix,rightFix,i,j,k,5)
            if loc_error < err_array[j][i]:
                err_array[j,i] = loc_error
                disp_array[j][i] = k
#CHANGed the I and J
plt.imshow(disp_array,cmap=plt.cm.gray);
plt.show();
#plt.imsave('/Users/Teban1503/Documents/Universidad de los Andes/Septimo Semestre/NYU/ComputerVision/SecondHw/Pentagon/Convolution/Edge/Fix/stereoFix.png',disp_array, cmap='gray');
