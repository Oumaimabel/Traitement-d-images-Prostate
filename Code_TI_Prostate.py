#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pydicom as dicom
import os
import scipy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib as mpl
import skimage
from skimage import exposure
import cv2


# In[11]:


dataset=dicom.read_file("C:\\Users\\hp\\Desktop\\Prostate dcm\\1-11.dcm")
print(dataset)


# In[12]:


plt.imshow(dataset.pixel_array,plt.cm.bone)
plt.show()


# In[13]:


#Nombre de lignes et nombre de colonnes
print(dataset.Rows)
print(dataset.Columns)


# In[14]:


#size
def dicom_to_array(filename):
    d = dicom.read_file(filename)
    a = d.pixel_array
    return np.array(a)

a1 = dicom_to_array("C:\\Users\\hp\\Desktop\\Prostate dcm\\1-11.dcm")
print(a1.size)


# In[15]:


print(a1.shape)


# In[16]:


print(np.ndarray.min(a1))
print(np.ndarray.max(a1))


# In[17]:


hist,bins = np.histogram(a1, bins=256)
plt.figure()
plt.hist(a1)


# In[18]:


#Si on veut améliorer le contraste d’une image
a1_eq = exposure.equalize_hist(a1)
hist_eq,bins_eq = np.histogram(a1_eq, bins=256)
plt.figure() # créer une nouvelle figure
plt.hist(a1_eq)


# In[19]:


# grayscale
fig1  = plt.figure()
plt.imshow(a1, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig1.suptitle("Original + Gray colormap", fontsize=12)


# In[20]:


fig2 = plt.figure()
plt.imshow(a1_eq, cmap="gray", interpolation="bicubic")
plt.colorbar()
fig2.suptitle("Histogram equalization + Gray colormap", fontsize=12)


# In[21]:


#Slicing 
part=a1[200:400,100:300]
plt.subplot(2,2,1),plt.imshow(a1,cmap = 'gray')
plt.title('image')
plt.subplot(2,2,2),plt.imshow(part,cmap = 'gray')
plt.title('la partie affiché ')

plt.show()


# In[22]:


#Affichage de quelques images
import matplotlib.pyplot as plt
import pydicom
folder = 'C:\\Users\\hp\\Desktop\\Prostate dcm\\' #folder where photos are stored
fig=plt.figure(figsize=(8, 8))
columns=4
rows=5
#plot 9 images
for i in range(1,10):

    # define subplot 1-0.dcm

    x= folder + '1-0' + str(i) + '.dcm'
    ds = pydicom.filereader.dcmread(x)
    fig.add_subplot(rows,columns,i)
    # load image pixels
    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
plt.show()
plt.show()


# In[23]:


#Erosion
import cv2
import numpy as np
a1 = dicom_to_array("C:\\Users\\hp\\Desktop\\Prostate dcm\\1-11.dcm")

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(a1,kernel,iterations = 2)
plt.imshow(erosion,cmap='gray')


# In[24]:


#Dilation
dilation = cv2.dilate(a1,kernel,iterations = 1)
plt.imshow(dilation,cmap='gray')


# In[25]:


#Opening
opening = cv2.morphologyEx(a1, cv2.MORPH_OPEN, kernel)
plt.imshow(opening,cmap='gray')


# In[26]:


#Closing
closing = cv2.morphologyEx(a1, cv2.MORPH_CLOSE, kernel)
plt.imshow(closing,cmap='gray')


# In[27]:


#Morphological Gradient
gradient = cv2.morphologyEx(a1, cv2.MORPH_GRADIENT, kernel)
plt.imshow(gradient,cmap='gray')


# In[28]:


#Top Hat
tophat = cv2.morphologyEx(a1, cv2.MORPH_TOPHAT, kernel)
plt.imshow(tophat,cmap='gray')


# In[29]:


#Black Hat
blackhat = cv2.morphologyEx(a1, cv2.MORPH_BLACKHAT, kernel)
plt.imshow(blackhat,cmap='gray')


# In[30]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
import pylab
import matplotlib.cm as cm
from skimage.color import rgb2gray
from scipy import ndimage
from skimage.io import imread
from skimage.color import rgb2gray
from tkinter import Tk, W, E
from tkinter.ttk import Frame, Button, Entry, Style


# In[31]:


#Filtre Gaussien
blurred_face = ndimage.gaussian_filter(a1, sigma=3)
very_blurred = ndimage.gaussian_filter(a1, sigma=5)
plt.figure(figsize=(11, 6))

plt.subplot(121), plt.imshow(blurred_face, cmap='gray')
plt.title('blurred_face'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(very_blurred, cmap='gray')
plt.title('very_blurred'), plt.xticks([]), plt.yticks([])

plt.show()


# In[32]:


plt.hist(a1.flatten(), bins=50, color='c')
plt.xlabel("Hounsfield Units (HU)")
plt.ylabel("Frequency")
plt.show()


# In[33]:


#Filtre sobel et laplacian
laplacian = cv2.Laplacian(a1,cv2.CV_64F)
sobelx = cv2.Sobel(a1,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(a1,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(a1,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])


# In[34]:


#Filtre gaussien
blurred_face = ndimage.gaussian_filter(a1, sigma=3)
very_blurred = ndimage.gaussian_filter(a1, sigma=5)
plt.figure(figsize=(11, 6))
plt.subplot(121), plt.imshow(blurred_face, cmap='gray')
plt.title('blurred_face'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(very_blurred, cmap='gray')
plt.title('very_blurred'), plt.xticks([]), plt.yticks([])
plt.show()


# In[35]:


#gray = rgb2gray(a1)
plt.imshow(gray, cmap='gray')
gray.shape
gray_r = gray.reshape(gray.shape[0] * gray.shape[1])
for i in range(gray_r.shape[0]):
        if gray_r[i] > gray_r.mean():
            gray_r[i] = 1
        else:
            gray_r[i] = 0
gray = gray_r.reshape(gray.shape[0], gray.shape[1])

plt.subplot(121), plt.imshow(a1, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(gray, cmap='gray')
plt.title('Segmentation'), plt.xticks([]), plt.yticks([])
plt.show()


# In[36]:


#Convolution 2D
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(a1,-1,kernel)
plt.subplot(121),plt.imshow(a1,'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst,'gray'),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()


# In[ ]:




