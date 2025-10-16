#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Name : JANARTHANAN K
# Reg.No: 212223040072


# In[12]:


# 1. Smoothing Filters

# In[1]:Using Averaging Filter
import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[13]:


image1 = cv2.imread("beaut.jpg")   # <-- Replace with your image name
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)


# In[14]:


kernel = np.ones((5,5), np.float32) / 25
image3 = cv2.filter2D(image2, -1, kernel)


# In[15]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(image3)
plt.title("Averaging Filter Image")
plt.axis("off")
plt.show()


# In[16]:


# In[2]:Using Weighted Averaging Filter

kernel1 = np.array([[1,2,1],
                    [2,4,2],
                    [1,2,1]], np.float32) / 16
image4 = cv2.filter2D(image2, -1, kernel1)


# In[17]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(image4)
plt.title("Weighted Average Filter Image")
plt.axis("off")
plt.show()


# In[18]:


# In[3]:Using Gaussian Filter

gaussian_blur = cv2.GaussianBlur(image2, (5,5), 0)


# In[19]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(gaussian_blur)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()


# In[20]:


# In[4]:Using Median Filter

median = cv2.medianBlur(image2, 5)


# In[21]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(median)
plt.title("Median Filter Image")
plt.axis("off")
plt.show()


# In[22]:


# 2. Sharpening Filters

# In[4]: Using Laplacian Kernel

import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[29]:


image1 = cv2.imread("space.jpg")     # <-- replace with your image name
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)


# In[30]:


# Apply averaging filter first (for comparison)
kernel = np.ones((11,11), np.float32) / 121
image3 = cv2.filter2D(image2, -1, kernel)


# In[31]:


# Laplacian sharpening kernel
kernel2 = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])

sharpened_img = cv2.filter2D(image2, -1, kernel2)


# In[32]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(sharpened_img)
plt.title("Sharpened Image (Laplacian Kernel)")
plt.axis("off")
plt.show()


# In[33]:


# In[5]: Using Laplacian Operator

laplacian = cv2.Laplacian(image2, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))   # Convert to displayable form


# In[34]:


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.imshow(image2)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(laplacian)
plt.title("Laplacian Operator Image")
plt.axis("off")
plt.show()


# In[ ]:




