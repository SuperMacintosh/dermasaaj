import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import cv2

def is_black(image,start,treshold,percentage):

    #treshold = le seuil de noir. Noir absolu = 0 par exemple

    #percentage= le pourcentage de la ligne qui est noire
    return  np.sum(image[start,start:-start,:]<treshold)/(len(image[0])*3)>percentage

def create_x0(image,start,treshold,percentage):

    y=round(image.shape[0]/2)

    for i in range(start,image.shape[1]-start-5):



        if (np.sum(image[y,i:i+5,:]>treshold)/15)>=percentage:


            return i

def create_x1(image,start,treshold,percentage):

    y=round(image.shape[0]/2)
    length=image.shape[1]

    for i in range(start,image.shape[1]-start-5):

        if (np.sum(image[y,length-5-i:length-i,:]>treshold)/15)>=percentage:
            return image.shape[1]-i



def create_y0(image,start,treshold,percentage):

    x=round(image.shape[1]/2)


    for i in range(start,image.shape[0]-start):
        if (np.sum(image[i:i+5,x,:]>treshold)/15)>=percentage:
            return i

def create_y1(image,start,treshold,percentage):

    x=round(image.shape[1]/2)
    length=image.shape[0]


    for i in range(start,image.shape[0]-start-5):
        if (np.sum(image[length-5-i:length-i,x,:]>treshold)/15)>=percentage:
            return image.shape[0]-i

def return_coordinates(image,start,treshold,percentage):
    x0=create_x0(image,start,treshold,percentage)
    x1=create_x1(image,start,treshold,percentage)

    y0=create_y0(image,start,treshold,percentage)
    y1=create_y1(image,start,treshold,percentage)

    if x0>start and x1<image.shape[1]-start:


                return [x0,x1,y0,y1]

def return_center(coordinates):
    x=round((coordinates[1]+coordinates[0])/2)
    y=round((coordinates[3]+coordinates[2])/2)

    return [x,y]

def return_radius(coordinates,center):
    radius1=center[0]-coordinates[0]
    radius2=coordinates[1]-center[0]
    radius3=coordinates[3]-center[1]

    radius4=center[1]-coordinates[2]


    return min([radius1,radius2,radius3,radius4])

def crop_square_external(image,start,treshold,percentage):

    if is_black(image,start,treshold,percentage):
        coordinates=return_coordinates(image,start,treshold,percentage)
        x0,x1,y0,y1=coordinates
        new_image=image[y0:y1,x0:x1].copy()

        return new_image
    else:
        return image


def crop_square_internal(image,start,treshold,percentage):

    if is_black(image,start,treshold,percentage):
        coordinates=return_coordinates(image,start,treshold,percentage)
        center=return_center(coordinates)
        radius=return_radius(coordinates,center)

        c0,c1=center

        half_side= (2**0.5)*radius/2
        x0=int(c0-half_side)

        x1=int(c0+half_side)
        y0=int(c1-half_side)

        y1=int(c1+half_side)


        new_image=image[y0:y1,x0:x1].copy()

        return new_image
    else:
        return image
