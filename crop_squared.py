import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import cv2
import os

import shutil



PATH_IMG = 'raw_data/data/train/'
start,treshold,percentage=8,23,0.97




def is_black(image,start,treshold,percentage):


    #treshold = le seuil de noir. Noir absolu = 0 par exemple



    #percentage= le pourcentage de la ligne qui est noire
    # Si une ligne horizontale ou verticale noire


    return  np.sum(image[start,start:-start,:]<treshold)/(len(image[0])*3)>percentage or np.sum(image[start:-start,start,:]<treshold)/(image.shape[0]*3)>percentage

def is_black2(image,start,treshold,percentage):

    #test couleur noir uniquement sur une petite ligne du coin superieur gauche au cas où le cercle déborderait sur les marges
    return np.sum(image[start,start:start+85,:]<treshold)/(100*3) >percentage



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

    #if x0>start and x1<image.shape[1]-start:

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
def crop_square_external_raw(image,start,treshold,percentage):

    coordinates=return_coordinates(image,start,treshold,percentage)
    x0,x1,y0,y1=coordinates
    new_image=image[y0:y1,x0:x1].copy()

    return new_image
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

def crop_square_internal(image,start,treshold,percentage):

    if is_black2(image,start,treshold,percentage):
        coordinates=return_coordinates(image,start,treshold,percentage)
        center=return_center(coordinates)
        radius=return_radius(coordinates,center)

        c0,c1=center


        half_side= (2**0.5)*(radius-15)/2




        x0=int(c0-half_side)

        x1=int(c0+half_side)
        y0=int(c1-half_side)

        y1=int(c1+half_side)


        new_image=image[y0:y1,x0:x1].copy()

        return new_image
    else:
        return image


def crop_square_internal_raw(image,start,treshold,percentage):

    coordinates=return_coordinates(image,start,treshold,percentage)
    center=return_center(coordinates)
    radius=return_radius(coordinates,center)

    c0,c1=center



    half_side= (2**0.5)*(radius-15)/2
    x0=int(c0-half_side)

    x1=int(c0+half_side)
    y0=int(c1-half_side)

    y1=int(c1+half_side)


    new_image=image[y0:y1,x0:x1].copy()

    return new_image
def get_image(file,directory):



    image_path = f'{PATH_IMG}{directory}/{file}'

    img = cv2.imread(image_path)

    return img

def get_image_color_lecture(file):
    #better for  reading the image in jupyter but potential color issues  in writing

    image_path = f'{PATH_IMG}{file}'


    #img = cv2.imread(image_path)[...,[2,1,0]]
    img = cv2.imread(image_path,cv2.COLOR_BGR2RGB)

    return img
def get_image_color_modif(file):
    #color inversion in writting, be careful  when using

    image_path = f'{PATH_IMG}{file}'

    img = cv2.imread(image_path)[...,[2,1,0]]

    return img


def move_original_file(file,directory_name):
    source_file = f'{PATH_IMG}{directory_name}/{file}'
    destination_folder= directory_name

    shutil.copy(source_file, destination_folder)

def save_squared_files(start,treshold,percentage,files,directory_name):

    #you need to create directories manually first


    for file in files:


        image=get_image(file,directory_name)


        if is_black2(image,start,treshold,percentage):

            new_image=crop_square_internal_raw(image,start,treshold,percentage)
            if new_image.shape[0] <256:

                extern_image=crop_square_external_raw(image,start,treshold,percentage)
                if extern_image.shape[0] <256:


                    move_original_file(file,directory_name)
                else:
                    cv2.imwrite(f"{directory_name}/cropped_{file}", extern_image,[cv2.IMWRITE_JPEG_QUALITY, 95])


            else:


                cv2.imwrite(f"{directory_name}/cropped_{file}", new_image,[cv2.IMWRITE_JPEG_QUALITY, 95])



        else:

            move_original_file(file,directory_name)
files_AK = [x for x in os.listdir(f'raw_data/data/train/AK') if not x.startswith('.')]
files_BCC = [x for x in os.listdir(f'raw_data/data/train/BCC') if not x.startswith('.')]

files_BKL = [x for x in os.listdir(f'raw_data/data/train/BKL') if not x.startswith('.')]

files_DF = [x for x in os.listdir(f'raw_data/data/train/DF') if not x.startswith('.')]

files_MEL = [x for x in os.listdir(f'raw_data/data/train/MEL') if not x.startswith('.')]

files_NV = [x for x in os.listdir(f'raw_data/data/train/NV') if not x.startswith('.')]



files_SCC = [x for x in os.listdir(f'raw_data/data/train/SCC') if not x.startswith('.')]
files_VASC = [x for x in os.listdir(f'raw_data/data/train/VASC') if not x.startswith('.')]
# to work on the files from BCC use the line below
# save_squared_files(start,treshold,percentage,files_BCC,'BCC')
