#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@Env     		:   grizli
@File    		:   ~/emulator/emulator_v0.8.5/utils.py
@Time    		:   2024/08/14 12:50:25
@Author  		:   Run Wen
@Version 		:   0.8.5
@Contact 		:   wenrun@pmo.ac.cn
@Description	:   Some small functions used in emulator.
'''

import numpy as np 
import math 

c = 2.9979e8 # m/s
c_aa = 2.9979e18 #AA/s
h = 6.626e-27 # erg*s

def uJy2mAB(flux): return -2.5*np.log10(flux)+23.9
def fnu2mAB(flux): return -2.5*np.log10(flux)-48.6
def mAB2uJy(mag): return np.power(10,(mag-23.9)/-2.5)
def mAB2fnu(mag): return np.power(10,(mag+48.6)/-2.5)
def dflux(mag_err,flux): return mag_err*flux/1.08574
def row2arr(x): return np.array(list(x))
def flam2fnu(wave,flam): return (wave**2/c_aa)*flam
def fnu2flam(wave,fnu): return (c_aa/wave**2)*fnu
def fnu2fphot(wave,fnu): return (fnu / (wave*h))
def fphot2fnu(wave,fphot): return  (fphot * (wave*h))

def find_nearest(array, value):
    '''
    Utility function to find array element closest to input value
    
    parameters:
    ----------
        array - array in which you want to find the value - [array]
        value - value you want to find in the array - [float]
        
    return:
    ------
        idx - index of the array containing the element closest to input value - [int]
    '''
    array = array[np.isfinite(array)]
    idx = (np.abs(array-value)).argmin()
    return idx

def gaussian(length, std):
    """ 
    A simple 1D gaussian kernel generator

    parameters:
    ----------
        length - the size of this gaussian kernel - [int]
        std - standard deviation of the gaussian kernel - [int]
    
    return:
    ------
        y - normalized 1D gaussian kernel with size of 2*length+1 - [array]
    """
    size = np.arange(-length,length+1,1)
    y = np.exp(-np.power(size , 2.) / (2 * np.power(std, 2.)))
    return y/sum(y)


def odd(value):
    """
    Detect the value whether it is an odd, if not, return an odd number. 

    parameters:
    ----------
        value - the number to be test for whether odd - [float]

    return:
    ------
        i - an odd number - [int]
    """
    i = math.ceil(value)
    if i % 2 == 0: return i +1
    else: return i

def array_cut(array1, array2, cut_range):
    """
    
    """
    mask = (array1 >= cut_range[0]) & (array1 <= cut_range[-1])
    
    # 根据索引截取波长和流量数组
    truncated_array1 = array1[mask]
    truncated_array2 = array2[mask]
    # print(mask)
    return truncated_array1, truncated_array2

def interp(xout, xin, yin):
    """Applies `numpy.interp` to the last dimension of `yin`"""    

    yout = [np.interp(xout, xin, y) for y in yin.reshape(-1, xin.size)]
    
    return np.reshape(yout, (*yin.shape[:-1], -1))
