#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 18:02:01 2018

@author: joaquim
"""
def txt_to_int(file=None):
	file = open(file, 'r') 
	lines = file.read().split('\n')
	del(lines[len(lines)-1])
	y = []
	for line in lines:
	    #linha = list(map(int, line.split('-')))
		if(line=='0-0-0-0'):
			y.append(0)
		elif (line=='1-1-0-0'):
			y.append(1)
		elif (line=='1-0-1-0'):
			y.append(2)
		elif (line=='1-0-0-1'):
			y.append(3)
	return y    