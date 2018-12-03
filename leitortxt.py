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
	    linha = list(map(int, line.split('-')))
	    y.append(linha)
	return y    