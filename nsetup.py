import numpy as np
import os
import glob
import time
import random
import leitortxt
from sklearn.model_selection import train_test_split as spliting

np.random.seed(seed=int(time.time()))

rng = np.random.RandomState()

def get_folders(data_base):
  data_folders = []
  for name in os.listdir(data_base):
    if(os.path.isdir(data_base + name)):
      data_folders.append(name)
  print(data_folders)

  return data_folders

def flatten(list_to_flatten):
    for elem in list_to_flatten:
        if isinstance(elem,(list, tuple)):
            for x in flatten(elem):
                yield x
        else:
            yield elem

def config_base(database=None, test_prop=0.0, valid_prop=0.0, discart_prop=0.0):

    train_list = []
    y_train_list = []
    test_list = []
    y_test_list = []
    valid_list = []
    y_valid_list = []
    for b in database:
        print("In databaset " + b["url"] + ":")
        #folders = get_folders(b["url"])
        dataset = glob.glob(b["url"]+"/*." + b["img_type"])
        #decartando 90% das amostras
        y = leitortxt.txt_to_int(b["output"]);
        if discart_prop > 0:
            dataset, X_descart, y, y_descart = spliting(dataset, y, test_size=discart_prop, random_state=rng)
        
        datasize = len(dataset)

        test_num = int(datasize*test_prop)
        valid_num = int(datasize*valid_prop)
        train_num = int(datasize - test_num - valid_num)
        '''
        print("In folder " + b["url"]+ ": " + str(datasize) + " images found.")
        print("Train data: " + str(train_num))
        print("Test data: " + str(test_num))
        print("Valid data: " + str(valid_num))
        #print(b["url"]+'marcações.out')
 
        '''
        
       
        X_train, X_test, y_train, y_test = spliting(dataset, y, test_size=test_num, random_state=rng)
        
        X_train, X_valid, y_train, y_valid = spliting(X_train, y_train, test_size=valid_num, random_state=rng)

        train_list.append(X_train)
        test_list.append(X_test)
        valid_list.append(X_valid)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        y_valid_list.append(y_valid)
        '''
        print("Data selected...")
        print("Train: " + str(len(X_train)))
        print("Test: " + str(len(X_test)))
        print("Valid: " + str(len(X_valid)))
        '''

    train_list = list(flatten(train_list))
    test_list = list(flatten(test_list))
    valid_list = list(flatten(valid_list))
    
    #y_train_list = list(flatten(y_train_list))
    #y_test_list = list(flatten(y_test_list))
    #y_valid_list = list(flatten(y_valid_list))
    print(list_valid_list)

    train = list(zip(train_list, y_train_list))
    test = list(zip(test_list, y_test_list))
    valid = list(zip(valid_list, y_valid_list))

    random.shuffle(train)
    random.shuffle(test)
    random.shuffle(valid)

    train_list, y_train_list = zip(*train)
    test_list, y_test_list = zip(*test)
    valid_list, y_valid_list = zip(*valid)

    print("\n\nTOTAL Data selected...")
    print("Train: " + str(len(train_list)))
    print("Test: " + str(len(test_list)))
    print("Valid: " + str(len(valid_list)))
    print(str(len(train_list) + len(test_list) + len(valid_list)) + " Images.")

    return (train_list, y_train_list), (test_list, y_test_list), (valid_list, y_valid_list)
