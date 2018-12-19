#!/usr/bin/python
# coding:utf-8
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# The GPU id to use, usually either "0" or "1"
#os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import setup
import ip
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.applications import vgg19, vgg16, densenet
import json
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def getFile(file):
	arq = open(file, 'r')
	texto = arq.readlines()
	arq.close()
	return int(texto)
	
def setFile(file, texto):
	arq = open(file, 'w')
	arq.writelines(texto)
	arq.close()




def executeCNN(architecture='DenseNet169', MLPinput=4096, MLPhidden=4096, optimizer='sgd', pesos=None, discart_prop=0):
	img_rows = 224
	img_cols = 224
	channels = 3
	num_classes = 4
	batch_size=32

	database = [
    	{"url": 'base/laramin/', "img_type": "jpg", "output":"base/marcacoes.out"}
	]

	

	(train_list, y_train), (test_list, y_test), (valid_list, y_valid) = setup.config_base(database=database, test_prop=0.3, valid_prop=0.1, discart_prop=0)
	X_train = ip.list_to_array(train_list, (img_rows, img_cols), channels)
	X_test = ip.list_to_array(test_list, (img_rows, img_cols), channels)
	X_valid = ip.list_to_array(valid_list, (img_rows, img_cols), channels)

	

	
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	y_valid = keras.utils.to_categorical(y_valid, num_classes)


	if architecture == 'DenseNet169':
		model = densenet.DenseNet169(include_top=True, weights=None, input_shape=(img_cols, img_rows, channels))
	elif architecture=='VGG16':
		model = vgg16.VGG16(include_top=False,weights=None, input_shape=(img_cols, img_rows, channels))
	elif architecture=='VGG19':
		model = vgg19.VGG19(include_top=False, weights=None, input_shape=(img_cols, img_rows, channels))
	
	fully = model.layers.pop()
	fully = (model.layers[-1].output)
	
	#fully = model.output
	#fully = Flatten()(fully)
	if MLPinput>0:
		fully = Dense(units=MLPinput, activation='relu', name="MLPInput")(fully)
	#fully = Dropout(dropout)(fully)
	if MLPhidden>0:
		fully = Dense(units=MLPhidden, activation='relu', name="MLPhidden")(fully)
	
	fully = Dense(units=num_classes, activation='softmax', name="output")(fully)
	model = Model(inputs=model.input,outputs=fully)


	model.summary()
	#print('Memória usada no modelo:', get_model_memory_usage(batch_size,model))
	if(pesos!=None):
		model.load_weights(pesos)

	if optimizer=='sgd':
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(), metrics=['accuracy'])
	else:
		model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.RMSprop(), metrics=['accuracy'])


	score = model.evaluate(X_test, y_test, verbose=0)
	y_pred = model.predict(X_test, verbose=0)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	#s(y_test, y_pred)
	print(convertSolucao(y_test))
	y_pred = np.argmax(y_pred,axis=-1)
	convertSaida()
	#cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
	#confusion_matrix(y_test, np.argmax(y_pred,axis=-1))



def convertSolucao(solucao):
	type(solucao);

def plot(file, title):
	arq = open(file,"r")
	json_dict = json.load(arq)
	arq.close()
	plt.xlabel("Épocas", fontsize=14)
	plt.suptitle("Métricas", fontsize=16)
	plt.ylim(0, 1.0)
	plt.axis([0, len(json_dict['val_acc']), 0.0, 1.2])
	plt.title(title, fontsize=10)
	plt.plot(json_dict['val_acc'],label='Acc Val', color='red', linewidth=1.0) #r--  g^  bs  ro
	plt.plot(json_dict['acc'],label='Acc', color='blue', linewidth=1.0) #r--  g^  bs  ro
	plt.legend(loc="lower right")
	plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


executeCNN(architecture='DenseNet169', MLPinput=0, MLPhidden=0, optimizer='sgd', pesos='/content/densenet/dados/best_densenet.hdf5', discart_prop=0.99)