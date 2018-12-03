import numpy as np
import os
import glob
import time
from keras.preprocessing import image
from PIL import Image

def resize_centered(img):
    w = img.size[0]; h = img.size[1]; c = len(img.split())
    maior = w
    if(h > maior):
      maior = h

    if(maior % 2 == 1):
      maior += 1

    n = maior

    old_size = img.size
    new_size = (n, n)
    new_im = Image.new("RGB", new_size)   ## luckily, this is already black!
    new_im.paste(img, (int((new_size[0]-old_size[0])/2), int((new_size[1]-old_size[1])/2)))

    return new_im

def list_to_array(set, image_size=(224, 224), channels=3):
    gray = channels == 1

    x = []

    for img_path in set:
      img = image.load_img(img_path, grayscale=gray, target_size=None)
      img_pad = resize_centered(img)
      res = image.img_to_array(img_pad.resize(image_size))

      # img.show()
      # img_pad.show()
      # parar

      x.append(res)

    x = np.asarray(x)
    x = x.astype('float32')
    x /= 255

    return x
