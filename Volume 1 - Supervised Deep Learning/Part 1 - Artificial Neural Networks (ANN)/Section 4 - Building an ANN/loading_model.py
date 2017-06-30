# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 16:17:56 2017

@author: Domin8R
"""
import keras
import numpy as np
from keras.models import load_model

model = load_model('my_model.h5')
new_prediction = model.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)
print(new_prediction)

