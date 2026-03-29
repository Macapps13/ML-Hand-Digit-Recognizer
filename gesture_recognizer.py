import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model('model.h5')

cap = cv2.VideoCapture(0)

