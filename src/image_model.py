import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from matplotlib.image import imread
from tensorflow.keras.preprocessing import image
import os 



from tensorflow.keras import models, layers, applications

def image_model_pipeline():
    base = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
        pooling='avg'
    )

    inputs = layers.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(6, activation='softmax')(x)
    model = models.Model(inputs, outputs)


    model.load_weights("dominator.h5")

    my_image = image.load_img('C://Users//Yashas/img16.jpg',target_size=(224,224,3))

    my_image = image.img_to_array(my_image)

    my_image = np.expand_dims(my_image, axis=0)

    pred=model.predict(my_image)

    class_idx = np.argmax(pred)

    label_dict = {0:'BacterialBlights',
    1:'Healthy',
    2:'Mosaic',
    3:'RedRot',
    4:'Rust',
    5:'Yellow'}

    pred = model.predict(my_image)
    pred_sorted = pred.argsort(axis=1)
    top3_indices = pred_sorted[0, -1:-4:-1]  
    top3_labels = [label_dict[i] for i in top3_indices] 

    return top3_labels

