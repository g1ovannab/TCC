import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import time
import glob
import random
from PIL import Image

NAME = "Tcc_cnn_64_{}".format(int(time.time()))


all_traning_paths = glob.glob('DATASETS/train/*/*.jpg')
all_val_paths = glob.glob('DATASETS/val/*/*.jpg')

img_path=random.choice(all_traning_paths)
print(img_path)
Image.open(img_path)



from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()


datagen.flow_from_directory('DATASETS/train/',
    target_size =(224,224),
    class_mode="binary",
    batch_size=32 
)


from tensorflow.keras.applications.resnet50 import preprocess_input
datagen_resnet = ImageDataGenerator(preprocessing_function=preprocess_input)



train_gen = datagen_resnet.flow_from_directory('DATASETS/train/',
    target_size=(224,224),
    class_mode="binary",
    batch_size=32 
)


validation_gen = datagen_resnet.flow_from_directory('DATASETS/val/',
    target_size=(224,224),
    class_mode="binary",
    batch_size=32
)


from tensorflow.keras.applications.resnet50 import ResNet50


base_model=ResNet50(include_top=False,
    input_shape=(224,224,3)
)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "Tcc_cnn_64_{}".format(int(time.time()))

tensorBoard = TensorBoard(log_dir='logs/{}'.format(NAME))



for layer in base_model.layers:
    layer.trainable=False

modelo = Sequential([ base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(128, activation='relu'),
    Dense(36, activation='relu'),
    Dense(2, activation='Softmax')
])

modelo.summary()



from tensorflow.keras.optimizers import Adam


modelo.compile(optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = modelo.fit(train_gen, 
    validation_data=validation_gen,
    epochs=10,
    batch_size=32,
    callbacks=[tensorBoard]
)


import pandas as pd

pd.DataFrame(history.history)

df = pd.DataFrame(history.history)

df[['accuracy', 'val_accuracy']].plot()

df[['loss', 'val_loss']].plot()





modelo2 = Sequential([ base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(128, activation='relu'),
    Dense(36, activation='relu'),
    Dense(2, activation='Softmax')
])

modelo2.compile(optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history2 = modelo2.fit(train_gen, 
    validation_data=validation_gen,
    epochs=7,
    batch_size=32,
    callbacks=[tensorBoard]
)

pd.DataFrame(history2.history)

df2 = pd.DataFrame(history2.history)

df2[['accuracy', 'val_accuracy']].plot()

df2[['loss', 'val_loss']].plot()





import numpy as np

def predicao(modelo, path):
    img= Image.open(path)
    img = img.resize((224,224))
    img_np=np.array(img)
    img_np = preprocess_input(img_np)
    imp_np2=img_np.reshape(1,224,224,3)
    result = modelo.predict(imp_np2)
    id_max= result[0].argmax()
    index_to_class = {v: k for k, v in train_gen.class_indices.items()}
    plt.title(f'Resultado: {index_to_class[id_max]}')
    plt.imshow(img)




#TESTE DE PACIENTE COM CANCER - MODELO COM 10 ÉPOCAS
predicao(modelo,'DATASETS/val/pacientes_COM_cancer/T0138.2.1.D.2013-09-06.00 (1).jpg')

#TESTE DE PACIENTE SEM CANCER - MODELO COM 10 ÉPOCAS
predicao(modelo,'DATASETS/val/pacientes_SEM_cancer/T0002.1.1.D.2012-10-08.04 (1).jpg')



#TESTE DE PACIENTE COM CANCER - MODELO COM 7 ÉPOCAS
predicao(modelo2,'DATASETS/val/pacientes_COM_cancer/T0202.1.1.D.2013-10-11.07 (1).jpg')

#TESTE DE PACIENTE SEM CANCER - MODELO COM 7 ÉPOCAS
predicao(modelo2,'DATASETS/val/pacientes_SEM_cancer/T0013.1.1.D.2012-10-26.18 (1).jpg')

