#Training

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

datadir = 'train_images/'
hex_min = u'0900'
hex_max = u'097f'
img_size = [64,64,1]

import os
counts = {}
n_files = 0
for i in os.listdir(datadir):
    if i.find('.png')!=-1:
        filename = i.split('.')[0].split('_')[3:]
        n_files+=1
        for j in filename:
            if j in counts.keys():
                counts[j]+=1
            else:
                counts[j]=1
'''
for i in sorted(counts.keys()):
    print(i, counts[i], chr(int(i)))
'''

label_list = sorted(counts.keys())
template_label = np.zeros((len(counts.keys())))

#for i in label_list:
#    print(i, ',')

#print('Range : %d - %d' % (int(hex_min, 16),int(hex_max, 16)))

def preprocess(img, datadir, img_size):
    image = Image.open(os.path.join(datadir,img)).convert('L')
    newimage = image.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
    newimage = np.array(newimage)
    return newimage

data = np.zeros(([n_files]+img_size))
labels = np.zeros((n_files, len(template_label)))

marker = 0
for i in os.listdir(datadir):
    if i.find('.png')!=-1:
        filename = i.split('.')[0].split('_')[3:]
        img = preprocess(i, datadir, img_size)
        data[marker,:,:,0] = img
        label = template_label.copy()
        for idx in range(len(label_list)):
            for j in filename:
                if j==label_list[idx]:
                    label[idx]=1
        labels[marker] = label.copy()
    marker+=1

random_permutation = np.random.permutation(n_files)
data = data[random_permutation]
labels = labels[random_permutation]
'''
print(data.shape, labels.shape)
print(labels[0])
plt.imshow(data[0,:,:,0], cmap='gray')
plt.show()
'''
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=img_size))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(template_label), activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpoint = ModelCheckpoint("./models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

for i in range(50):
    print('Epoch ', i)
    model.fit(data, labels, batch_size=32, epochs=1, validation_split=0.15, callbacks=[checkpoint], verbose=1)
    test_idx = np.random.permutation(n_files)[0]
    test_image = data[test_idx:test_idx+1]
    #plt.imshow(test_image[0,:,:,0], cmap='gray')
    #plt.show()
    result = model.predict(test_image)
    print(result[0])
    print('Predicted')
    for j in range(len(result[0])):
        if result[0][j]>=0.5:
            print (label_list[j], chr(int(label_list[j])))
    print('Label')
    for j in range(len(labels[test_idx])):
        if labels[test_idx][j]==1:
            print (label_list[j], chr(int(label_list[j])))

    if i%10==0:
        result = model.predict(data, verbose=1)
        result = (result>0.5)*1
        corr_pred = 0
        for examples_idx in range(n_files):
            if np.sum((result[examples_idx] == labels[examples_idx])*1)==len(template_label):
                corr_pred+=1
        print('Absolute accuracy', corr_pred/np.float(n_files))
        checkpoint = ModelCheckpoint("./models/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

model2 = load_model('test_model')

result = model2.predict(data, verbose=1)
result = (result>0.5)*1
corr_pred = 0
for examples_idx in range(n_files):
    if np.sum((result[examples_idx] == labels[examples_idx])*1)==len(template_label):
        corr_pred+=1
print('Absolute accuracy', corr_pred/np.float(n_files))
