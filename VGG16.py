
# coding: utf-8

# In[2]:


#현재 dropout 설정 없음/ padding은 0인상태 전 층을 1로 바꿔줘야함
#마지막 계층 Dense는 출력 노드가 1개인지 2개인지 확인 필요
#class_mode를 무엇을 사용해야하는지 모르겠다.... binary vs categorical
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

#데이터셋 생성
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Projects\\keras_talk\\train_set',
    target_size=(224, 224),
    batch_size=35,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./225)

test_generator = train_datagen.flow_from_directory(
    'C:\\Projects\\keras_talk\\test_set',
    target_size=(224, 224),
    batch_size=35,
    class_mode='categorical')

#VGG16모델 생성
model = Sequential()

#VGG16 계층 생성
model.add(Conv2D(64, (3,3), padding='same', activation='relu',strides=1, input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),padding='same',activation='relu',strides=1))

model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Conv2D(128,(3,3),padding='same',activation='relu',strides=1))
model.add(Conv2D(128,(3,3),padding='same',activation='relu',strides=1))

model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Conv2D(256,(3,3),padding='same',activation='relu',strides=1))
model.add(Conv2D(256,(3,3),padding='same',activation='relu',strides=1))
model.add(Conv2D(256,(3,3),padding='same',activation='relu',strides=1))

model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Conv2D(512,(3,3),padding='same',activation='relu',strides=1))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',strides=1))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',strides=1))

model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Conv2D(512,(3,3),padding='same',activation='relu',strides=1))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',strides=1))
model.add(Conv2D(512,(3,3),padding='same',activation='relu',strides=1))

model.add(MaxPooling2D(pool_size=(2,2),strides=2))

model.add(Flatten())

model.add(Dense(4096,activation='relu'))
model.add(Dense(4096,activation='relu'))
model.add(Dense(2,activation='sigmoid'))

#학습과절 설정 - 손실함수는 크로스엔트로피, 가중치 검색은 아담
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

#VGG16 - 학습하기
model.fit_generator(train_generator, steps_per_epoch=20, epochs=10, validation_data=test_generator, validation_steps=50)


