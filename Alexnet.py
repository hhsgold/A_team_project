
# coding: utf-8

# In[5]:


#현재 dropout 설정 없음/ padding은 0인 상태... 알맞은 상태로 설정을 해야함
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

#데이터셋 설정
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'C:\\Users\\thswl\\PycharmProjects\\junchuri\\train',
    target_size=(224, 224),
    batch_size=35,
    class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./225)

test_generator = train_datagen.flow_from_directory(
    'C:\\Users\\thswl\\PycharmProjects\\junchuri\\test',
    target_size=(224, 224),
    batch_size=35,
    class_mode='categorical')

#알렉스넷 모델 생성
model = Sequential()

#Alexnet - 계층 1 : 11x11 필터를 96개를 사용, strides = 4, 활화화함수 = relu, 
#                   입력 데이터 크기 224x224 , 3x3 크기의 풀리계층 사용
model.add(Conv2D(96, (11,11), padding='same',activation='relu', strides=4, input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))

#Alexnet - 계층 2 : 5X5 필터를 256개 사용 , strides = 1, 활화화함수 = relu, 3x3 크기의 풀리계층 사용
model.add(Conv2D(256,(5,5), padding='same',activation='relu', strides=1))
model.add(MaxPooling2D(pool_size=(3,3),strides=2))

#Alexnet - 계층 3 : 3x3 필터를 384개 사용, strides =1 , 활성화함수 = relu
model.add(Conv2D(384,(3,3), padding='same',activation='relu', strides=1))

#Alexnet - 계층 4 : 3x3 필터를 384개 사용, strides =1 , 활성화함수 = relu
model.add(Conv2D(384,(3,3), padding='same',activation='relu', strides=1))

#Alexnet - 계층 5 : 3x3 필터를 256개 사용, strides =1 , 활성화함수 = relu, 3x3 크기의 풀리계층 사용
model.add(Conv2D(256,(3,3), padding='same',activation='relu',strides=1))
model.add(MaxPooling2D(pool_size=(3,3), strides=2))

#계산을 위해서 1차원 배열로 전환
model.add(Flatten())

#Alexnet - 계층 6 : 4096개의 출력뉴런, 활성화함수 = relu
model.add(Dense(4096, activation='relu'))

#Alexnet - 계층 7 : 4096게의 출력뉴런, 활성화함수 = relu
model.add(Dense(4096, activation='relu'))

#Alexnet - 계층 8 : 1개의 출력뉴런, 활성화함수 = sigmoid
model.add(Dense(2, activation='sigmoid'))

#학습과정 설정 - 손실함수는 크로스엔트로피, 가중치 검색은 아담
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

#Alexnet - 학습하기
model.fit_generator(train_generator, steps_per_epoch=20, epochs=10, validation_data=test_generator, validation_steps=50)




