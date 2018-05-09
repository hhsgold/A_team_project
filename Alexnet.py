
# coding: utf-8

# In[42]:


'''
풀 커넥션 부분에 드롭아웃 설정했음 (0.5)의 값으로
no padding 은 padding 값을 valid로 주어야한다
마지막 계층 Dense는 출력 노드가 1개로 설정
class_mode는 binary를 사용했다 : 분류할 클래스는 2개 뿐이라서
만일 데이테 셋을 100개 정도만 하면 test 값이 일정한 값으로 고정이 되어버림....
'''

#dropout 위치?
#padding은 0인 상태... 알맞은 상태로 수정필요...
#모델 저장해서 테스트 일 때만 학습된것을 불러와서 사용하는것이 가능한지?
#val_acc를 이용하여 현재의 이미지 상태 호출(함수로 구현)
#데이터 셋 설정시 매개변수 설정을 하면 몇개나 만들어 지는지?
#내가 만든 모델이 정확한 값으로 가는지 의문점.... 강아지 고양이 데이터 셋으로 학인필요함

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import matplotlib.pyplot as plt


#데이터셋 설정
train_datagen = ImageDataGenerator(rescale=1./255,rotation_range = 90,width_shift_range=0.1,
                                   height_shift_range =0.1, zoom_range=0.2, horizontal_flip =True, vertical_flip = True)

train_generator = train_datagen.flow_from_directory(
    'C:\\Projects\\keras_talk\\train1_set',
    target_size=(224, 224),
    batch_size=4,
    class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./225, rotation_range =90, width_shift_range=0.1,
                                 height_shift_range =0.1, zoom_range=0.2, horizontal_flip= True, vertical_flip=True)

test_generator = train_datagen.flow_from_directory(
    'C:\\Projects\\keras_talk\\test1_set',
    target_size=(224, 224),
    batch_size=4,
    class_mode='binary')

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
model.add(Dropout(0.5))

#Alexnet - 계층 7 : 4096게의 출력뉴런, 활성화함수 = relu
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

#Alexnet - 계층 8 : 1개의 출력뉴런, 활성화함수 = sigmoid
model.add(Dense(1, activation='sigmoid'))

#학습과정 설정 - 손실함수는 크로스엔트로피, 가중치 검색은 아담
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

model.summary()

#Alexnet - 학습하기
hist = model.fit_generator(train_generator, steps_per_epoch=2, epochs=6, validation_data=test_generator, validation_steps=1)

#Alexnet - 그래프 그리기
fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'],'y',label='train loss')
loss_ax.plot(hist.history['val_loss'],'r',label = 'val loss')

acc_ax.plot(hist.history['acc'],'b',label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
loss_ax.legend(loc='lower left')

plt.show()

#모델 저장하기
#model.save('Alexnet.h5')

#모델 평가하기
'''
test 할 이미지를 다시 imagedatagenerator를 이용하여 평가를 한다
이때의 함수는 evaluate_generator() 이용
'''

#모델 사용하기
'''
함수를 이용하여 구축
if를 이용한 val_acc를 판단
'''

