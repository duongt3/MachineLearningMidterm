import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense
from keras.layers import Input
from keras.utils import np_utils
from keras.datasets import cifar10

epochs = 100

# Get the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Get the data ready
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Create imput
input_img = Input(shape = (32, 32, 3))


# Create Volumes for the Inception module
volume_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)

volume_2 = Conv2D(96, (1,1), padding='same', activation='relu')(input_img)
volume_2 = Conv2D(128, (3,3), padding='same', activation='relu')(volume_2)

volume_3 = Conv2D(16, (1,1), padding='same', activation='relu')(input_img)
volume_3 = Conv2D(32, (5,5), padding='same', activation='relu')(volume_3)

volume_4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
volume_4 = Conv2D(32, (1,1), padding='same', activation='relu')(volume_4)

# Concatenate all volumes of the Inception module
inception_module = keras.layers.concatenate([volume_1, volume_2, volume_3,
                                             volume_4], axis = 3)


# Create Volumes for the Inception module
volume_5 = Conv2D(128, (1,1), padding='same', activation='relu')(inception_module)

volume_6 = Conv2D(128, (1,1), padding='same', activation='relu')(inception_module)
volume_6 = Conv2D(192, (3,3), padding='same', activation='relu')(volume_6)

volume_7 = Conv2D(32, (1,1), padding='same', activation='relu')(inception_module)
volume_7 = Conv2D(96, (5,5), padding='same', activation='relu')(volume_7)

volume_8 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inception_module)
volume_8 = Conv2D(64, (1,1), padding='same', activation='relu')(volume_8)

# Concatenate all volumes of the Inception module
inception_module1 = keras.layers.concatenate([volume_5, volume_6, volume_7,
                                             volume_8], axis = 3)

# Create Volumes for the Inception module
volume_9 = Conv2D(192, (1,1), padding='same', activation='relu')(inception_module1)

volume_10 = Conv2D(96, (1,1), padding='same', activation='relu')(inception_module1)
volume_10 = Conv2D(208, (3,3), padding='same', activation='relu')(volume_10)

volume_11 = Conv2D(16, (1,1), padding='same', activation='relu')(inception_module1)
volume_11 = Conv2D(48, (5,5), padding='same', activation='relu')(volume_11)

volume_12 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inception_module1)
volume_12 = Conv2D(64, (1,1), padding='same', activation='relu')(volume_12)

# Concatenate all volumes of the Inception module
inception_module2 = keras.layers.concatenate([volume_9, volume_10, volume_11,
                                             volume_12], axis = 3)

# Create Volumes for the Inception module
volume_13 = Conv2D(160, (1,1), padding='same', activation='relu')(inception_module2)

volume_14 = Conv2D(112, (1,1), padding='same', activation='relu')(inception_module2)
volume_14 = Conv2D(224, (3,3), padding='same', activation='relu')(volume_14)

volume_15 = Conv2D(24, (1,1), padding='same', activation='relu')(inception_module2)
volume_15 = Conv2D(64, (5,5), padding='same', activation='relu')(volume_15)

volume_16 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inception_module2)
volume_16 = Conv2D(64, (1,1), padding='same', activation='relu')(volume_16)

# Concatenate all volumes of the Inception module
inception_module3 = keras.layers.concatenate([volume_13, volume_14, volume_15,
                                             volume_16], axis = 3)

# Create Volumes for the Inception module
volume_17 = Conv2D(112, (1,1), padding='same', activation='relu')(inception_module3)

volume_18 = Conv2D(144, (1,1), padding='same', activation='relu')(inception_module3)
volume_18 = Conv2D(288, (3,3), padding='same', activation='relu')(volume_18)

volume_19 = Conv2D(32, (1,1), padding='same', activation='relu')(inception_module3)
volume_19 = Conv2D(64, (5,5), padding='same', activation='relu')(volume_19)

volume_20 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inception_module3)
volume_20 = Conv2D(64, (1,1), padding='same', activation='relu')(volume_20)

# Concatenate all volumes of the Inception module
inception_module4 = keras.layers.concatenate([volume_17, volume_18, volume_19,
                                             volume_20], axis = 3)

# Create Volumes for the Inception module
volume_21 = Conv2D(256, (1,1), padding='same', activation='relu')(inception_module4)

volume_22 = Conv2D(160, (1,1), padding='same', activation='relu')(inception_module4)
volume_22 = Conv2D(320, (3,3), padding='same', activation='relu')(volume_10)

volume_23 = Conv2D(32, (1,1), padding='same', activation='relu')(inception_module4)
volume_23 = Conv2D(128, (5,5), padding='same', activation='relu')(volume_11)

volume_24 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inception_module4)
volume_24 = Conv2D(128, (1,1), padding='same', activation='relu')(volume_12)

# Concatenate all volumes of the Inception module
inception_module5 = keras.layers.concatenate([volume_21, volume_22, volume_23,
                                             volume_24], axis = 3)

# Create Volumes for the Inception module
volume_25 = Conv2D(256, (1,1), padding='same', activation='relu')(inception_module5)

volume_26 = Conv2D(160, (1,1), padding='same', activation='relu')(inception_module5)
volume_26 = Conv2D(320, (3,3), padding='same', activation='relu')(volume_26)

volume_27 = Conv2D(32, (1,1), padding='same', activation='relu')(inception_module5)
volume_27 = Conv2D(128, (5,5), padding='same', activation='relu')(volume_27)

volume_28 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inception_module5)
volume_28 = Conv2D(128, (1,1), padding='same', activation='relu')(volume_28)

# Concatenate all volumes of the Inception module
inception_module6 = keras.layers.concatenate([volume_25, volume_26, volume_27,
                                             volume_28], axis = 3)

# Create Volumes for the Inception module
volume_29 = Conv2D(256, (1,1), padding='same', activation='relu')(inception_module6)

volume_30 = Conv2D(160, (1,1), padding='same', activation='relu')(inception_module6)
volume_30 = Conv2D(320, (3,3), padding='same', activation='relu')(volume_30)

volume_31 = Conv2D(32, (1,1), padding='same', activation='relu')(inception_module6)
volume_31 = Conv2D(128, (5,5), padding='same', activation='relu')(volume_31)

volume_32 = MaxPooling2D((3,3), strides=(1,1), padding='same')(inception_module6)
volume_32 = Conv2D(128, (1,1), padding='same', activation='relu')(volume_32)

# Concatenate all volumes of the Inception module
inception_module7 = keras.layers.concatenate([volume_29, volume_30, volume_31,
                                             volume_32], axis = 3)

x = Dropout(0.4)(inception_module7)
output = Flatten()(x)
out    = Dense(10, activation='softmax')(output)


model = Model(inputs = input_img, outputs = out)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128)


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
