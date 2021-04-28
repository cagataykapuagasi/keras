# Baseline MLP for MNIST dataset
import math

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import keras.backend as K

# keras için custom image import etme
datagen = ImageDataGenerator(
    rescale=1. / 255)

train_generator = datagen.flow_from_directory(
    'mnist-train',
    target_size=(28, 28),
    batch_size=2000,
    class_mode='categorical',
    color_mode="rgb",
    shuffle=False
)

test_generator = datagen.flow_from_directory(
    'mnist-test',
    target_size=(28, 28),
    batch_size=1000,
    class_mode='categorical',
    color_mode="rgb",
    shuffle=False
)


# custom imageler ve labelları
X_train, y_train = next(train_generator)
X_test, y_test = next(test_generator)

# resmi yeniden çizdirmek için gereken pixel
num_pixels = 2352

# resimler aynı adette float32 tipinde tekrar çizdiriliyor
X_train = X_train.reshape((2000, num_pixels)).astype('float32')
X_test = X_test.reshape((1000, num_pixels)).astype('float32')

# normalize hale getiriliyor
X_train = X_train / 255
X_test = X_test / 255

# Sınıflandırma probleminde toplam sınıf sayısı.
num_classes = y_test.shape[1]


# model fonksiyonu
def baseline_model():
    # model oluşumu
    model = Sequential()
    # hidden layerlar
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))

    # modeli compile etmek
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# fonksiyonu çağır
model = baseline_model()

# modeli eğit
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# öğrenme hızı
print(K.eval(model.optimizer.lr))

# modelden test ve trainler için tahmin iste
predictTrain = model.predict_classes(X_train)
predictTest = model.predict_classes(X_test)


# manuel accuracy hesabı, array ve frekans parametresi
# frekans: her sayıda kaç adet resim olduğu. ör: train için 200 test için 100
# array: train yada test array i
def calculateAcc(array, frequency: int):
    size = len(array)
    total_right_answers = 0
    total_wrong_answers = 0

    # ayrı ayrı her sayı için doğru cevapların tutulduğu obje
    right_answers = {
        "0": {
            "right_answers": 0,
            "wrong_answers": 0,
        },
        "1": {
            "right_answers": 0,
            "wrong_answers": 0,
        },
        "2": {
            "right_answers": 0,
            "wrong_answers": 0,
        },
        "3": {
            "right_answers": 0,
            "wrong_answers": 0,
        },
        "4": {
            "right_answers": 0,
            "wrong_answers": 0,
        },
        "5": {
            "right_answers": 0,
            "wrong_answers": 0,
        },
        "6": {
            "right_answers": 0,
            "wrong_answers": 0,
        },
        "7": {
            "right_answers": 0,
            "wrong_answers": 0,
        },
        "8": {
            "right_answers": 0,
            "wrong_answers": 0,
        },
        "9": {
            "right_answers": 0,
            "wrong_answers": 0,
        }

    }

    # array in boyutu kadar for
    for i in range(0, size):
        # mevcut cevabı bulmak için index / frekans. ör 500 / 200 = 2. yani sayı "2"
        current_answer = i // frequency

        # cevapla array eşleşiyor ise doğru cevap, eşleşmiyor ise yanlış cevap
        if current_answer == array[i]:
            total_right_answers += 1
            right_answers[str(current_answer)]["right_answers"] += 1
        else:
            total_wrong_answers += 1
            right_answers[str(current_answer)]["wrong_answers"] += 1

    # ortalama accuracy hesabı
    acc = str.format('{0:.3f}', total_right_answers / size)

    if frequency == 200:
        print("Train Accuracies: ")
    else:
        print("Test Accuracies: ")

    # accuracy değerlerinin formatlarla gösterilmesi
    print("average acc: ", acc)
    print("'0' acc: ", str.format('{0:.3f}', right_answers["0"]["right_answers"] / frequency))
    print("'1' acc: ", str.format('{0:.3f}', right_answers["1"]["right_answers"] / frequency))
    print("'2' acc: ", str.format('{0:.3f}', right_answers["2"]["right_answers"] / frequency))
    print("'3' acc: ", str.format('{0:.3f}', right_answers["3"]["right_answers"] / frequency))
    print("'4' acc: ", str.format('{0:.3f}', right_answers["4"]["right_answers"] / frequency))
    print("'5' acc: ", str.format('{0:.3f}', right_answers["5"]["right_answers"] / frequency))
    print("'6' acc: ", str.format('{0:.3f}', right_answers["6"]["right_answers"] / frequency))
    print("'7' acc: ", str.format('{0:.3f}', right_answers["7"]["right_answers"] / frequency))
    print("'8' acc: ", str.format('{0:.3f}', right_answers["8"]["right_answers"] / frequency))
    print("'9' acc: ", str.format('{0:.3f}', right_answers["9"]["right_answers"] / frequency))


# fonksiyonun parametrelerle çağırılması
calculateAcc(predictTrain, 200)
calculateAcc(predictTest, 100)
