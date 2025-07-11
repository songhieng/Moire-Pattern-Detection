#!/usr/bin/env python
from matplotlib import pyplot as plt
import numpy as np
import sys
import argparse
from os import listdir
from os.path import isfile, join
import os
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import Callback
from mCNN import createModel

# constants
WIDTH = 500  # 384
HEIGHT = 375  # 512

def scaleData(inp, minimum, maximum):
    minMaxScaler = preprocessing.MinMaxScaler(copy=True, feature_range=(minimum, maximum))
    inp = inp.reshape(-1, 1)
    inp = minMaxScaler.fit_transform(inp)
    return inp

def readAndScaleImage(f, customStr, trainImagePath,
                      X_LL, X_LH, X_HL, X_HH, X_index, Y,
                      sampleIndex, sampleVal):
    fname = os.path.splitext(f)[0]
    fLL = f.replace(fname, fname + customStr + '_LL').replace('.jpg', '.tiff')
    fLH = f.replace(fname, fname + customStr + '_LH').replace('.jpg', '.tiff')
    fHL = f.replace(fname, fname + customStr + '_HL').replace('.jpg', '.tiff')
    fHH = f.replace(fname, fname + customStr + '_HH').replace('.jpg', '.tiff')

    try:
        imgLL = Image.open(join(trainImagePath, fLL))
        imgLH = Image.open(join(trainImagePath, fLH))
        imgHL = Image.open(join(trainImagePath, fHL))
        imgHH = Image.open(join(trainImagePath, fHH))
    except Exception as e:
        print(f"Error: Couldn't read the file {fname}. Make sure only images are present.")
        print("Exception:", e)
        return None

    imgLL = scaleData(np.array(imgLL), 0, 1).reshape(1, WIDTH * HEIGHT)
    imgLH = scaleData(np.array(imgLH), -1, 1).reshape(1, WIDTH * HEIGHT)
    imgHL = scaleData(np.array(imgHL), -1, 1).reshape(1, WIDTH * HEIGHT)
    imgHH = scaleData(np.array(imgHH), -1, 1).reshape(1, WIDTH * HEIGHT)

    X_LL[sampleIndex, :] = imgLL
    X_LH[sampleIndex, :] = imgLH
    X_HL[sampleIndex, :] = imgHL
    X_HH[sampleIndex, :] = imgHH
    Y[sampleIndex, 0]    = sampleVal
    X_index[sampleIndex, 0] = sampleIndex

    return True

def readImageSet(imageFiles, trainImagePath,
                 X_LL, X_LH, X_HL, X_HH, X_index, Y,
                 sampleIndex, bClass):
    for f in imageFiles:
        ret = readAndScaleImage(f, '', trainImagePath,
                                X_LL, X_LH, X_HL, X_HH, X_index, Y,
                                sampleIndex, bClass)
        if ret:
            sampleIndex += 1
        ret = readAndScaleImage(f, '_180', trainImagePath,
                                X_LL, X_LH, X_HL, X_HH, X_index, Y,
                                sampleIndex, bClass)
        if ret:
            sampleIndex += 1
        ret = readAndScaleImage(f, '_180_FLIP', trainImagePath,
                                X_LL, X_LH, X_HL, X_HH, X_index, Y,
                                sampleIndex, bClass)
        if ret:
            sampleIndex += 1
    return sampleIndex

def readWaveletData(posPath, negPath, posTrainPath, negTrainPath):
    posFiles = [f for f in listdir(posPath) if isfile(join(posPath, f))]
    negFiles = [f for f in listdir(negPath) if isfile(join(negPath, f))]
    posCount = len(posFiles) * 4
    negCount = len(negFiles) * 4
    total   = posCount + negCount

    print(f"positive samples: {posCount}")
    print(f"negative samples: {negCount}")

    X_LL   = np.zeros((total, WIDTH * HEIGHT))
    X_LH   = np.zeros((total, WIDTH * HEIGHT))
    X_HL   = np.zeros((total, WIDTH * HEIGHT))
    X_HH   = np.zeros((total, WIDTH * HEIGHT))
    X_index= np.zeros((total, 1))
    Y      = np.zeros((total, 1))

    idx = 0
    idx = readImageSet(posFiles, posTrainPath, X_LL, X_LH, X_HL, X_HH, X_index, Y, idx, 0)
    print("positive data loaded.")
    idx = readImageSet(negFiles, negTrainPath, X_LL, X_LH, X_HL, X_HH, X_index, Y, idx, 1)
    print("negative data loaded.")
    print(f"Total Samples Loaded: {idx}")

    return X_LL, X_LH, X_HL, X_HH, X_index, Y, total

def splitTrainTestDataForBands(data, train_idx, test_idx):
    train = np.zeros((len(train_idx), WIDTH * HEIGHT))
    test  = np.zeros((len(test_idx),  WIDTH * HEIGHT))
    for i, ind in enumerate(train_idx[:,0]):
        train[i, :] = data[int(ind), :]
    for i, ind in enumerate(test_idx[:,0]):
        test[i, :]  = data[int(ind), :]
    return train, test

def trainTestSplit(X_LL, X_LH, X_HL, X_HH, X_index, Y, totalCount):
    test_frac = 0.1
    X_train_idx, X_test_idx, y_train, y_test = train_test_split(
        X_index, Y, test_size=test_frac, random_state=1, stratify=Y
    )

    X_LL_train, X_LL_test = splitTrainTestDataForBands(X_LL, X_train_idx, X_test_idx)
    X_LH_train, X_LH_test = splitTrainTestDataForBands(X_LH, X_train_idx, X_test_idx)
    X_HL_train, X_HL_test = splitTrainTestDataForBands(X_HL, X_train_idx, X_test_idx)
    X_HH_train, X_HH_test = splitTrainTestDataForBands(X_HH, X_train_idx, X_test_idx)

    num_train = len(y_train)
    # reshape for CNN inputs
    def reshape(x):
        return x.reshape((num_train if x is X_LL_train else totalCount - num_train,
                          HEIGHT, WIDTH, 1))

    X_LL_train = reshape(X_LL_train)
    X_LH_train = reshape(X_LH_train)
    X_HL_train = reshape(X_HL_train)
    X_HH_train = reshape(X_HH_train)
    X_LL_test  = reshape(X_LL_test)
    X_LH_test  = reshape(X_LH_test)
    X_HL_test  = reshape(X_HL_test)
    X_HH_test  = reshape(X_HH_test)

    return (X_LL_train, X_LH_train, X_HL_train, X_HH_train,
            y_train, X_LL_test, X_LH_test, X_HL_test, X_HH_test, y_test)

def trainCNNModel(X_LL_train, X_LH_train, X_HL_train, X_HH_train, y_train,
                  X_LL_test,  X_LH_test,  X_HL_test,  X_HH_test,  y_test,
                  num_epochs):

    batch_size = 32
    num_classes = len(np.unique(y_train))
    Y_train = to_categorical(y_train, num_classes)
    Y_test  = to_categorical(y_test, num_classes)

    checkpoint_folder = 'checkPoint'
    os.makedirs(checkpoint_folder, exist_ok=True)

    class SaveEvery10Epochs(Callback):
        def __init__(self, folder):
            super().__init__()
            self.folder = folder

        def on_epoch_end(self, epoch, logs=None):
            epoch_num = epoch + 1
            if epoch_num % 10 == 0:
                val_loss = logs.get('val_loss')
                fname = f"Weights-{epoch_num:03d}--{val_loss:.5f}.keras"
                path = os.path.join(self.folder, fname)
                self.model.save(path)
                print(f"\n✅ Saved model at epoch {epoch_num} → {path}")

    save_cb = SaveEvery10Epochs(checkpoint_folder)

    model = createModel(HEIGHT, WIDTH, 1, num_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    model.fit(
        [X_LL_train, X_LH_train, X_HL_train, X_HH_train],
        Y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        verbose=1,
        validation_split=0.1,
        callbacks=[save_cb]
    )

    # Optionally save final model
    model.save('moirePattern3CNN_final.keras')

    return model

def evaluate(model, X_LL_test, X_LH_test, X_HL_test, X_HH_test, y_test):
    preds = model.predict([X_LL_test, X_LH_test, X_HL_test, X_HH_test])
    TP = TN = FP = FN = 0
    for i in range(len(y_test)):
        true = int(y_test[i])
        pred = int(np.argmax(preds[i]))
        if true == 0 and pred == 0: TP += 1
        if true == 1 and pred == 1: TN += 1
        if true == 1 and pred == 0: FN += 1
        if true == 0 and pred == 1: FP += 1

    total = TP + TN + FP + FN
    accuracy  = 100 * (TP + TN) / total
    precision = 100 * TP / (TP + FP) if (TP + FP) else 0
    recall    = 100 * TP / (TP + FN) if (TP + FN) else 0

    print("\n\033[1mconfusion matrix (test)\033[0m")
    print(f" true positive:  {TP}")
    print(f" false positive: {FP}")
    print(f" true negative:  {TN}")
    print(f" false negative: {FN}")
    print(f"\n\033[1maccuracy:  \033[0m{accuracy:.4f} %")
    print(f"\033[1mprecision: \033[0m{precision:.4f} %")
    print(f"\033[1mrecall:    \033[0m{recall:.4f} %")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('positiveImages', type=str, help='Dir with positive (Moire) images.')
    parser.add_argument('negativeImages', type=str, help='Dir with negative (Normal) images.')
    parser.add_argument('trainingDataPositive', type=str, help='Dir with transformed positive images.')
    parser.add_argument('trainingDataNegative', type=str, help='Dir with transformed negative images.')
    parser.add_argument('epochs', type=int, help='Number of epochs for training')
    return parser.parse_args(argv)

def main(args):
    X_LL, X_LH, X_HL, X_HH, X_idx, Y, total = readWaveletData(
        args.positiveImages,
        args.negativeImages,
        args.trainingDataPositive,
        args.trainingDataNegative
    )

    (X_LL_tr, X_LH_tr, X_HL_tr, X_HH_tr,
     y_tr, X_LL_te, X_LH_te, X_HL_te, X_HH_te, y_te) = trainTestSplit(
        X_LL, X_LH, X_HL, X_HH, X_idx, Y, total
    )

    model = trainCNNModel(
        X_LL_tr, X_LH_tr, X_HL_tr, X_HH_tr, y_tr,
        X_LL_te, X_LH_te, X_HL_te, X_HH_te, y_te,
        args.epochs
    )

    evaluate(model, X_LL_te, X_LH_te, X_HL_te, X_HH_te, y_te)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
