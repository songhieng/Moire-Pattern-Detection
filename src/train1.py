#!/usr/bin/env python
import sys
import argparse
import os
from os import listdir
from os.path import isfile, join

import numpy as np
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.callbacks import Callback
from mCNN import createModel

# constants
WIDTH  = 500
HEIGHT = 375

def scaleData(inp, minimum, maximum):
    scaler = preprocessing.MinMaxScaler(feature_range=(minimum, maximum))
    arr = inp.reshape(-1, 1)
    return scaler.fit_transform(arr)

def readAndScaleImage(fname, suffix, srcDir, 
                      X_LL, X_LH, X_HL, X_HH, X_idx, Y, idx, label):
    base = os.path.splitext(fname)[0]
    names = {
        'LL': fname.replace(base, base + suffix + '_LL').replace('.jpg', '.tiff'),
        'LH': fname.replace(base, base + suffix + '_LH').replace('.jpg', '.tiff'),
        'HL': fname.replace(base, base + suffix + '_HL').replace('.jpg', '.tiff'),
        'HH': fname.replace(base, base + suffix + '_HH').replace('.jpg', '.tiff'),
    }
    try:
        imgs = {k: np.array(Image.open(join(srcDir, n))) for k, n in names.items()}
    except Exception as e:
        print(f"Error reading {base}{suffix}: {e}")
        return False

    # scale and flatten
    X_LL[idx] = scaleData(imgs['LL'],  0,  1).reshape(-1)
    X_LH[idx] = scaleData(imgs['LH'], -1,  1).reshape(-1)
    X_HL[idx] = scaleData(imgs['HL'], -1,  1).reshape(-1)
    X_HH[idx] = scaleData(imgs['HH'], -1,  1).reshape(-1)
    Y[idx, 0]    = label
    X_idx[idx,0] = idx
    return True

def readWaveletData(posDir, negDir, posTDir, negTDir):
    posFiles = [f for f in listdir(posDir) if isfile(join(posDir, f))]
    negFiles = [f for f in listdir(negDir) if isfile(join(negDir, f))]
    posCount = len(posFiles) * 3  # each file yields 3 variants
    negCount = len(negFiles) * 3
    total    = posCount + negCount

    X_LL    = np.zeros((total, WIDTH * HEIGHT))
    X_LH    = np.zeros((total, WIDTH * HEIGHT))
    X_HL    = np.zeros((total, WIDTH * HEIGHT))
    X_HH    = np.zeros((total, WIDTH * HEIGHT))
    X_idx   = np.zeros((total, 1))
    Y       = np.zeros((total, 1))

    idx = 0
    for f in posFiles:
        for suffix in ['', '_180', '_180_FLIP']:
            if readAndScaleImage(f, suffix, posTDir, X_LL, X_LH, X_HL, X_HH, X_idx, Y, idx, 0):
                idx += 1
    for f in negFiles:
        for suffix in ['', '_180', '_180_FLIP']:
            if readAndScaleImage(f, suffix, negTDir, X_LL, X_LH, X_HL, X_HH, X_idx, Y, idx, 1):
                idx += 1

    print(f"Loaded {idx} samples ({posCount} positive, {negCount} negative)")
    return X_LL, X_LH, X_HL, X_HH, X_idx, Y, idx

def trainTestSplit(X_LL, X_LH, X_HL, X_HH, X_idx, Y, totalCount):
    # split indices & labels
    X_tr_idx, X_te_idx, y_tr, y_te = train_test_split(
        X_idx, Y, test_size=0.1, random_state=1, stratify=Y
    )
    tr_idx = X_tr_idx.flatten().astype(int)
    te_idx = X_te_idx.flatten().astype(int)

    # gather data
    X_LL_tr = X_LL[tr_idx];  X_LL_te = X_LL[te_idx]
    X_LH_tr = X_LH[tr_idx];  X_LH_te = X_LH[te_idx]
    X_HL_tr = X_HL[tr_idx];  X_HL_te = X_HL[te_idx]
    X_HH_tr = X_HH[tr_idx];  X_HH_te = X_HH[te_idx]

    # reshape using actual row counts
    n_tr = len(y_tr)
    n_te = len(y_te)
    X_LL_tr = X_LL_tr.reshape((n_tr, HEIGHT, WIDTH, 1))
    X_LH_tr = X_LH_tr.reshape((n_tr, HEIGHT, WIDTH, 1))
    X_HL_tr = X_HL_tr.reshape((n_tr, HEIGHT, WIDTH, 1))
    X_HH_tr = X_HH_tr.reshape((n_tr, HEIGHT, WIDTH, 1))
    X_LL_te = X_LL_te.reshape((n_te, HEIGHT, WIDTH, 1))
    X_LH_te = X_LH_te.reshape((n_te, HEIGHT, WIDTH, 1))
    X_HL_te = X_HL_te.reshape((n_te, HEIGHT, WIDTH, 1))
    X_HH_te = X_HH_te.reshape((n_te, HEIGHT, WIDTH, 1))

    return (X_LL_tr, X_LH_tr, X_HL_tr, X_HH_tr, y_tr,
            X_LL_te, X_LH_te, X_HL_te, X_HH_te, y_te)

def trainCNNModel(X_LL_tr, X_LH_tr, X_HL_tr, X_HH_tr, y_tr,
                  X_LL_te, X_LH_te, X_HL_te, X_HH_te, y_te,
                  epochs):

    batch_size = 32
    num_classes = len(np.unique(y_tr))
    Y_tr = to_categorical(y_tr, num_classes)
    Y_te = to_categorical(y_te, num_classes)

    # checkpoint folder
    ckpt = 'checkPoint'
    os.makedirs(ckpt, exist_ok=True)

    class SaveEvery10(Callback):
        def __init__(self, folder): super().__init__(); self.folder = folder
        def on_epoch_end(self, epoch, logs=None):
            e = epoch + 1
            if e % 10 == 0:
                loss = logs.get('val_loss')
                fn = f"Weights-{e:03d}--{loss:.5f}.keras"
                self.model.save(join(self.folder, fn))
                print(f"\nSaved epoch {e} → {fn}")

    cb = SaveEvery10(ckpt)
    model = createModel(HEIGHT, WIDTH, depth=1, num_classes=num_classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(
        [X_LL_tr, X_LH_tr, X_HL_tr, X_HH_tr], Y_tr,
        batch_size=batch_size, epochs=epochs,
        validation_split=0.1, callbacks=[cb], verbose=1
    )
    model.save('moirePattern3CNN_final.keras')
    return model

def evaluate(model, X_LL_te, X_LH_te, X_HL_te, X_HH_te, y_te):
    preds = model.predict([X_LL_te, X_LH_te, X_HL_te, X_HH_te])
    TP = TN = FP = FN = 0
    for i, true in enumerate(y_te.flatten()):
        pred = preds[i].argmax()
        if true == 0:
            if pred == 0: TP += 1
            else: FP += 1
        else:
            if pred == 1: TN += 1
            else: FN += 1
    total = TP + TN + FP + FN
    print(f"Accuracy:  {100*(TP+TN)/total:.2f}%")
    print(f"Precision: {100*TP/(TP+FP):.2f}%")
    print(f"Recall:    {100*TP/(TP+FN):.2f}%")

def parse_arguments(argv):
    p = argparse.ArgumentParser()
    p.add_argument('positiveImages', help='Dir with Moiré images.')
    p.add_argument('negativeImages', help='Dir with normal images.')
    p.add_argument('trainingDataPositive', help='Dir with augmented Moiré.')
    p.add_argument('trainingDataNegative', help='Dir with augmented normal.')
    p.add_argument('epochs', type=int, help='Number of epochs.')
    return p.parse_args(argv)

def main(args):
    X_LL, X_LH, X_HL, X_HH, X_idx, Y, total = readWaveletData(
        args.positiveImages, args.negativeImages,
        args.trainingDataPositive, args.trainingDataNegative
    )

    (X_LL_tr, X_LH_tr, X_HL_tr, X_HH_tr, y_tr,
     X_LL_te, X_LH_te, X_HL_te, X_HH_te, y_te) = trainTestSplit(
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
