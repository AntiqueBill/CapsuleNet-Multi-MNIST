#coding=utf
from keras import utils
import numpy as np
from keras import backend as K
import argparse
from keras.datasets import mnist
import scipy.io as sio

def load_mnist(args):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    y_train = utils.to_categorical(y_train, 10)
    y_test = utils.to_categorical(y_test, 10)
    
    idx_tr = range(len(x_train))
    np.random.shuffle(idx_tr)
    X_train = np.zeros(x_train.shape)
    X_train0 = np.zeros(x_train.shape)
    X_train1 = np.zeros(x_train.shape)
    for i in xrange(x_train.shape[0]):
        shifti = np.random.randint(-args.max_shift, args.max_shift, size=[2,2])
        x_train_0 = shift_2d(x_train[i,:,:,0], shifti[0], args)
        x_train_1 = shift_2d(x_train[idx_tr[i],:,:,0], shifti[1], args)
        X_train[i] = np.minimum(np.expand_dims(np.add(x_train_0, x_train_1), axis = 2), 255)
        X_train0[i] = np.expand_dims(x_train_0, -1)
        X_train1[i] = np.expand_dims(x_train_1, -1)  
    Y_train1 = np.vstack([y_train.argmax(1), y_train[idx_tr].argmax(1)]).T
    X_train = X_train[Y_train1[:,0] != Y_train1[:,1]] #确保两个数字不一样
    X_train0 = X_train0[Y_train1[:,0] != Y_train1[:,1]]
    X_train1 = X_train1[Y_train1[:,0] != Y_train1[:,1]]
    Y_train1 = Y_train1[Y_train1[:,0] != Y_train1[:,1]]
    Y_train = K.eval(K.one_hot(Y_train1, 10))
    
    idx_te = range(len(x_test))
    np.random.shuffle(idx_te)
    X_test = np.zeros(x_test.shape)
    X_test0 = np.zeros(x_test.shape)
    X_test1 = np.zeros(x_test.shape)
    for i in xrange(x_test.shape[0]):
        shifti = np.random.randint(-args.max_shift, args.max_shift, size=[2,2])
        x_test_0 = shift_2d(x_test[i,:,:,0], shifti[0], args)
        x_test_1 = shift_2d(x_test[idx_te[i],:,:,0], shifti[1], args)
        X_test[i] = np.minimum(np.expand_dims(np.add(x_test_0, x_test_1), axis = 2), 255)
        X_test0[i] = np.expand_dims(x_test_0, -1)
        X_test1[i] = np.expand_dims(x_test_1, -1)   
    Y_test1 = np.vstack([y_test.argmax(1), y_test[idx_te].argmax(1)]).T
    X_test = X_test[Y_test1[:,0] != Y_test1[:,1]] #确保两个数字不一样
    X_test0 = X_test0[Y_test1[:,0] != Y_test1[:,1]]
    X_test1 = X_test1[Y_test1[:,0] != Y_test1[:,1]]
    Y_test1 = Y_test1[Y_test1[:,0] != Y_test1[:,1]]
    Y_test = K.eval(K.one_hot(Y_test1, 10))
    
    X_train /= 255
    X_test /= 255
    X_train0 /=255
    X_train1 /=255
    X_test0 /=255
    X_test1 /=255
    return (X_train, Y_train), (X_test, Y_test), (X_train0, X_train1), (X_test0, X_test1), (Y_train1, Y_test1)

def shift_2d(image, shift, args):
    max_shift = args.max_shift
    max_shift += 1
    padded_image = np.pad(image, max_shift, 'constant')
    rolled_image = np.roll(padded_image, shift[0], axis=0)
    rolled_image = np.roll(rolled_image, shift[1], axis=1)
    shifted_image = rolled_image[max_shift:-max_shift, max_shift:-max_shift]
    return shifted_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-MNIST generation.")
    parser.add_argument('-sd', '--save_data', default='mnist_shifted.mat',
                        help="Name of saved data file")
    parser.add_argument('-m', '--max_shift', default=6, type=int,
                        help="maximum shift of mnist images before adding them together")

    args = parser.parse_args()
    print(args)
    
    
    (x_train, y_train), (x_test, y_test), (x_train0, x_train1), (x_test0, x_test1), (y_train1, y_test1) = load_mnist(args=args)
    y_test = K.eval(K.sum(y_test, -2))
    y_train = K.eval(K.sum(y_train, -2))
    
    print('Saving data:%s' %args.save_data)
    sio.savemat(args.save_data, {'x_train':x_train,'y_train':y_train,
                                 'x_test':x_test,'y_test':y_test,
                                 'x_train0':x_train0,'x_train1':x_train1,
                                 'x_test0':x_test0,'x_test1':x_test1,
                                 'y_train1':y_train1, 'y_test1':y_test1})