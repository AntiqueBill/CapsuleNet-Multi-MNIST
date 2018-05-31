#coding=utf


import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers import Lambda
import matplotlib.pyplot as plt
import tensorflow as tf
from capsulelayers2 import CapsuleLayer, PrimaryCap, Length, Mask
from keras import callbacks
import argparse
import scipy.io as sio

K.set_image_data_format('channels_last')


def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', 
                          activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9,
                             strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)
    out_caps = Length(name='capsnet')(digitcaps)

    y = layers.Input(shape=(2,))

    masked_by_y = Mask()([digitcaps, y])
    masked_by_y0 =  Lambda(lambda x: x[:,0,:])(masked_by_y)
    masked_by_y1 =  Lambda(lambda x: x[:,1,:])(masked_by_y)

    masked = Mask()(digitcaps)
    masked0 =  Lambda(lambda x: x[:,0,:])(masked)
    masked1 =  Lambda(lambda x: x[:,1,:])(masked)

    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='sigmoid', input_dim=16))
    decoder.add(layers.Dense(1024, activation='sigmoid'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y0), decoder(masked_by_y1)])
    eval_model = models.Model(x, [out_caps, decoder(masked0), decoder(masked1)])
    return train_model, eval_model


def margin_loss(y_true, y_pred, margin = 0.4, downweight = 0.5):
    y_pred = y_pred - 0.5
    positive_cost = y_true * K.cast(
                    K.less(y_pred, margin), 'float32') * K.pow((y_pred - margin), 2)
    negative_cost = (1 - y_true) * K.cast(
                    K.greater(y_pred, -margin), 'float32') * K.pow((y_pred + margin), 2)
    return 0.5 * positive_cost + downweight * 0.5 * negative_cost


def train(model, data, args):
    (x_train, y_train), (x_test, y_test), (x_train0, x_train1), (x_test0, x_test1),(y_train1, y_test1) = data

    checkpoint = callbacks.ModelCheckpoint(args.save_file, monitor='train_capsnet_loss', verbose=1, save_best_only=True, 
                                  save_weights_only=True, mode='auto', period=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse', 'mse'],
                  loss_weights=[1., args.lam_recon, args.lam_recon],
                  metrics={})
    hist = model.fit([x_train, y_train1], [y_train, x_train0, x_train1], batch_size=args.batch_size, epochs=args.epochs,
                     validation_data=[[x_test, y_test1], [y_test, x_test0, x_test1]], callbacks=[checkpoint, lr_decay])
    return hist.history


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--lr', default=0.002, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.5, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=4, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('-sf', '--save_file', default='caps2.h5',
                        help="Name of saved weight file")
    parser.add_argument('-t', '--test', default=0,type=int,
                        help="Test only model")
    parser.add_argument('-l', '--load', default=0,type=int,
                        help="load weight file or not")
    parser.add_argument('-p', '--plot', default=0,type=int,
                        help="plot training loss after finished if plot==1")
    parser.add_argument('-d', '--dataset', default='mnist_shifted.mat',
                        help="name of dataset that needs loading")
    args = parser.parse_args()
    print(args)
    
    K.set_image_data_format('channels_last')
    
    data = sio.loadmat(args.dataset, appendmat=False)
    for i in data:
        locals()[i] = data[i]
    del data
    del i

    model, eval_model = CapsNet(input_shape=x_train.shape[1:], n_class=10, routings=args.routings)
        
        
    if args.test == 0:    
        if args.load == 1:
            model.load_weights(args.save_file)
            print('Loading %s' %args.save_file)
        history = train(model=model, data=(
                        (x_train, y_train), (x_test, y_test), (x_train0, x_train1), (x_test0, x_test1)
                        , (y_train1, y_test1)), args=args)
        if args.plotplot == 1:    
            train_loss = np.array(history['loss'])
            val_loss = np.array(history['val_loss'])
            plt.plot(np.arange(0, args.epochs, 1),train_loss,label="train_loss",color="red",linewidth=1.5)
            plt.plot(np.arange(0, args.epochs, 1),val_loss,label="val_loss",color="blue",linewidth=1.5)
            plt.legend()
            plt.show()
            plt.savefig('loss.jpg')
    else:
        model.load_weights(args.save_file)
        print('Loading %s' %args.save_file)
      
    print('-'*30 + 'Begin: test' + '-'*30)
    y_pred_tr, x_recon0_tr, x_recon1_tr = eval_model.predict(x_train, batch_size=args.batch_size)
    _, y_pred1_tr = tf.nn.top_k(y_pred_tr, 2)
    y_pred1_tr = K.eval(y_pred1_tr)
    y_pred1_tr.sort(axis = 1)
    y_train1.sort(axis = 1)
    y_pred1_tr = np.reshape(y_pred1_tr, np.prod(y_pred1_tr.shape))
    y_train1 = np.reshape(y_train1, np.prod(y_train1.shape))
    print('Train acc:', np.sum(y_pred1_tr == y_train1)/np.float(y_train1.shape[0]))
    
    
    y_pred, x_recon0, x_recon1 = eval_model.predict(x_test, batch_size=args.batch_size)
    _, y_pred1 = tf.nn.top_k(y_pred, 2)
    y_pred1 = K.eval(y_pred1)
    y_pred1.sort(axis = 1)
    y_test1.sort(axis = 1)
    y_pred1 = np.reshape(y_pred1, np.prod(y_pred1.shape))
    y_test1 = np.reshape(y_test1, np.prod(y_test1.shape))
    print('Test acc:', np.sum(y_pred1 == y_test1)/np.float(y_test1.shape[0]))
    print('-' * 30 + 'End: test' + '-' * 30)   
'''
    from keras.utils import plot_model
    plot_model(model, to_file='model.png',show_shapes = True)
    plot_model(eval_model, to_file='eval_model.png',show_shapes = True)
'''
