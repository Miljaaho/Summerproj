# Filters = 20
# Epochs = 10
# Convolutions = 4
# Dropouts = 4
# Fraction Dropout = 0.55
# Matrix size = 2*3x3 + 2*5x5
# Optimizer = adam
# Activation func = SReLU
#same as 12 but trained on different data

from __future__ import division
import glob

import numpy as np
import tables
import os

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import pylab as plt

from keras.layers import Input, Dense, Convolution2D, merge, Flatten, Dropout
from keras.layers.advanced_activations import SReLU
from keras.regularizers import l2
from keras.models import Model
from keras.callbacks import ModelCheckpoint

from keras.utils.visualize_util import model_to_dot
import pygraphviz as pgv

from create_features import PAD_WIDTH, keys_h5_matrix, keys_h5_vec

#matrix size 2
PAD_WIDTH2 = 3
#file name
Fname = os.path.splitext(__file__)[0]

#Num of filters used
N_FILTS = 20

#Num of times you send the data through the network
N_EPOCHS = 10

#batch size?
B_SIZE = 512

#Fraction of dropouts
DROPOUT = 0.55

DROPOUT_INP = 0.2
N_DIM_OUTPUT = 1

L1_REG = 1e-5
L2_REG = 1e-5

# Weight regularizer that applies a penalty to layers. Why not use b/activity?
#REGULAR = dict(W_regularizer=l1l2(L1_REG, L2_REG), activity_regularizer=activity_l1l2(L1_REG, L2_REG))
REGULAR = dict(W_regularizer=(l2(L2_REG)))

#names for output files, can be optimized
prefix = 'macro_model_reg_val_{}_ep_{}_batch_{}'.format(Fname, N_EPOCHS, B_SIZE)


if __name__ == '__main__':

	
	dir= os.path.realpath(Fname)
	if not os.path.exists(dir):
   		os.makedirs(dir)
	else:
		raise Exception('Folder already exists')

	#function that adds the non-linearity function of x with a certain N (
	def add_dense(N, x, dropout=True):
	    x = Dense(N, activation='linear', **REGULAR)(x)
	    x = SReLU()(x)

	    if dropout:
		x = Dropout(DROPOUT)(x)
	    return x

#Function that creates the output model
	def get_model(draw=True):

	    # matrixes: [mi, nmi, h]
	    input_matrices = [Input(shape=(1, PAD_WIDTH, PAD_WIDTH), name=key) for key in keys_h5_matrix]

	    # vectors: [pssm0_i, pssm1_i, pssm0_j, pssm1_j]
	    input_vectors = [Input(shape=(21,), name=key) for key in keys_h5_vec]
	  
	    input_floats = [Input(shape=(1,), name='seq_dist')]

	    # add all inputs together: [mi, nmi, h, pssm0_i, pssm1_i, pssm0_j, pssm1_j, seq_dist]
	    inputs = input_matrices + input_vectors + input_floats
	 
	    flat_layers = []
    
	    # for mi, nmi & h
	    for inp in input_matrices:
		# does a dropout, 2 convolutions followed by an activation function
		x = Dropout(DROPOUT_INP)(inp)
		x = Convolution2D(N_FILTS, PAD_WIDTH2, PAD_WIDTH2, border_mode='same', **REGULAR)(x)	  
		x = SReLU()(x)
		x = Dropout(DROPOUT)(x)
		x = Convolution2D(N_FILTS, PAD_WIDTH2, PAD_WIDTH2, border_mode='same', **REGULAR)(x)	  
		x = SReLU()(x)
		x = Dropout(DROPOUT)(x)
		x = Convolution2D(N_FILTS, PAD_WIDTH, PAD_WIDTH, border_mode='same', **REGULAR)(x)	
		x = SReLU()(x)
		x = Dropout(DROPOUT)(x)
		x = Convolution2D(N_FILTS, PAD_WIDTH, PAD_WIDTH, border_mode='valid', **REGULAR)(x)
		x = SReLU()(x)
       

	
		#pooling?
		x = Flatten()(x)
		x = Dropout(DROPOUT)(x)
		x = add_dense(20, x)
		x = Dense(50)(x)
		x = SReLU()(x)
		#by the end of the loop: flat_layers = [Elemwise{add,no_inplace}.0, Elemwise{add,no_inplace}.0, Elemwise{add,no_inplace}.0]
		flat_layers.append(x)


	    vec_layers = []
	    for inp in input_vectors:
		x = Dropout(DROPOUT_INP)(inp)
		x = add_dense(20, x)
		x = add_dense(10, x, dropout=False)
		vec_layers.append(x)
	    #first and third
	    x1 = merge(vec_layers[::2], mode='sum')
	    #first and fourth
	    x2 = merge(vec_layers[::3], mode='sum')
	    #second and third
	    x3 = merge(vec_layers[1:3], mode='sum')
	    #second and fourth
	    x4 = merge(vec_layers[1::2], mode='sum')

	    for x in x1, x2, x3, x4:
		x = Dropout(DROPOUT)(x)
		x = add_dense(10, x, False)
		flat_layers.append(x)
	    
	    flat_layers.extend(input_floats)
	    x = merge(flat_layers, mode='concat')

	    x = Dropout(DROPOUT)(x)
	    x = add_dense(512, x)
	    x = add_dense(256, x)
	    x = add_dense(64, x)
	    if N_DIM_OUTPUT == 1:
		predictions = Dense(1, activation='sigmoid')(x)
	    else:
		predictions = Dense(N_DIM_OUTPUT, activation='softmax')(x)
	   
	    model = Model(input=inputs, output=predictions)
	    loss = 'binary_crossentropy' if N_DIM_OUTPUT == 1 else 'categorical_crossentropy'

	    #weight_file = max(glob.glob('weights/macro_model_reg*.hdf5'))
	    #print 'Loading weights from', weight_file
	    #model.load_weights(weight_file)

	    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

	    if draw:
		model_to_dot(model, True).write_raw('majestic_graph.dot')
		G = pgv.AGraph('majestic_graph.dot')
		G.layout(prog='dot')
		G.draw(Fname + '/' + prefix + 'macro_network.png')

		model_to_dot(model, False).write_raw('majestic_graph.dot')
		G = pgv.AGraph('majestic_graph.dot')
		G.layout(prog='dot')
		G.draw(Fname + '/' + prefix + 'macro_network_condensed.png')
	    return model

	class Data(object):
	    def __init__(self, h5file):
		self.h5 = tables.open_file(h5file)
	
	   
	    #loads the training data
	    def load_training(self):
		table = self.h5.root.train
		features = [np.reshape(getattr(table.cols, k)[:], (-1, 1, PAD_WIDTH, PAD_WIDTH)) for k in keys_h5_matrix]
		features.extend([getattr(table.cols, k)[:] for k in keys_h5_vec])
		features.append(table.cols.seq_sep[:])

		labels = table.cols.is_contact[:]

		mask = np.zeros_like(labels, dtype=np.bool)
		mask1 = np.zeros(np.sum(labels==0), dtype=np.bool)

		mask[labels == 1] = True
		mask1[:np.sum(labels == 1)] = True
		np.random.shuffle(mask1)
		mask[labels == 0] = mask1

		features = [f[mask, ...] for f in features]
		labels = labels[mask]

		if N_DIM_OUTPUT == 2:
		    labels_ = np.zeros(labels.shape + (2,))
		    labels_[labels == 0, 0] = 1
		    labels_[labels == 1, 1] = 1
		    labels = labels_

		order = np.arange(labels.shape[0])
		np.random.shuffle(order)
		features = [feat[order, ...] for feat in features]
		labels = labels[order, ...]
		return features, labels
	    
	    #loads the testing data
	    def load_test(self):
		table = self.h5.root.test
		
		features = [
		    np.reshape(getattr(table.cols, k)[:], (-1, 1, PAD_WIDTH, PAD_WIDTH))
		    for k in keys_h5_matrix]
		features.extend([getattr(table.cols, k)[:] for k in keys_h5_vec])
		features.append(table.cols.seq_sep[:])
		labels = table.cols.is_contact[:]

		if N_DIM_OUTPUT == 2:
		    labels_ = np.zeros(labels.shape + (2,))
		    labels_[labels == 0, 0] = 1
		    labels_[labels == 1, 1] = 1
		    return features, labels_
		return features, labels

	print 'Creating model'
	model = get_model()


	print 'Reading in data'
	data = Data('svmcon_dataset_filtered.h5')
	print data.h5.root.train
	features, labels = data.load_training()
	features_test, labels_test = data.load_test()

	print 'Begin training'
	cbacks = [ModelCheckpoint('weights/' + prefix + 'weights.{epoch:04d}-{loss:.4f}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')]

	history = model.fit(features, labels, nb_epoch=N_EPOCHS, batch_size=B_SIZE, shuffle=True, validation_data=(features_test, labels_test), callbacks=cbacks)

	print 'Saving'
	with open(Fname + '/' + prefix + 'model.json', 'w') as outf:
	    outf.write(model.to_json())
	model.save_weights(Fname + '/' + prefix + '.h5', overwrite=True)
	try:
	    with open(Fname + '/' + prefix + 'history.cP', 'w') as outf:
		import cPickle
		cPickle.dump(history.history, outf)
	except Exception as e:	
	    print e

	plt.figure()
	epochs = np.arange(0, len(history.history['loss'])) + 1
	plt.plot(epochs, history.history['loss'], label='Training')
	plt.plot(epochs, history.history['val_loss'], label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('Loss function')
	plt.legend(loc=0)
	plt.savefig(Fname + '/' + prefix + 'loss_epoch.png')
	plt.semilogx()
	plt.savefig(Fname + '/' + prefix + 'loss_epoch_log.png')
	plt.close()

	plt.plot(epochs, history.history['acc'], label='Training')
	plt.plot(epochs, history.history['val_acc'], label='Test')
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy')
	plt.legend(loc=0)
	plt.savefig(Fname + '/' + prefix + 'acc_epoch.png')
	plt.semilogx()
	plt.savefig(Fname + '/' + prefix + 'acc_epoch_log.png')
	plt.close()

	print
	print
	print 'Model evaluation:'
	features, labels = data.load_test()
	pos_labels = labels == 1

	y_pred = model.predict(features, verbose=1)[:, 0]
	print 'Max/min:', y_pred.max(), y_pred.min()

	threshold = 0.4
	y_pred_ = y_pred > threshold

	print 'Threshold:', threshold
	print 'PPV:', np.logical_and(y_pred_ == 1, pos_labels).sum() / np.sum(y_pred_ == 1)
	print
	print 'Threshold: 0.5'
	y_pred_ = y_pred > 0.5
	print 'PPV:', np.logical_and(y_pred_ == 1, pos_labels).sum() / np.sum(y_pred_ == 1)

	plt.figure()
	plt.hist(y_pred, bins=500, histtype='step', color='k')
	plt.hist(y_pred[pos_labels == 0], bins=500, histtype='step', color='r')
	plt.hist(y_pred[pos_labels == 1], bins=500, histtype='step', color='g')
	plt.savefig(Fname + '/' + prefix + 'values_hist.png')

	plt.figure()
	plt.hist(y_pred, bins=500, histtype='step', color='k', normed=True)
	plt.hist(y_pred[pos_labels == 0], bins=500, histtype='step', color='r', normed=True)
	plt.hist(y_pred[pos_labels == 1], bins=500, histtype='step', color='g', normed=True)
	plt.savefig(Fname + '/' + prefix + 'values_hist_normed.png')

	plt.figure('roc')
	fpr, tpr, thresholds = roc_curve(pos_labels, y_pred, drop_intermediate=False)
	auc = roc_auc_score(pos_labels, y_pred)
	ppv, recall, thresholds_ppv = precision_recall_curve(pos_labels, y_pred)

	plt.plot(fpr, tpr)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.text(0.2, 0.6, 'AUC={:.2f}'.format(auc))
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve')
	plt.savefig(Fname + '/' + prefix + 'roc_curve.png')

	plt.figure()
	plt.plot(thresholds, fpr, label='False positive rate')
	plt.plot(thresholds, tpr, label='True Positive Rate')
	plt.plot(thresholds_ppv, ppv[:-1], label='PPV')
	plt.plot(thresholds_ppv, recall[:-1], label='Recall')
	plt.legend(loc=0)
	plt.xlabel('Score threshold')
	plt.savefig(Fname + '/' + prefix + 'score_fpr.png')

	plt.figure('prec')
	plt.plot(ppv, recall)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('Precision')
	plt.ylabel('Recall')
	plt.savefig(Fname + '/' + prefix + 'pr_curve.png')

