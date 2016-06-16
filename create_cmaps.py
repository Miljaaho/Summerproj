from __future__ import division
import numpy as np
import tables
import os

import matplotlib.pyplot as plt

from keras.models import model_from_json
from create_features import PAD_WIDTH, keys_h5_matrix, keys_h5_vec
from neural_network15 import N_FILTS, N_EPOCHS, B_SIZE, Fname

FRAC = 2
onlyName= os.path.basename(os.path.normpath(Fname))
print onlyName
prefix = 'macro_model_reg_val_{}_ep_{}_batch_{}'.format(onlyName, N_EPOCHS, B_SIZE)


def get_model():
    model = model_from_json(open(onlyName + '/' + prefix + 'model.json').read())
    model.load_weights(onlyName + '/' + prefix + '.h5')
    model.compile(optimizer='adagrad', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #model.predict_function = None
    #model._function_kwargs = dict()
    return model

model = get_model()

h5 = tables.open_file('svmcon_dataset_verify2.h5')
test_attr = [t for t in dir(h5.root) if t.startswith('test_')]
train_attr = [t for t in dir(h5.root) if t.startswith('train_')]
print len(test_attr)

ppvs = []
for prot in test_attr:
    print '******', prot, '*******'
    table = getattr(h5.root, prot)
    features = [np.reshape(getattr(table.cols, k)[:],
                           (-1, 1, PAD_WIDTH, PAD_WIDTH)) for
                k in keys_h5_matrix]
    features.extend([getattr(table.cols, k)[:] for k in keys_h5_vec])
    features.append(table.cols.seq_sep[:])
    labels = table.cols.is_contact[:]

    predictions = model.predict(features, verbose=True)[:, 0]

    index_i = table.cols.i[:]
    index_j = table.cols.j[:]
    dmap = table.cols.dist[:]
    N = max(index_i.max(), index_j.max()) + 1
    cmap = np.full((N, N), np.nan)

    num = N // FRAC
    index = len(predictions) - num
    threshold = np.partition(predictions, index)[index]
    #threshold = 0.9
    #num = np.sum(predictions > threshold)
    true_preds = 0
    for i, j, pred, dist in zip(index_i, index_j, predictions, dmap):
        if np.isnan(dist):
            color = 'y'
            flag = np.nan

        else:
            color = 'g' if dist < 8 else 'r'
            flag = dist < 8

        if pred > threshold:
            true_preds += 1 if color == 'g' else 0
            if color == 'y':
                num -= 1
            plt.scatter(j, i, c=color)
        cmap[j, i] = flag
    print 
    print 'PPV:', true_preds / num
    print true_preds, num
    ppvs.append(true_preds / num)
    plt.imshow(cmap)
    plt.xlabel('PPV = {:.3f}, frac = {:.3f}'.format(true_preds / num, num / N))
    plt.title(prot.split('_')[1])
    plt.savefig(onlyName + '/' + prefix + '_' + prot.split('_')[1] + '.png')
    plt.close()
    print 'Shown'

print np.mean(ppvs), np.std(ppvs)
