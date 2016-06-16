from __future__ import division

import os
import glob
import time
import math
import random
import itertools
import collections

import numpy as np
import tables

from tqdm import tqdm

PATH = dict(train='../train_input', test='../test_input')
PAD_WIDTH = 5
keys = ('mi_corr', 'nmi_corr', 'cross_h')
keys_h5_mat = ('mi', 'nmi', 'cross_h')
keys_h5_vec = ('pssm0_i', 'pssm1_i', 'pssm0_j', 'pssm1_j')
keys_h5_float = ('seq_sep')
keys_h5_matrix=('mi', 'nmi', 'cross_h', 'mi_corr', 'nmi_corr')
keys_h5 = keys_h5_matrix + keys_h5_vec



def parse_PSSM(alignment):
    one2number = 'ARNDCEQGHILKMFPSTWYV-'
    bi = [0.0825, 0.0553, 0.0406, 0.0545, 0.0137, 0.0393, 0.0675, 0.0707, 0.0227, 0.0595, 0.0966, 0.0584, 0.0242, 0.0386, 0.0470, 0.0657, 0.0534, 0.0108, 0.0292, 0.0687]
    b = dict()
    for index, letter in enumerate(one2number[:-1]):
        b[letter] = bi[index]

    counts = collections.defaultdict(lambda: collections.defaultdict(int))
    seqcount = 0.
    gapcount = 0
    for line in open(alignment):
        line = line.strip()
        seqcount += 1
        for position, letter in enumerate(line):
            counts[position][letter] += 1
        gapcount += line.count('-')

    b['-'] = gapcount/(seqcount * len(counts.keys()))

    pssm = np.zeros((len(counts.keys()), len(one2number)), dtype=np.float32)
    pssm2 = np.zeros((len(counts.keys()), len(one2number)), dtype=np.float32)
    for position in counts:
        q0 = []
        q1 = []
        for letter in one2number:
            p = counts[position][letter] / (b[letter] * seqcount)
            if p > 0:
                q0.append(math.log(p))
                q1.append(p * math.log(p))
            else:
                q0.append(math.log(0.1 / (b[letter] * seqcount)))
                q1.append(0)

        pssm[position, :] = q0
        pssm2[position, :] = q1

    return pssm,  pssm2

class MIData(tables.IsDescription):
    is_contact = tables.Int8Col()
    mi = tables.Float32Col(shape=(PAD_WIDTH, PAD_WIDTH))
    ami = tables.Float32Col(shape=(PAD_WIDTH, PAD_WIDTH))
    nmi = tables.Float32Col(shape=(PAD_WIDTH, PAD_WIDTH))
    cross_h = tables.Float32Col(shape=(PAD_WIDTH, PAD_WIDTH))
    nmi_corr = tables.Float32Col(shape=(PAD_WIDTH, PAD_WIDTH))
    mi_corr = tables.Float32Col(shape=(PAD_WIDTH, PAD_WIDTH))   

    pssm0_i = tables.Float32Col(shape=(21,))
    pssm1_i = tables.Float32Col(shape=(21,))
    pssm0_j = tables.Float32Col(shape=(21,))
    pssm1_j = tables.Float32Col(shape=(21,))

    seq_sep = tables.Float32Col()

    dist = tables.Float32Col()

    i = tables.Int32Col()
    j = tables.Int32Col()

if __name__ == '__main__':
    h5 = tables.open_file('svmcon_dataset_filtered.h5', mode='w')
    h5_v = tables.open_file('svmcon_dataset_verify2.h5', mode='w')

    for mode in ('train', 'test'):
        t0 = time.time()
        prots = set(p.strip('.pdb') for p in glob.glob(os.path.abspath(os.path.join(PATH[mode], '*.pdb'))))
        cases = []
        for pr in tqdm(prots):
            cmap_f = pr + '.pdb.npz'
            mi_fname = pr + '.fa.jhE3.npz'
	    

            if os.path.exists(cmap_f) and os.path.exists(mi_fname):
                cmap_f = np.load(cmap_f)
                cmap = cmap_f['cmap']
                dmap = cmap_f['dmap']
                mi_f = np.load(mi_fname)
		mi_corr = mi_f['mi_corr']
		nmi_corr = mi_f['nmi_corr']
		
		col,row=nmi_corr.shape
		for c in xrange(0, row):
			for r in xrange(0, col):
				if c>r:
					nmi_corr[r,c]=nmi_corr[c,r]
		col,row=mi_corr.shape
		for c in xrange(0, row):
			for r in xrange(0, col):
				if c>r:
					mi_corr[r,c]=mi_corr[c,r]
		

                
                matrices = [np.pad(mi_f[ky], PAD_WIDTH // 2, 'constant', constant_values=0) 
			for ky in keys_h5_mat]
		matrices.append(np.pad(mi_corr, PAD_WIDTH // 2, 'constant', constant_values=0))
		matrices.append(np.pad(nmi_corr, PAD_WIDTH // 2, 'constant', constant_values=0))
                N = cmap.shape[0]

                pssm0, pssm1 = parse_PSSM(mi_fname.replace('.npz', '.aln'))
                this_protein = []

                for i, j in itertools.combinations_with_replacement(xrange(N), 2):
                    if abs(i - j) < 5:
                        continue
                    if np.isnan(cmap[i, j]):
                        continue

                    #if 8 <= dmap[i, j] <= 14 and mode == 'train':
                    #    continue

                    data = [m[i:i + PAD_WIDTH, j:j + PAD_WIDTH] for m in matrices]
                    data.extend((pssm0[i, :], pssm0[j, :], pssm1[i, :], pssm1[j, :]))

                    cases.append((data, cmap[i, j], i - j, dmap[i, j], i, j))
                    this_protein.append((data, cmap[i, j], i - j, dmap[i, j], i, j))
		    
                # Here save this_protein in its own table in h5_v
                table_name = mode + '_' + pr.split('/')[-1]
                table = h5_v.create_table(h5_v.root, table_name, MIData, filters=tables.Filters(9, 'blosc:snappy'))
                row = table.row
 		for each in this_protein:
           	    for key, matrix in zip(keys_h5, each[0]):
                	row[key] = matrix
            	    row['is_contact'] = each[1]
		    row['seq_sep'] = each[2]
		    row['dist'] = each[3]
		    row['i'] = each[4]
		    row['j'] = each[5]
                    row.append()
                table.flush()
	
        # Create master table, with all the proteins mixed
        random.shuffle(cases)
        print len(cases)
        print '{:.1f}s loading data'.format(time.time() - t0)

        t0 = time.time()
        table = h5.create_table(h5.root, mode, MIData, filters=tables.Filters(9, 'blosc:snappy'))
        row = table.row
	

        for each in cases:
            for key, matrix in zip(keys_h5, each[0]):
                row[key] = matrix
            row['is_contact'] = each[1]
            row['seq_sep'] = each[2]
            row['dist'] = each[3]
            row['i'] = each[4]
            row['j'] = each[5]

            row.append()
        table.flush()

        print '{:.1f}s saving data'.format(time.time() - t0)
   
    h5.close()

