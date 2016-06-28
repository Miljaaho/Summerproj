import os
import glob
from collections import defaultdict
import operator
import itertools
import shutil
import numpy as np
from shutil import copyfile
from Bio.PDB import PDBParser, CaPPBuilder, is_aa
import Bio.SeqIO
from Bio import pairwise2

pdb_parser = Bio.PDB.PDBParser()
fasta_parser = Bio.SeqIO.parse


codes = glob.glob('*.fasta')
for line in codes:
	pdb_id = line[:4]
        chain = line[4]
        file = open(line, "r")
	par_file = open(pdb_id + chain + '.fa', "w")
	fixer = 'False'	
	for l in file:
		if l[0] == '>' and l[6] == chain:	
			par_file.write(l)
			fixer = 'True'
		
		if l[0] != '>' and fixer == 'True':
			par_file.write(l)
		
		if l[0] =='>' and l[6] != chain:
			fixer ='False'






