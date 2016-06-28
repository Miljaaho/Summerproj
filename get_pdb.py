import os
from Bio import PDB
from Bio.PDB import PDBList 
import numpy as np
import shutil
from Bio import SeqIO
	
class ChainSplitter:
    def __init__(self, out_dir=None):
        """ Create parsing and writing objects, specify output directory. """
        self.parser = PDB.PDBParser()
        self.writer = PDB.PDBIO()
        if out_dir is None:
            out_dir = os.getcwd()
        self.out_dir = out_dir

    def make_pdb(self, pdb_path, chain_letters, overwrite=False, struct=None):
        """ Create a new PDB file containing only the specified chains.

        Returns the path to the created file.

        :param pdb_path: full path to the crystal structure
        :param chain_letters: iterable of chain characters (case insensitive)
        :param overwrite: write over the output file if it exists
        """
        chain_letters = [chain.upper() for chain in chain_letters]

        # Input/output files
	
        (pdb_dir, pdb_fn) = os.path.split(pdb_path)
        pdb_id = pdb_fn[3:7]
	
	
        out_name = "%s%s.pdb" % (pdb_id.upper(), "".join(chain_letters))
	
        out_path = os.path.join(self.out_dir, out_name)
        print "OUT PATH:",out_path
        plural = "s" if (len(chain_letters) > 1) else ""  # for printing

        # Skip PDB generation if the file already exists
        if (not overwrite) and (os.path.isfile(out_path)):
            print("Chain%s %s of '%s' already extracted to '%s'." %
                    (plural, ", ".join(chain_letters), pdb_id, out_name))
            return out_path

        print("Extracting chain%s %s from %s..." % (plural,
                ", ".join(chain_letters), pdb_fn))

        # Get structure, write new file with only given chains
        if struct is None:
            struct = self.parser.get_structure(pdb_id, pdb_path)
        self.writer.set_structure(struct)
        self.writer.save(out_path, select=SelectChains(chain_letters))

        return out_path


class SelectChains(PDB.Select):
    """ Only accept the specified chains when saving. """
    def __init__(self, chain_letters):
        self.chain_letters = chain_letters

    def accept_chain(self, chain):
        return (chain.get_id() in self.chain_letters)


if __name__ == "__main__":
    """ Parses PDB id's desired chains, and creates new PDB structures. """
    import sys
    if not len(sys.argv) == 2:
        print "Usage: $ python %s 'pdb.txt'" % __file__
        sys.exit()

    pdb_textfn = sys.argv[1]
    pdbList = PDB.PDBList()
    splitter = ChainSplitter()  # Change me.

    with open(pdb_textfn) as pdb_textfile:
        for line in pdb_textfile:
            pdb_id = line[:4]
            chain = line[4] 
            pdb_fn = pdbList.retrieve_pdb_file(pdb_id)
            splitter.make_pdb(pdb_fn, chain)
	    shutil.rmtree(os.path.split(pdb_fn)[0])
	    
