#!/usr/bin/env python
# coding: utf-8

# ## Central dogma of molecular biology that describes the basic flow of genetic information

# In[1]:


# DNA -> RNA -> Proteins
# DNA: Sequence of characters where,
#       each character belongs to {A,G,C,T}
# Nucleotide/Triplet: Sequence of three characters -> Amino Acid
# Total amino acids: 20
# Sequence of amino acids -> Proteins
# dna.txt      -> DNA sequence 
# protein.txt  -> Translated Protein
# accession number: NM_201917.1
# Source of gene: Mus musculus (house mouse)


# NCBI Data is used for this case study ->
# [Link for Data](https://www.ncbi.nlm.nih.gov/nuccore/NM_207618.2)

# In[9]:


# Import Data and remove special characters
def process_data(file):
    """file: (string) name of the text file to process
       1. Removes special characters 
       2. Return a DNA sequence"""
    
    with open(file) as f:
        seq = f.read()
    seq = seq.replace("\n", "")
    seq = seq.replace("\r", "")
    return seq
dna = process_data("dna.txt")
prt = process_data("protein.txt")


# In[3]:


# DNA Translation to Protein


def translate(seq):
    """Translate a string containing a nucleotide sequence into a string 
    containing the corresponding sequence of amino acids. 
    Nucleotides are translated in triplets using the table dictionary; each amino acid 
    4 is encoded with a string of length 1. """

    table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W',
    }

    seq_len = len(seq)
    if seq_len % 3 == 0:
        protein = ""
        for i in range(0, seq_len, 3):
            codon = seq[i: i+3]
            protein += table[codon]
        return protein
    else:
        return f"Invalid Input Sequence, len = {seq_len}"


# In[12]:


# 21 and 938 are start and end of DNA sequence
# Sice last colon is stop codon so we remove that
translate(dna[20:935]) == prt


# In[13]:


translate(dna[20:938])[:-1] == translate(dna[20:935])


# In[ ]:




