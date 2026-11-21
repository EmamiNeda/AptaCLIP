"""Sequence normalization from the supplied final experiment notebook (cell 2)."""
import re

def clean_protein_matrix(s):
    s = re.sub(r'[^A-Za-z]', '', str(s).upper())
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', 'X', s)

def clean_aptamer_matrix(s):
    raw = re.sub(r'\s+', '', str(s).upper())
    flag = int('U' in raw)
    dna = re.sub(r'[^ACGTU]', '', raw).replace('U', 'T')
    return dna, flag

