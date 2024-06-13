"""Functions for parsing PCPs by region (CDR v FWK)."""

import pandas as pd

def seq_df_of_pcp_df(pcp_df):
    """
    Get all unique AA sequences from a PCP data set.
    """
    columns = [
        'sample_id', 'family', 'name', 'seq', 'depth', 'distance', 'v_gene', 
        'cdr1_codon_start', 'cdr1_codon_end', 
        'cdr2_codon_start', 'cdr2_codon_end', 
        'cdr3_codon_start', 'cdr3_codon_end', 
        'is_naive', 'is_leaf', 'v_family'
    ]
    seq_df = pd.DataFrame(columns=columns)

    added_sequences = {}

    def add_sequence(row, name_type, seq_type, is_parent):
        key = (row['sample_id'], row['family'], row[name_type])
        if key not in added_sequences:
            new_row = {
                'sample_id': row['sample_id'],
                'family': row['family'],
                'name': row[name_type],
                'seq': row[seq_type],
                'depth': row['depth'],
                'distance': row['distance'],
                'v_gene': row['v_gene'],
                'cdr1_codon_start': row['cdr1_codon_start'],
                'cdr1_codon_end': row['cdr1_codon_end'],
                'cdr2_codon_start': row['cdr2_codon_start'],
                'cdr2_codon_end': row['cdr2_codon_end'],
                'cdr3_codon_start': row['cdr3_codon_start'],
                'cdr3_codon_end': row['cdr3_codon_end'],
                'is_naive': True if is_parent and row['parent_is_naive'] else False,
                'is_leaf': False if is_parent else row['child_is_leaf'],
                'v_family': row['v_family']
            }
            seq_df.loc[len(seq_df)] = new_row
            added_sequences[key] = True

    for index, row in pcp_df.iterrows():
        add_sequence(row, 'parent_name', 'parent', True)  # Parent sequences, check parent_is_naive
        add_sequence(row, 'child_name', 'child', False)   # Child sequences, child_is_leaf only

    seq_df.set_index(['sample_id', 'family', 'name'], inplace=True)

    return seq_df


def aa_regions_of_row(row):
    """
    Calculate Python-style amino acid start and end positions for CDR and FWK regions from a single DataFrame row.
    
    By Python-style, we mean that the start position is inclusive and the end position is exclusive. 
    """
    regions = {
        'CDR1': (row['cdr1_codon_start'] // 3, (row['cdr1_codon_end'] // 3) + 1),
        'CDR2': (row['cdr2_codon_start'] // 3, (row['cdr2_codon_end'] // 3) + 1),
        'CDR3': (row['cdr3_codon_start'] // 3, (row['cdr3_codon_end'] // 3) + 1)
    }
    if "dnsm" in row:
        length = len(row["dnsm"])
    elif "parent_aa" in row:
        length = len(row["parent_aa"])
    else:
        raise ValueError("Row must have either 'dnsm' or 'parent_aa' column")
    regions.update({
        'FWK1': (0, regions['CDR1'][0]),
        'FWK2': (regions['CDR1'][1], regions['CDR2'][0]),
        'FWK3': (regions['CDR2'][1], regions['CDR3'][0]),
        'FWK4': (regions['CDR3'][1], length) 
    })
    regions = {k: regions[k] for k in ['FWK1', 'CDR1', 'FWK2', 'CDR2', 'FWK3', 'CDR3', 'FWK4']}
    return regions


def aa_seq_by_region(aa_seq, regions):
    """
    Get amino acid sequences by region from a single amino acid sequence.
    """
    return {region: aa_seq[start:end] for region, (start, end) in regions.items()}


def aaprob_by_region(aaprob, regions):
    """
    Get amino acid probabilities by region from a single amino acid probability sequence.
    """
    return {region: aaprob[start:end, :] for region, (start, end) in regions.items()}


def get_cdr_fwk_seqs(row):
    """
    Get amino acid sequences by region from a single amino acid sequence.
    """

    regions = aa_regions_of_row(row)
    parent_aa = row['parent_aa']
    child_aa = row['child_aa']

    # intialize the strings
    # this naming is maybe confusing because masked_cdr is the framework sequence and vice versa
    parent_masked_cdr = ['-' for _ in range(len(parent_aa))]
    parent_masked_fwk = ['-' for _ in range(len(parent_aa))]
    child_masked_cdr = ['-' for _ in range(len(child_aa))]
    child_masked_fwk = ['-' for _ in range(len(child_aa))]

    # fill in regions
    for region, (start, end) in regions.items():
        if region.startswith('CDR'):
            for i in range(start, end):
                parent_masked_fwk[i] = parent_aa[i]
                child_masked_fwk[i] = child_aa[i]
        else:
            for i in range(start, end):
                parent_masked_cdr[i] = parent_aa[i]
                child_masked_cdr[i] = child_aa[i]

    # convert lists to strings
    parent_masked_cdr = ''.join(parent_masked_cdr)
    parent_masked_fwk = ''.join(parent_masked_fwk)
    child_masked_cdr = ''.join(child_masked_cdr)
    child_masked_fwk = ''.join(child_masked_fwk)

    return parent_masked_cdr, parent_masked_fwk, child_masked_cdr, child_masked_fwk
