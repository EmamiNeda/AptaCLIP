"""Faithful adaptation of code cell 2 of the supplied final notebook.
Run from the repository root. Paths are configurable through environment variables.
This is a validation-only grid search, not a guaranteed reproduction of manuscript metrics.
"""
from dataset import clean_protein_matrix, clean_aptamer_matrix
from model import (triplet_loss_matrix, protein_projection_matrix, aptamer_projection_matrix, build_model_matrix)

# FROZEN WINNING BASELINE — SHARED SETUP FOR THE 3 x 4 ENCODER/DATA MATRIX
# This cell prepares the exact data protocols and reusable frozen embeddings.
# It does NOT train an experiment. Each experiment is launched from its own cell.
#
# Frozen for every experiment:
#   classification rows derived from each valid triplet (P,A+,A-):
#       (P,A+) -> 1 and (P,A-) -> 0
#   Triplet loss always sees the valid orientation (P,A+,A-)
#   RNA/DNA flag follows the aptamer used by the classification head
#   Triplet loss: ignores y_true; margin=2; weight=2
#   BCE weight=2; exact projection/interaction head
#   Adam(1e-4), batch=16, val_loss patience=10
#   seeds=[42,52,62,72,82], fixed stratified 80/20 Validation
#   ensemble mean, threshold selected by Validation Accuracy only
#
# Authorized dimensions of comparison only:
#   Encoders: DB6+ESM650, DB2+ESM35, DB2+ESM650
#   Data: random, cluster0p5, matched_length, cluster0p5_matched_length

import os, re, gc, sys, json, time, shutil, subprocess, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, precision_recall_curve, auc as calc_auc
)
try:
    from IPython.display import display, FileLink
except ImportError:
    display = lambda *args: None
    FileLink = lambda p: str(p)

SOURCE_TRAIN_MATRIX = Path(os.environ.get('APTACLIP_TRAIN', 'data/train_set.csv'))
SOURCE_TEST_MATRIX = Path(os.environ.get('APTACLIP_TEST', 'data/test_set.csv'))
MATRIX_ROOT = Path(os.environ.get('APTACLIP_OUTPUT', 'outputs/frozen_winner_encoder_data_matrix'))
MATRIX_DATA = MATRIX_ROOT / 'shared_data'
MATRIX_CACHE = MATRIX_ROOT / 'shared_encoder_cache'
MATRIX_DATA.mkdir(parents=True, exist_ok=True)
MATRIX_CACHE.mkdir(parents=True, exist_ok=True)
assert SOURCE_TRAIN_MATRIX.exists() and SOURCE_TEST_MATRIX.exists(), 'Upload train_set.csv and test_set.csv'

FROZEN_CONFIG_MATRIX = {
    'Rows': 'classification: (P,A+)->1 and (P,A-)->0; triplet always (P,A+,A-)',
    'Negative_flag': 'flag follows classification aptamer A+ or A-',
    'Triplet_loss': 'margin selected on cluster-disjoint Validation only',
    'Loss_weights': 'BCE=2; Triplet weight selected on cluster-disjoint Validation only',
    'Optimizer': 'Adam lr=1e-4',
    'Batch_size': 16,
    'Early_stopping': 'val_loss patience=10',
    'Seeds': [42,52,62,72,82],
    'Validation': 'protein-cluster-disjoint split from protocol Train; random_state=42',
    'Threshold': 'selected on ensemble Validation only; Test never used for selection',
    'Test_used_for_selection': False,
}
ENCODER_LABELS_MATRIX = {
    'db6_esm650': 'DNABERT-6 + ESM2-650M',
    'db2_esm35': 'DNABERT-2 + ESM2-35M',
    'db2_esm650': 'DNABERT-2 + ESM2-650M',
}
PROTOCOL_LABELS_MATRIX = {
    'random': 'original Train/Test + unrestricted random Class=0 rows',
    'cluster0p5': 'CD-HIT protein cluster 0.5 Train/Test + unrestricted random Class=0 rows',
    'matched_length': 'original Train/Test + Class=0 B within per-P [min positive length-10, max+10]',
    'cluster0p5_matched_length': 'CD-HIT protein cluster 0.5 Train/Test + Class=0 B within per-P [min positive length-10, max+10]',
}



source_train_matrix = pd.read_csv(SOURCE_TRAIN_MATRIX, encoding='latin1').fillna('')
source_test_matrix = pd.read_csv(SOURCE_TEST_MATRIX, encoding='latin1').fillna('')
required_matrix = {'Target_Sequence','Aptamer_Sequence','Class'}
assert required_matrix.issubset(source_train_matrix.columns)
assert required_matrix.issubset(source_test_matrix.columns)
for df_matrix in (source_train_matrix, source_test_matrix):
    df_matrix['Class'] = pd.to_numeric(df_matrix['Class']).astype(int)
    assert set(df_matrix['Class']).issubset({0,1})

# ----- Exact previous CD-HIT 0.5 definition and exact seed-42 cluster split -----
all_rows_matrix = pd.concat([
    source_train_matrix.assign(_Original_Split='Train'),
    source_test_matrix.assign(_Original_Split='Test')
], ignore_index=True)
all_rows_matrix['_Clean_Protein'] = all_rows_matrix['Target_Sequence'].map(clean_protein_matrix)
unique_proteins_matrix = sorted(all_rows_matrix['_Clean_Protein'].unique())
seq_to_id_matrix = {s:f'protein_{i:06d}' for i,s in enumerate(unique_proteins_matrix)}
id_to_seq_matrix = {v:k for k,v in seq_to_id_matrix.items()}
fasta_matrix = MATRIX_DATA / 'all_unique_proteins.fasta'
with fasta_matrix.open('w') as h:
    for s in unique_proteins_matrix:
        h.write(f'>{seq_to_id_matrix[s]}\n{s}\n')
if shutil.which('cd-hit') is None:
    raise RuntimeError('CD-HIT is required. Install cd-hit before running (e.g., conda install -c bioconda cd-hit).')
cdhit_out_matrix = MATRIX_DATA / 'cdhit_protein_0p5'
for p in (cdhit_out_matrix, Path(str(cdhit_out_matrix)+'.clstr')):
    if p.exists(): p.unlink()
subprocess.check_call([
    'cd-hit','-i',str(fasta_matrix),'-o',str(cdhit_out_matrix),
    '-c','0.50','-n','3','-G','1','-g','1','-d','0','-T','0','-M','0'
])
id_to_cluster_matrix = {}
current_cluster_matrix = None
with Path(str(cdhit_out_matrix)+'.clstr').open() as h:
    for raw in h:
        line = raw.strip()
        if line.startswith('>Cluster'):
            current_cluster_matrix = int(line.split()[1])
        elif '>' in line:
            ident = line.split('>',1)[1].split('...',1)[0]
            id_to_cluster_matrix[ident] = current_cluster_matrix
next_cluster_matrix = max(id_to_cluster_matrix.values())+1 if id_to_cluster_matrix else 0
for ident in sorted(set(id_to_seq_matrix)-set(id_to_cluster_matrix)):
    id_to_cluster_matrix[ident] = next_cluster_matrix
    next_cluster_matrix += 1
seq_to_cluster_matrix = {
    seq:int(id_to_cluster_matrix[ident]) for ident,seq in id_to_seq_matrix.items()
}
all_rows_matrix['Protein_Cluster_ID'] = all_rows_matrix['_Clean_Protein'].map(seq_to_cluster_matrix).astype(int)
stats_matrix = all_rows_matrix.groupby('Protein_Cluster_ID',sort=True).agg(
    rows=('Class','size'),
    positives=('Class',lambda x:int(pd.Series(x).astype(int).eq(1).sum()))
)
target_rows_matrix = len(source_test_matrix)
target_rate_matrix = float(source_test_matrix['Class'].eq(1).mean())
cluster_ids_matrix = stats_matrix.index.to_numpy(int)
rng_split_matrix = np.random.default_rng(42)
best_selection_matrix, best_score_matrix = None, np.inf
for _ in range(5000):
    perm = rng_split_matrix.permutation(cluster_ids_matrix)
    rows_now = positives_now = 0
    selected = []
    for cid in perm:
        selected.append(int(cid))
        rows_now += int(stats_matrix.loc[cid,'rows'])
        positives_now += int(stats_matrix.loc[cid,'positives'])
        if rows_now >= max(1,target_rows_matrix*.75):
            rate_now = positives_now/max(1,rows_now)
            score = abs(rows_now-target_rows_matrix)/max(1,target_rows_matrix)+2*abs(rate_now-target_rate_matrix)
            if score < best_score_matrix:
                best_score_matrix, best_selection_matrix = score, tuple(selected)
        if rows_now > target_rows_matrix*1.25: break
assert best_selection_matrix is not None
test_clusters_matrix = set(best_selection_matrix)
cluster_test_matrix = all_rows_matrix[all_rows_matrix['Protein_Cluster_ID'].isin(test_clusters_matrix)].copy()
cluster_train_matrix = all_rows_matrix[~all_rows_matrix['Protein_Cluster_ID'].isin(test_clusters_matrix)].copy()
assert set(cluster_train_matrix['Protein_Cluster_ID']).isdisjoint(set(cluster_test_matrix['Protein_Cluster_ID']))
drop_internal_matrix = ['_Original_Split','_Clean_Protein']
cluster_train_matrix = cluster_train_matrix.drop(columns=drop_internal_matrix).reset_index(drop=True)
cluster_test_matrix = cluster_test_matrix.drop(columns=drop_internal_matrix).reset_index(drop=True)
cluster_train_matrix.to_csv(MATRIX_DATA/'train_cluster0p5.csv',index=False)
cluster_test_matrix.to_csv(MATRIX_DATA/'test_cluster0p5.csv',index=False)
CLUSTER_AUDIT_MATRIX = {
    'unique_proteins':len(unique_proteins_matrix),
    'clusters':len(set(seq_to_cluster_matrix.values())),
    'train_rows':len(cluster_train_matrix),'test_rows':len(cluster_test_matrix),
    'train_clusters':int(cluster_train_matrix['Protein_Cluster_ID'].nunique()),
    'test_clusters':int(cluster_test_matrix['Protein_Cluster_ID'].nunique()),
    'cluster_overlap':0,
}
print('CLUSTER AUDIT',CLUSTER_AUDIT_MATRIX,flush=True)

# ----- Frozen sequence universe and component caches -----
all_original_matrix = pd.concat([source_train_matrix,source_test_matrix],ignore_index=True)
all_original_matrix['_P'] = all_original_matrix['Target_Sequence'].map(clean_protein_matrix)
all_original_matrix['_A'] = all_original_matrix['Aptamer_Sequence'].map(lambda x:clean_aptamer_matrix(x)[0])
protein_keys_matrix = all_original_matrix['_P'].drop_duplicates().tolist()
aptamer_keys_matrix = all_original_matrix['_A'].drop_duplicates().tolist()
assert all(len(x)>=6 for x in aptamer_keys_matrix)
device_matrix = 'cuda' if torch.cuda.is_available() else 'cpu'
assert device_matrix == 'cuda', 'GPU is required for this matrix'

def masked_mean_matrix(hidden, attention_mask, input_ids, tokenizer):
    mask = attention_mask.bool()
    for sid in tokenizer.all_special_ids:
        mask = mask & input_ids.ne(sid)
    denom = mask.sum(dim=1,keepdim=True).clamp(min=1)
    return (hidden*mask.unsqueeze(-1)).sum(dim=1)/denom

esm650_file_matrix = MATRIX_CACHE/'ESM2_650M_proteins.npz'
if not esm650_file_matrix.exists():
    print('BUILDING ESM2-650M COMPONENT CACHE',flush=True)
    start = time.time()
    esm650_tok_matrix = AutoTokenizer.from_pretrained('facebook/esm2_t33_650M_UR50D')
    esm650_model_matrix = AutoModel.from_pretrained(
        'facebook/esm2_t33_650M_UR50D',torch_dtype=(torch.float16 if device_matrix == 'cuda' else torch.float32),low_cpu_mem_usage=True
    ).to(device_matrix).eval()
    @torch.inference_mode()
    def encode_esm650_one_matrix(seq):
        chunks=[seq[i:i+1022] for i in range(0,len(seq),1022)]
        vals=[]; weights=[]
        for ch in chunks:
            tok=esm650_tok_matrix(ch,return_tensors='pt',padding=False,truncation=True,max_length=1024).to(device_matrix)
            with torch.autocast(device_type=device_matrix, dtype=torch.float16, enabled=(device_matrix == 'cuda')):
                hidden=esm650_model_matrix(**tok).last_hidden_state
                pooled=masked_mean_matrix(hidden,tok['attention_mask'],tok['input_ids'],esm650_tok_matrix)[0]
            vals.append(pooled.float().cpu().numpy()); weights.append(len(ch))
            del tok,hidden,pooled
        return np.average(np.stack(vals),axis=0,weights=np.asarray(weights)).astype(np.float32)
    vecs=[]
    for i,s in enumerate(protein_keys_matrix,1):
        vecs.append(encode_esm650_one_matrix(s))
        if i%10==0 or i==len(protein_keys_matrix):
            print(f'ESM2-650M {i}/{len(protein_keys_matrix)} elapsed={time.time()-start:.1f}s',flush=True)
    np.savez_compressed(esm650_file_matrix,protein_sequences=np.asarray(protein_keys_matrix,dtype=object),protein_embeddings=np.stack(vecs))
    del esm650_model_matrix,esm650_tok_matrix,vecs
    torch.cuda.empty_cache(); gc.collect()
else:
    print('REUSING',esm650_file_matrix,flush=True)
esm650_cache_matrix=np.load(esm650_file_matrix,allow_pickle=True)
ESM650_MAP_MATRIX={str(k):v.astype(np.float32) for k,v in zip(esm650_cache_matrix['protein_sequences'],esm650_cache_matrix['protein_embeddings'])}

db2_file_matrix=MATRIX_CACHE/'DNABERT2_aptamers.npz'
if not db2_file_matrix.exists():
    if importlib.util.find_spec('multimolecule') is None:
        subprocess.check_call([sys.executable,'-m','pip','install','-q','multimolecule'])
    _original_register_matrix = AutoConfig.register
    def _safe_register_matrix(model_type, config, exist_ok=False):
        return _original_register_matrix(model_type, config, exist_ok=True)
    AutoConfig.register = _safe_register_matrix
    from multimolecule import AutoTokenizer as MMAutoTokenizer, AutoModel as MMAutoModel
    print('BUILDING DNABERT-2 COMPONENT CACHE',flush=True)
    start=time.time()
    db2_tok_matrix=MMAutoTokenizer.from_pretrained('multimolecule/dnabert2')
    db2_model_matrix=MMAutoModel.from_pretrained('multimolecule/dnabert2').to(device_matrix).eval()
    @torch.inference_mode()
    def encode_db2_batch_matrix(seqs):
        tok=db2_tok_matrix(seqs,return_tensors='pt',padding=True,truncation=True,max_length=150).to(device_matrix)
        out=db2_model_matrix(**tok).last_hidden_state
        return masked_mean_matrix(out,tok['attention_mask'],tok['input_ids'],db2_tok_matrix).float().cpu().numpy().astype(np.float32)
    vals=[]
    for i in range(0,len(aptamer_keys_matrix),32):
        vals.append(encode_db2_batch_matrix(aptamer_keys_matrix[i:i+32]))
        done=min(i+32,len(aptamer_keys_matrix))
        if done%320==0 or done==len(aptamer_keys_matrix):
            print(f'DNABERT-2 {done}/{len(aptamer_keys_matrix)} elapsed={time.time()-start:.1f}s',flush=True)
    np.savez_compressed(db2_file_matrix,aptamer_sequences_dna=np.asarray(aptamer_keys_matrix,dtype=object),aptamer_embeddings=np.vstack(vals))
    del db2_model_matrix,db2_tok_matrix,vals
    torch.cuda.empty_cache(); gc.collect()
else:
    print('REUSING',db2_file_matrix,flush=True)
db2_cache_matrix=np.load(db2_file_matrix,allow_pickle=True)
DB2_MAP_MATRIX={str(k):v.astype(np.float32) for k,v in zip(db2_cache_matrix['aptamer_sequences_dna'],db2_cache_matrix['aptamer_embeddings'])}

assert set(ESM650_MAP_MATRIX)==set(protein_keys_matrix)
assert set(DB2_MAP_MATRIX)==set(aptamer_keys_matrix)

def maps_for_encoder_matrix(encoder_key):
    if encoder_key=='db2_esm650': return ESM650_MAP_MATRIX,DB2_MAP_MATRIX
    raise ValueError(encoder_key)

def split_for_protocol_matrix(protocol_key):
    if protocol_key in ('cluster0p5','cluster0p5_matched_length'):
        return cluster_train_matrix,cluster_test_matrix
    if protocol_key in ('random','matched_length'):
        return source_train_matrix,source_test_matrix
    raise ValueError(protocol_key)

def uncorrected_triplets_matrix(df,protocol_key,seed):
    pos=df[df['Class'].astype(int).eq(1)].reset_index(drop=True).copy()
    neg=df[df['Class'].astype(int).eq(0)].reset_index(drop=True).copy()
    assert len(pos)>0 and len(neg)>0
    pos['_P']=pos['Target_Sequence'].map(clean_protein_matrix)
    pos['_A']=pos['Aptamer_Sequence'].map(lambda x:clean_aptamer_matrix(x)[0])
    pos['_L']=pos['_A'].str.len()
    neg['_B']=neg['Aptamer_Sequence'].map(lambda x:clean_aptamer_matrix(x)[0])
    neg['_L']=neg['_B'].str.len()
    use_length=protocol_key in ('matched_length','cluster0p5_matched_length')
    bounds=pos.groupby('_P')['_L'].agg(['min','max']).to_dict('index') if use_length else {}
    rng=np.random.default_rng(seed)
    triplets=[]; audit=[]
    for _,pr in pos.iterrows():
        if use_length:
            lo=max(1,int(bounds[pr['_P']]['min'])-10)
            hi=int(bounds[pr['_P']]['max'])+10
            cand=neg[neg['_L'].between(lo,hi)]
            fallback=False
            if cand.empty:
                fallback=True
                lens=neg['_L'].to_numpy()
                distance=np.where(lens<lo,lo-lens,np.where(lens>hi,lens-hi,0))
                cand=neg.iloc[np.flatnonzero(distance==distance.min())]
        else:
            lo=hi=None; cand=neg; fallback=False

        nr=cand.iloc[int(rng.integers(0,len(cand)))]
        P=str(pr['Target_Sequence'])
        A=str(pr['Aptamer_Sequence'])   # A+
        B=str(nr['Aptamer_Sequence'])   # A- selected from Class=0 pool

        flagA=int('U' in re.sub(r'\s+','',A.upper()))
        flagB=int('U' in re.sub(r'\s+','',B.upper()))

        # CORRECTED LOGIC:
        # Both rows carry the SAME valid triplet orientation (P, A+, A-).
        # cls_ap is the aptamer actually shown to the classification head.
        #
        # Positive classification pair: (P,A+) -> 1
        triplets.append((P,A,B,A,flagA,1))
        # Negative classification pair: (P,A-) -> 0
        triplets.append((P,A,B,B,flagB,0))

        alen=len(clean_aptamer_matrix(A)[0]); blen=len(clean_aptamer_matrix(B)[0])
        audit.append({
            'P':clean_protein_matrix(P),
            'A_length':alen,'B_length':blen,'lower':lo,'upper':hi,
            'within':True if not use_length else bool(lo<=blen<=hi),
            'fallback':fallback
        })
    return triplets,pd.DataFrame(audit)


def encode_triplets_matrix(triplets,protein_map,aptamer_map):
    # tuple layout:
    # (P, triplet_A_positive, triplet_A_negative, classification_aptamer, flag, label)
    p=[clean_protein_matrix(t[0]) for t in triplets]
    a_pos=[clean_aptamer_matrix(t[1])[0] for t in triplets]
    a_neg=[clean_aptamer_matrix(t[2])[0] for t in triplets]
    a_cls=[clean_aptamer_matrix(t[3])[0] for t in triplets]

    assert not (set(p)-set(protein_map))
    assert not (set(a_pos+a_neg+a_cls)-set(aptamer_map))

    X=[
        np.stack([protein_map[k] for k in p]).astype(np.float32),
        np.stack([aptamer_map[k] for k in a_pos]).astype(np.float32),
        np.stack([aptamer_map[k] for k in a_neg]).astype(np.float32),
        np.stack([aptamer_map[k] for k in a_cls]).astype(np.float32),
        np.asarray([t[4] for t in triplets],np.float32).reshape(-1,1)
    ]
    labels=np.asarray([t[5] for t in triplets],np.float32)
    return X,{'interaction':labels,'triplet_output':np.zeros(len(labels),np.float32)}






def metrics_matrix(y,p,t):
    y=np.asarray(y).astype(int); p=np.asarray(p).reshape(-1); z=(p>=t).astype(int)
    pc,rc,_=precision_recall_curve(y,p)
    return {'Threshold':float(t),'Accuracy':float(accuracy_score(y,z)),'Precision':float(precision_score(y,z,zero_division=0)),'Recall':float(recall_score(y,z,zero_division=0)),'F1':float(f1_score(y,z,zero_division=0)),'MCC':float(matthews_corrcoef(y,z)),'ROC_AUC':float(roc_auc_score(y,p)),'AUPR':float(calc_auc(rc,pc))}

def choose_threshold_matrix(y,p):
    rows=[metrics_matrix(y,p,t) for t in np.linspace(.10,.90,161)]
    tab=pd.DataFrame(rows); tab['Distance_from_0p5']=(tab['Threshold']-.5).abs()
    best=tab.sort_values(['Accuracy','F1','Precision','Distance_from_0p5'],ascending=[False,False,False,True]).iloc[0]
    return float(best['Threshold']),tab

def _cluster_disjoint_train_val_matrix(train_df, val_fraction=0.20, seed=42):
    """
    Split protocol-Train into protein-cluster-disjoint fit/validation subsets.
    Uses Protein_Cluster_ID when present (cluster protocols).
    """
    if 'Protein_Cluster_ID' not in train_df.columns:
        raise ValueError(
            "Cluster-disjoint validation requires Protein_Cluster_ID. "
            "Use cluster0p5 or cluster0p5_matched_length."
        )

    stats = train_df.groupby('Protein_Cluster_ID', sort=True).agg(
        rows=('Class','size'),
        positives=('Class', lambda x:int(pd.Series(x).astype(int).eq(1).sum()))
    )
    total_rows = len(train_df)
    target_rows = max(1, int(round(total_rows * val_fraction)))
    overall_rate = float(train_df['Class'].astype(int).eq(1).mean())

    cluster_ids = stats.index.to_numpy(int)
    rng = np.random.default_rng(seed)
    best_sel, best_score = None, np.inf

    # Search many deterministic random cluster subsets for a validation split
    # close to 20% rows and close to the training positive rate.
    for _ in range(5000):
        perm = rng.permutation(cluster_ids)
        selected = []
        rows_now = 0
        pos_now = 0
        for cid in perm:
            selected.append(int(cid))
            rows_now += int(stats.loc[cid,'rows'])
            pos_now += int(stats.loc[cid,'positives'])

            if rows_now >= max(1, target_rows * 0.75):
                rate_now = pos_now / max(1, rows_now)
                score = (
                    abs(rows_now-target_rows)/max(1,target_rows)
                    + 2.0*abs(rate_now-overall_rate)
                )
                if score < best_score:
                    best_score = score
                    best_sel = tuple(selected)

            if rows_now > target_rows * 1.25:
                break

    assert best_sel is not None
    val_clusters = set(best_sel)

    val_df = train_df[
        train_df['Protein_Cluster_ID'].isin(val_clusters)
    ].copy().reset_index(drop=True)

    fit_df = train_df[
        ~train_df['Protein_Cluster_ID'].isin(val_clusters)
    ].copy().reset_index(drop=True)

    assert set(fit_df['Protein_Cluster_ID']).isdisjoint(
        set(val_df['Protein_Cluster_ID'])
    )

    audit = {
        'fit_rows': int(len(fit_df)),
        'validation_rows': int(len(val_df)),
        'fit_clusters': int(fit_df['Protein_Cluster_ID'].nunique()),
        'validation_clusters': int(val_df['Protein_Cluster_ID'].nunique()),
        'fit_validation_cluster_overlap': 0,
        'fit_positive_rate': float(fit_df['Class'].astype(int).eq(1).mean()),
        'validation_positive_rate': float(val_df['Class'].astype(int).eq(1).mean()),
    }
    return fit_df, val_df, audit


def _mean_validation_score_matrix(metrics):
    """
    Primary model-selection score:
      1) AUPR
      2) ROC-AUC
      3) MCC
      4) F1
    Test is never consulted.
    """
    return (
        float(metrics['AUPR']),
        float(metrics['ROC_AUC']),
        float(metrics['MCC']),
        float(metrics['F1'])
    )


def run_frozen_matrix_experiment(encoder_key,protocol_key):
    assert encoder_key in ENCODER_LABELS_MATRIX
    assert protocol_key in PROTOCOL_LABELS_MATRIX

    train_df,test_df=split_for_protocol_matrix(protocol_key)
    protein_map,aptamer_map=maps_for_encoder_matrix(encoder_key)

    if 'cluster0p5' not in protocol_key:
        raise ValueError(
            "This controlled search is designed for cluster-disjoint validation. "
            "Run it with cluster0p5 or cluster0p5_matched_length."
        )

    # ------------------------------------------------------------
    # 1) CLUSTER-DISJOINT FIT / VALIDATION
    # ------------------------------------------------------------
    fit_df,val_df,val_cluster_audit = _cluster_disjoint_train_val_matrix(
        train_df,val_fraction=0.20,seed=42
    )

    fit_clusters=set(fit_df['Protein_Cluster_ID'].astype(int))
    val_clusters=set(val_df['Protein_Cluster_ID'].astype(int))
    test_clusters=set(test_df['Protein_Cluster_ID'].astype(int))

    assert fit_clusters.isdisjoint(val_clusters)
    assert fit_clusters.isdisjoint(test_clusters)
    assert val_clusters.isdisjoint(test_clusters)

    out=MATRIX_ROOT/f'{encoder_key}__{protocol_key}__controlled_search'
    data_dir=out/'data'
    weight_dir=out/'weights'
    data_dir.mkdir(parents=True,exist_ok=True)
    weight_dir.mkdir(parents=True,exist_ok=True)

    result_file=out/'final_result.json'
    pred_file=out/'independent_test_predictions.csv'

    print(
        '\nSTART CONTROLLED SEARCH',
        ENCODER_LABELS_MATRIX[encoder_key],
        '|',
        PROTOCOL_LABELS_MATRIX[protocol_key],
        flush=True
    )
    print('FROZEN CONFIG',json.dumps(FROZEN_CONFIG_MATRIX,indent=2),flush=True)
    print('VALIDATION CLUSTER AUDIT',json.dumps(val_cluster_audit,indent=2),flush=True)

    # ------------------------------------------------------------
    # 2) FREEZE NEGATIVE SAMPLING FOR FAIR HYPERPARAMETER COMPARISON
    # ------------------------------------------------------------
    # One fixed sampled dataset per split. All margins/weights/seeds see
    # exactly the same sampled triplets.
    FIT_SAMPLING_SEED=14201
    VAL_SAMPLING_SEED=14202
    TEST_SAMPLING_SEED=14203

    fit_triplets,fit_audit=uncorrected_triplets_matrix(
        fit_df,protocol_key,FIT_SAMPLING_SEED
    )
    val_triplets,val_audit=uncorrected_triplets_matrix(
        val_df,protocol_key,VAL_SAMPLING_SEED
    )
    test_triplets,test_audit=uncorrected_triplets_matrix(
        test_df,protocol_key,TEST_SAMPLING_SEED
    )

    fit_audit.to_csv(data_dir/'fixed_fit_sampling_audit.csv',index=False)
    val_audit.to_csv(data_dir/'fixed_validation_sampling_audit.csv',index=False)
    test_audit.to_csv(data_dir/'fixed_test_sampling_audit.csv',index=False)

    X_fit,y_fit=encode_triplets_matrix(fit_triplets,protein_map,aptamer_map)
    X_val,y_val=encode_triplets_matrix(val_triplets,protein_map,aptamer_map)
    X_test,y_test=encode_triplets_matrix(test_triplets,protein_map,aptamer_map)

    y_val_label=y_val['interaction'].astype(int)
    y_test_label=y_test['interaction'].astype(int)

    sampling_audit={
        'fit_positive_rows':int(len(fit_audit)),
        'validation_positive_rows':int(len(val_audit)),
        'test_positive_rows':int(len(test_audit)),
        'fit_within_fraction':float(fit_audit['within'].mean()),
        'validation_within_fraction':float(val_audit['within'].mean()),
        'test_within_fraction':float(test_audit['within'].mean()),
        'fit_fallbacks':int(fit_audit['fallback'].sum()),
        'validation_fallbacks':int(val_audit['fallback'].sum()),
        'test_fallbacks':int(test_audit['fallback'].sum()),
        'negative_sampling_frozen_across_hyperparameters':True,
        'fit_sampling_seed':FIT_SAMPLING_SEED,
        'validation_sampling_seed':VAL_SAMPLING_SEED,
        'test_sampling_seed':TEST_SAMPLING_SEED,
    }

    print('SAMPLING AUDIT',json.dumps(sampling_audit,indent=2),flush=True)

    # ------------------------------------------------------------
    # 3) VALIDATION-ONLY HYPERPARAMETER GRID
    # ------------------------------------------------------------
    # Includes classification-only control.
    search_grid=[
        {'name':'BCE2_TRIP2_M2',   'bce_weight':2.0,'triplet_weight':2.0,'margin':2.0},
        {'name':'BCE2_TRIP1_M1',   'bce_weight':2.0,'triplet_weight':1.0,'margin':1.0},
        {'name':'BCE2_TRIP0p5_M1', 'bce_weight':2.0,'triplet_weight':0.5,'margin':1.0},
        {'name':'BCE2_TRIP1_M0p5', 'bce_weight':2.0,'triplet_weight':1.0,'margin':0.5},
        {'name':'BCE2_TRIP0p5_M0p5','bce_weight':2.0,'triplet_weight':0.5,'margin':0.5},
        {'name':'BCE2_TRIP0p25_M0p5','bce_weight':2.0,'triplet_weight':0.25,'margin':0.5},
        {'name':'BCE2_ONLY',        'bce_weight':2.0,'triplet_weight':0.0,'margin':0.5},
    ]

    search_records=[]
    best_record=None
    best_score=None

    for cfg in search_grid:
        print('\nCONFIG',json.dumps(cfg),flush=True)

        cfg_val_probs=[]
        cfg_seed_records=[]

        cfg_weight_dir=weight_dir/cfg['name']
        cfg_weight_dir.mkdir(parents=True,exist_ok=True)

        for seed in FROZEN_CONFIG_MATRIX['Seeds']:
            keras.backend.clear_session()
            keras.utils.set_random_seed(seed)

            model=build_model_matrix(
                X_fit[0].shape[1],
                X_fit[1].shape[1]
            )

            model.compile(
                optimizer=keras.optimizers.Adam(1e-4),
                loss={
                    'interaction':'binary_crossentropy',
                    'triplet_output':triplet_loss_matrix(cfg['margin'])
                },
                loss_weights={
                    'interaction':cfg['bce_weight'],
                    'triplet_output':cfg['triplet_weight']
                },
                metrics={
                    'interaction':[
                        'accuracy',
                        keras.metrics.AUC(name='auc'),
                        keras.metrics.AUC(name='aupr',curve='PR')
                    ]
                }
            )

            start=time.time()
            hist=model.fit(
                X_fit,y_fit,
                validation_data=(X_val,y_val),
                epochs=200,
                batch_size=16,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        mode='min',
                        patience=10,
                        restore_best_weights=True,
                        verbose=0
                    )
                ]
            )

            vp=model.predict(X_val,verbose=0)[0].reshape(-1)

            # Per-seed threshold is only diagnostic.
            seed_threshold,_=choose_threshold_matrix(y_val_label,vp)
            seed_metrics=metrics_matrix(
                y_val_label,vp,seed_threshold
            )

            rec={
                'Seed':seed,
                'Epochs':len(hist.history['loss']),
                'Seconds':time.time()-start,
                'Validation_threshold':float(seed_threshold),
                'Validation':seed_metrics
            }
            print('SEED RESULT',json.dumps(rec),flush=True)

            model.save_weights(
                cfg_weight_dir/f'seed_{seed}.weights.h5'
            )

            cfg_val_probs.append(vp)
            cfg_seed_records.append(rec)

            del model,hist
            keras.backend.clear_session()
            gc.collect()

        ensemble_val=np.mean(np.stack(cfg_val_probs),axis=0)

        # Threshold is selected ONLY from ensemble validation.
        threshold,threshold_table=choose_threshold_matrix(
            y_val_label,ensemble_val
        )
        val_metrics=metrics_matrix(
            y_val_label,ensemble_val,threshold
        )

        threshold_table.to_csv(
            data_dir/f'{cfg["name"]}__validation_threshold_grid.csv',
            index=False
        )

        cfg_record={
            **cfg,
            'Selected_threshold':float(threshold),
            'Validation':val_metrics,
            'Per_seed':cfg_seed_records
        }
        search_records.append(cfg_record)

        score=_mean_validation_score_matrix(val_metrics)

        print(
            'CONFIG VALIDATION RESULT',
            json.dumps(cfg_record,indent=2),
            flush=True
        )

        if best_score is None or score>best_score:
            best_score=score
            best_record=cfg_record

    assert best_record is not None

    pd.DataFrame([
        {
            'Config':r['name'],
            'BCE_weight':r['bce_weight'],
            'Triplet_weight':r['triplet_weight'],
            'Margin':r['margin'],
            'Threshold':r['Selected_threshold'],
            **{f'Val_{k}':v for k,v in r['Validation'].items() if k!='Threshold'}
        }
        for r in search_records
    ]).to_csv(out/'validation_model_selection.csv',index=False)

    print(
        '\nSELECTED FROM VALIDATION ONLY',
        json.dumps(best_record,indent=2),
        flush=True
    )

    # ------------------------------------------------------------
    # 4) INDEPENDENT TEST — ONLY AFTER CONFIGURATION IS LOCKED
    # ------------------------------------------------------------
    selected_name=best_record['name']
    selected_margin=float(best_record['margin'])
    selected_bce=float(best_record['bce_weight'])
    selected_trip=float(best_record['triplet_weight'])
    selected_threshold=float(best_record['Selected_threshold'])

    # Reload the already-trained selected validation models.
    # No test-driven retraining or selection.
    selected_test_probs=[]

    for seed in FROZEN_CONFIG_MATRIX['Seeds']:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)

        model=build_model_matrix(
            X_fit[0].shape[1],
            X_fit[1].shape[1]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(1e-4),
            loss={
                'interaction':'binary_crossentropy',
                'triplet_output':triplet_loss_matrix(selected_margin)
            },
            loss_weights={
                'interaction':selected_bce,
                'triplet_output':selected_trip
            },
            metrics={'interaction':['accuracy']}
        )

        model.load_weights(
            weight_dir/selected_name/f'seed_{seed}.weights.h5'
        )

        tp=model.predict(X_test,verbose=0)[0].reshape(-1)
        selected_test_probs.append(tp)

        del model
        keras.backend.clear_session()
        gc.collect()

    ensemble_test=np.mean(
        np.stack(selected_test_probs),axis=0
    )

    test_metrics=metrics_matrix(
        y_test_label,
        ensemble_test,
        selected_threshold
    )

    pd.DataFrame({
        'Class':y_test_label,
        'Probability':ensemble_test,
        'Prediction':(
            ensemble_test>=selected_threshold
        ).astype(int)
    }).to_csv(pred_file,index=False)

    result={
        'Experiment':(
            f'{ENCODER_LABELS_MATRIX[encoder_key]} | '
            f'{PROTOCOL_LABELS_MATRIX[protocol_key]}'
        ),
        'Encoder_key':encoder_key,
        'Protocol_key':protocol_key,
        'Selection_policy':(
            'Hyperparameters and threshold selected exclusively on '
            'protein-cluster-disjoint Validation; Independent Test '
            'evaluated only after selection.'
        ),
        'Selected_configuration':{
            'Name':selected_name,
            'BCE_weight':selected_bce,
            'Triplet_weight':selected_trip,
            'Margin':selected_margin,
            'Threshold':selected_threshold,
        },
        'Validation_selection_metric_order':[
            'AUPR','ROC_AUC','MCC','F1'
        ],
        'Selected_Validation':best_record['Validation'],
        'Independent_Test':test_metrics,
        'Validation_cluster_audit':val_cluster_audit,
        'Three_way_cluster_overlap':{
            'fit_vs_validation':0,
            'fit_vs_test':0,
            'validation_vs_test':0
        },
        'Sampling_audit':sampling_audit,
        'All_validation_search_results':search_records,
        'Test_used_for_model_selection':False,
        'Test_used_for_threshold_selection':False,
    }

    result_file.write_text(
        json.dumps(result,indent=2),
        encoding='utf-8'
    )

    zip_path=Path(
        shutil.make_archive(str(out),'zip',root_dir=out)
    )

    print('\nFINAL RESULT',json.dumps(result,indent=2),flush=True)
    print('DOWNLOAD',zip_path,flush=True)

    display(FileLink(str(result_file)))
    display(FileLink(str(pred_file)))
    display(FileLink(str(out/'validation_model_selection.csv')))
    display(FileLink(str(zip_path)))

    return result


print('\nSETUP COMPLETE — NO EXPERIMENT TRAINED IN THIS CELL')
print('Encoders:',ENCODER_LABELS_MATRIX)
print('Protocols:',PROTOCOL_LABELS_MATRIX)
print('Every following experiment cell calls run_frozen_matrix_experiment(encoder_key, protocol_key).')


# CONTROLLED VALIDATION-ONLY HYPERPARAMETER SEARCH + FINAL INDEPENDENT TEST
result_db2_esm650_cluster_matched = run_frozen_matrix_experiment('db2_esm650', 'cluster0p5_matched_length')

print('\nREFERENCE RESULT FROM ORIGINAL NOTEBOOK (reproduction check only):')
print({'Threshold':0.510,'Accuracy':0.708029197080292,'Precision':0.665377,'Recall':0.836983,'F1':0.741379,'MCC':0.430627,'ROC_AUC':0.788268,'AUPR':0.751953})
