#!/usr/bin/env python3
"""Validate/rebuild frozen 0train epochs and build deterministic boundaries."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from src.data import _validate_and_augment_metadata, sha256_file
from src.adaptive_segmentation_v0 import (probe_sequence, adaptive_boundaries,
    valid_random_boundaries, matched_length_permutations, oas_cov)

LEGACY = yaml.safe_load((ROOT/'configs/bnci2014_001_5window.yaml').read_text())
TRJ = yaml.safe_load((ROOT/'configs/bnci2014_001_trajectory_v0.yaml').read_text())
NEW = yaml.safe_load((ROOT/'configs/bnci2014_001_adaptive_segmentation_v0.yaml').read_text())
OUT = ROOT/'outputs/bnci2014_001_adaptive_segmentation_v0'

def arrhash(a):
    a=np.ascontiguousarray(a); h=hashlib.sha256()
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def prepare_if_needed():
    cache=ROOT/'cache/bnci2014_001/prepared_epochs.npz'
    expected=TRJ['v1_inputs']['prepared_epochs']
    meta_path=ROOT/TRJ['v1_inputs']['prepared_metadata']['path']
    if cache.exists():
        if sha256_file(cache)!=expected['file_sha256']:
            raise RuntimeError('existing prepared_epochs cache hash mismatch; refusing rewrite')
        if not meta_path.exists() or sha256_file(meta_path)!=TRJ['v1_inputs']['prepared_metadata']['sha256']:
            raise RuntimeError('prepared metadata missing/hash mismatch; cache not trusted')
        z=np.load(cache,allow_pickle=False)
        if tuple(z.files)!=('X','y','channel_names','sampling_frequency_hz'):
            raise RuntimeError('prepared epoch cache key contract mismatch')
        X=np.asarray(z['X']); y=np.asarray(z['y']).astype(str); channels=np.asarray(z['channel_names']).astype(str)
        sfreq=float(z['sampling_frequency_hz'])
        if arrhash(X)!=expected['x_content_sha256']:
            raise RuntimeError('prepared epoch array content hash mismatch')
        metadata=pd.read_csv(meta_path,keep_default_na=False)
        source='existing cache; read-only'
    else:
        # Reapply only the original 5window config, but deliberately do not run
        # its inventory query, which opens session 1test metadata.
        moabb_dir=ROOT/'cache/moabb_data'; moabb_dir.mkdir(parents=True,exist_ok=True)
        os.environ['MNE_DATA']=str(moabb_dir); os.environ['MNE_DATASETS_BNCI_PATH']=str(moabb_dir)
        import mne
        from moabb.datasets import BNCI2014_001
        from moabb.datasets.bnci.base import data_path
        from moabb.datasets.bnci.bnci_2014 import _convert_mi
        from moabb.datasets.bnci import base as bnci_base
        from moabb.paradigms import MotorImagery
        mne.set_log_level('WARNING')
        ds=BNCI2014_001(subjects=list(range(1,10)),sessions=['0train'])
        prep=LEGACY['preprocessing']; dat=LEGACY['dataset']
        paradigm=MotorImagery(n_classes=4,events=dat['classes'],fmin=8.0,fmax=32.0,
            tmin=0.0,tmax=3.996,baseline=None,channels=dat['eeg_channels'],resample=None)
        # MOABB's built-in BNCI loader opens both AxxT and AxxE inside each
        # subject loader before applying the sessions argument. Resolve only
        # AxxT files and run the unchanged MOABB converters/paradigm pipeline.
        ch_names=['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz','EOG1','EOG2','EOG3']
        ch_types=['eeg']*22+['eog']*3
        process=paradigm.make_process_pipelines(ds,return_epochs=True)[0]
        label_pipeline=paradigm.make_labels_pipeline(ds,return_epochs=True)
        epochs_list=[]; labels_list=[]; meta_parts=[]; train_file_hashes={}
        for subject in range(1,10):
            url=f'{bnci_base.BNCI_URL}001-2014/A{subject:02d}T.mat'
            train_file=data_path(url,path=str(moabb_dir),verbose='WARNING')[0]
            train_file_hashes[str(subject)]=sha256_file(train_file)
            runs,_=_convert_mi(train_file,ch_names,ch_types,dataset_code='BNCI2014-001',subject_id=subject)
            for run_idx,raw in enumerate(runs):
                ep=process.transform(raw)
                if ep is None or len(ep)==0:continue
                lab=np.asarray(label_pipeline.transform(ep)).astype(str)
                epochs_list.append(ep);labels_list.append(lab)
                meta_parts.append(pd.DataFrame({'subject':subject,'session':'0train','run':str(run_idx)},index=np.arange(len(lab))))
        import mne
        epochs=mne.concatenate_epochs(epochs_list)
        labels=np.concatenate(labels_list);raw_meta=pd.concat(meta_parts,ignore_index=True)
        if epochs.ch_names!=dat['eeg_channels']: epochs.reorder_channels(dat['eeg_channels'])
        if epochs.get_channel_types()!=['eeg']*22: raise RuntimeError('unexpected channel type')
        if float(epochs.info['sfreq'])!=250.0: raise RuntimeError('unexpected sample rate')
        X=epochs.get_data(copy=True).astype(np.float32,copy=False); y=np.asarray(labels).astype(str)
        channels=np.asarray(dat['eeg_channels'],dtype=str); sfreq=250.0
        metadata=_validate_and_augment_metadata(raw_meta,y,list(range(1,10)),'0train',dat['classes'])
        if arrhash(X)!=expected['x_content_sha256']:
            raise RuntimeError('recomputed frozen X content hash mismatch')
        metadata_hash=hashlib.sha256(metadata.to_csv(index=False).encode()).hexdigest()
        if metadata_hash!=TRJ['v1_inputs']['prepared_metadata']['sha256']:
            raise RuntimeError('recomputed frozen trial metadata hash mismatch')
        source='regenerated with original MotorImagery 0train-only preprocessing; forbidden 1test was never queried'
    if X.shape!=(2592,22,1000) or X.dtype!=np.float32 or sfreq!=250.0:
        raise RuntimeError(f'dataset shape/dtype/sfreq gate failed: {X.shape}/{X.dtype}/{sfreq}')
    if tuple(channels)!=tuple(LEGACY['dataset']['eeg_channels']) or set(y)!=set(LEGACY['dataset']['classes']):
        raise RuntimeError('frozen channels/classes mismatch')
    if len(metadata)!=2592 or sorted(metadata['subject'].astype(int).unique())!=list(range(1,10)):
        raise RuntimeError('frozen subject/trial-count metadata mismatch')
    if set(metadata['session'].astype(str))!={'0train'}: raise RuntimeError('session barrier failed')
    for name in ('protocol','arrays','tables','figures','decisions','report'):
        OUT.joinpath(name).mkdir(parents=True,exist_ok=True)
    if source.startswith('regenerated'):
        np.savez(OUT/'arrays/prepared_epochs.npz',X=X,y=y,channel_names=channels,
                 sampling_frequency_hz=np.asarray(sfreq))
        metadata.to_csv(OUT/'protocol/prepared_metadata.csv',index=False)
    else:
        # Validated source is never copied or written.
        metadata=metadata.copy()
    info={'source':source,'shape':list(X.shape),'dtype':str(X.dtype),'sfreq_hz':sfreq,
          'sessions_observed':['0train'],'channel_names':channels.tolist(),
          'prepared_file_sha256':sha256_file(cache) if cache.exists() else None,
          'X_content_sha256':arrhash(X),'trial_metadata_sha256':hashlib.sha256(metadata.to_csv(index=False).encode()).hexdigest()}
    if 'train_file_hashes' in locals(): info['only_downloaded_training_mat_sha256_by_subject']=train_file_hashes
    (OUT/'protocol/data_provenance.json').write_text(json.dumps(info,indent=2)+'\n')
    return X,y,metadata

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--force',action='store_true');args=parser.parse_args()
    if (OUT/'arrays/boundaries_v0.npz').exists() and not args.force: raise FileExistsError('output exists; refusing overwrite')
    X,y,meta=prepare_if_needed(); records=[]; adaptive=[]; randoms=[]; matched=[]; matched_counts=[]; boundary_stats=[]
    classes=LEGACY['dataset']['classes']; trial_idx=meta['sample_index'].to_numpy(dtype=int)
    for i,x in enumerate(X):
        cov,z,centers=probe_sequence(x,125,25)
        b=adaptive_boundaries(z,centers,1000,5,25,125)
        rngseed=int(np.random.SeedSequence([NEW['protocol']['seed'],i]).generate_state(1)[0])
        rb=valid_random_boundaries(seed=rngseed,reps=20)
        mb=matched_length_permutations(b,seed=rngseed+1,max_reps=20)
        if b[0]!=0 or b[-1]!=1000 or np.min(np.diff(b))<125: raise RuntimeError('adaptive boundary coverage/min-length failure')
        if not np.array_equal(np.sort(np.diff(mb,axis=1),axis=1),np.sort(np.diff(b))[None,:].repeat(len(mb),axis=0)):
            raise RuntimeError('matched-length control failed')
        adaptive.append(b);randoms.append(rb);matched.append(mb);matched_counts.append(len(mb))
        records.append({'sample_index':int(trial_idx[i]),'subject':int(meta.iloc[i]['subject']),
            'trial_id':int(meta.iloc[i]['trial_id']),'class_label':str(y[i]),
            **{f'b{i}':int(v) for i,v in enumerate(b)},
            **{f'len{i}':int(v) for i,v in enumerate(np.diff(b))}})
        # Boundary versus matched valid non-boundary positions, exact width 125.
        valid=np.arange(125,876,25); valid=np.asarray([v for v in valid if v not in set(b[1:-1])])
        rg=np.random.default_rng(rngseed+2)
        if len(valid)<4: raise RuntimeError('insufficient non-boundary controls')
        controls=rg.choice(valid,size=4,replace=False)
        db=[oas_cov(x[:,t-125:t]) for t in b[1:-1]]
        da=[oas_cov(x[:,t:t+125]) for t in b[1:-1]]
        cr=[oas_cov(x[:,t-125:t]) for t in controls]; cd=[oas_cov(x[:,t:t+125]) for t in controls]
        from src.adaptive_segmentation_v0 import airm_distance
        boundary_stats.append({'sample_index':int(i),'subject':int(meta.iloc[i]['subject']),
            'boundary_airm':float(np.mean([airm_distance(a,c) for a,c in zip(db,da)])),
            'random_airm':float(np.mean([airm_distance(a,c) for a,c in zip(cr,cd)])),
            'within_airm':float(np.mean([airm_distance(oas_cov(x[:,a:(a+z)//2]),oas_cov(x[:,(a+z)//2:z])) for a,z in zip(b[:-1],b[1:])])),
            'across_airm':float(np.mean([airm_distance(oas_cov(x[:,a:z]),oas_cov(x[:,z:q])) for a,z,q in zip(b[:-2],b[1:-1],b[2:])]))})
        if i%100==0: print(f'boundary {i}/{len(X)}',flush=True)
    matched_padded=np.full((len(X),20,6),-1,dtype=np.int64)
    for i,mb in enumerate(matched): matched_padded[i,:len(mb)]=mb
    np.savez_compressed(OUT/'arrays/boundaries_v0.npz',adaptive=np.stack(adaptive),random=np.stack(randoms),matched=matched_padded,matched_counts=np.asarray(matched_counts),
        sample_index=trial_idx,subjects=meta['subject'].to_numpy(),labels=y.astype(str))
    for i,record in enumerate(records): record['matched_replicates']=matched_counts[i]
    pd.DataFrame(records).to_csv(OUT/'tables/boundaries_by_trial.csv',index=False)
    pd.DataFrame(boundary_stats).to_csv(OUT/'tables/boundary_validity_by_trial.csv',index=False)
    print(json.dumps({'trials':len(X),'adaptive_shape':list(np.stack(adaptive).shape),'random_shape':list(np.stack(randoms).shape),'matched_shape':list(matched_padded.shape),'matched_replicates_min':min(matched_counts),'matched_replicates_max':max(matched_counts),'dataset_source':json.loads((OUT/'protocol/data_provenance.json').read_text())['source']}))

if __name__=='__main__': main()
