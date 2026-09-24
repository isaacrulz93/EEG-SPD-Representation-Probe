#!/usr/bin/env python3
"""Run the preregistered nine-subject LOSO adaptive-segmentation falsification."""
from __future__ import annotations
import hashlib, json, platform, subprocess, sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from src.data import sha256_file
from src.adaptive_segmentation_v0 import probe_sequence, airm_distance, oas_cov, exact_sign_flip
from src.evaluation_adaptive_segmentation_v0 import trial_features,fit_score,metric_row,paired_table
from sklearn.metrics import balanced_accuracy_score

OUT=ROOT/'outputs/bnci2014_001_adaptive_segmentation_v0'
CLASSES=tuple(yaml.safe_load((ROOT/'configs/bnci2014_001_5window.yaml').read_text())['dataset']['classes'])

def read_data():
    cache=ROOT/'cache/bnci2014_001/prepared_epochs.npz'
    meta_path=ROOT/'cache/bnci2014_001/prepared_metadata.csv'
    if cache.exists():
        expect=yaml.safe_load((ROOT/'configs/bnci2014_001_trajectory_v0.yaml').read_text())['v1_inputs']['prepared_epochs']['file_sha256']
        if sha256_file(cache)!=expect: raise RuntimeError('prepared cache hash changed since boundary build')
        z=np.load(cache,allow_pickle=False);X=z['X'];y=z['y'].astype(str)
    else:
        p=OUT/'arrays/prepared_epochs.npz'
        if not p.exists():raise FileNotFoundError('neither verified cache nor regenerated 0train epochs exist')
        z=np.load(p,allow_pickle=False);X=z['X'];y=z['y'].astype(str)
        meta_path=OUT/'protocol/prepared_metadata.csv'
    meta=pd.read_csv(meta_path,keep_default_na=False)
    if X.shape!=(2592,22,1000) or set(meta['session'].astype(str))!={'0train'}:
        raise RuntimeError('final data contract failure')
    return X,y,meta

def score_one(boundaries,X,y,subjects,target):
    feats=trial_features(X,boundaries)
    train=np.flatnonzero(subjects!=target);test=np.flatnonzero(subjects==target)
    p,fit_audit=fit_score([feats[i] for i in train],y[train],[feats[i] for i in test],
                          [np.diff(boundaries[i]) for i in test])
    return test,p,fit_audit

def ensure_dirs():
    for name in ('protocol','tables','figures','arrays','decisions','report'):(OUT/name).mkdir(parents=True,exist_ok=True)

def main():
    start=time.time();ensure_dirs()
    if not (OUT/'arrays/boundaries_v0.npz').exists():
        subprocess.run([sys.executable,str(ROOT/'scripts/26_build_adaptive_segments_v0.py')],cwd=ROOT,check=True)
    X,y,meta=read_data();subjects=meta['subject'].to_numpy(dtype=int)
    class_map={name:i for i,name in enumerate(CLASSES)}; labels=np.asarray([class_map[v] for v in y],dtype=int)
    with np.load(OUT/'arrays/boundaries_v0.npz',allow_pickle=False) as z:
        adaptive=z['adaptive'];random=z['random'];matched=z['matched'];matched_counts=z['matched_counts']
        if not np.array_equal(z['sample_index'],meta['sample_index']) or not np.array_equal(z['subjects'],subjects) or not np.array_equal(z['labels'].astype(str),y):raise RuntimeError('boundary/data row alignment failure')
    conditions={'GLOBAL':np.tile(np.array([0,1000]),(len(y),1)),
                'FIXED5':np.tile(np.array([0,200,400,600,800,1000]),(len(y),1)),
                'ADAPT5':adaptive}
    subjects_sorted=sorted(np.unique(subjects)); rows=[]; scores={k:{s:None for s in subjects_sorted} for k in ('GLOBAL','FIXED5','ADAPT5','RANDOM5','RANDOM_MATCHED')}
    for condition,bounds in conditions.items():
        print(f'feature extraction {condition}',flush=True)
        features=trial_features(X,bounds)
        for target in subjects_sorted:
            test,p,audit=score_one_from_features(features,bounds,y,subjects,target)
            row={'condition':condition,'replicate':0,'target_subject':target,**metric_row(y[test],p,CLASSES),**audit}
            rows.append(row);scores[condition][target]=row['balanced_accuracy']
    for condition,bank,counts in (('RANDOM5',random,None),('RANDOM_MATCHED',matched,matched_counts)):
        nrep=20
        if counts is not None:nrep=int(np.min(counts))
        if nrep<1:raise RuntimeError('RANDOM_MATCHED has zero common non-identity permutations')
        for rep in range(nrep):
            bounds=bank[:,rep] if counts is None else bank[np.arange(len(bank)),rep % counts]
            features=trial_features(X,bounds)
            for target in subjects_sorted:
                test,p,audit=score_one_from_features(features,bounds,y,subjects,target)
                row={'condition':condition,'replicate':rep+1,'target_subject':target,**metric_row(y[test],p,CLASSES),**audit}
                rows.append(row)
        for target in subjects_sorted:
            scores[condition][target]=float(np.mean([r['balanced_accuracy'] for r in rows if r['condition']==condition and r['target_subject']==target]))
        print(f'completed {condition} {nrep} replicates',flush=True)
    table=pd.DataFrame(rows);table.to_csv(OUT/'tables/metrics_by_subject.csv',index=False)
    summary=[]
    for condition in scores:
        means=np.array([scores[condition][s] for s in subjects_sorted],dtype=float)
        reps=table.loc[table.condition==condition]
        summary.append({'condition':condition,'mean_loso_ba':float(means.mean()),'sd_subject_ba':float(means.std(ddof=1)),
            'accuracy_mean_subject':float(reps.groupby('target_subject').accuracy.mean().mean()),
            'macro_f1_mean_subject':float(reps.groupby('target_subject').macro_f1.mean().mean()),
            'replicates':int(reps.replicate.max()) if condition.startswith('RANDOM') else 1})
    pd.DataFrame(summary).to_csv(OUT/'tables/metrics_summary.csv',index=False)
    paired=paired_table(scores);pd.DataFrame([{k:v for k,v in p.items() if k!='subject_differences'} for p in paired]).to_csv(OUT/'tables/paired_comparisons.csv',index=False)
    (OUT/'tables/paired_subject_differences.json').write_text(json.dumps(paired,indent=2)+'\n')
    bvalid=pd.read_csv(OUT/'tables/boundary_validity_by_trial.csv')
    subj=bvalid.groupby('subject')[['boundary_airm','random_airm','within_airm','across_airm']].mean()
    boundary_test=exact_sign_flip((subj.boundary_airm-subj.random_airm).to_numpy())
    boundary_test.update({'comparison':'boundary_minus_valid_nonboundary_AIRM','subject_means':subj.reset_index().to_dict(orient='records')})
    subj.to_csv(OUT/'tables/boundary_validity_by_subject.csv')
    (OUT/'tables/boundary_validity_test.json').write_text(json.dumps(boundary_test,indent=2)+'\n')
    effect={p['comparison']:p for p in paired}
    go=(effect['ADAPT5_gt_FIXED5']['mean_difference']>0 and effect['ADAPT5_gt_FIXED5']['p_one_sided']<.05 and
        effect['ADAPT5_gt_RANDOM_MATCHED']['mean_difference']>0 and effect['ADAPT5_gt_RANDOM_MATCHED']['p_one_sided']<.05 and
        boundary_test['mean_difference']>0 and boundary_test['p_one_sided']<.05)
    terminal='GO_ADAPTIVE_BOUNDARY_V0' if go else 'STOP_ADAPTIVE_BOUNDARY_V0'
    (OUT/'decisions/terminal.json').write_text(json.dumps({'terminal':terminal,'gate':{'adapt5_gt_fixed5':effect['ADAPT5_gt_FIXED5'],'adapt5_gt_random_matched':effect['ADAPT5_gt_RANDOM_MATCHED'],'boundary_airm_gt_random':{k:v for k,v in boundary_test.items() if k!='subject_means'}},'result_adaptation':False},indent=2)+'\n')
    env={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'pandas':pd.__version__}
    import sklearn;env['scikit_learn']=sklearn.__version__
    (OUT/'protocol/environment.json').write_text(json.dumps(env,indent=2)+'\n')
    try:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except Exception:commit='unknown'
    prov={'implementation_commit':commit,'config_sha256':sha256_file(ROOT/'configs/bnci2014_001_adaptive_segmentation_v0.yaml'),
          'protocol_sha256':sha256_file(ROOT/'docs/PROTOCOL_ADAPTIVE_SEGMENTATION_V0.md'),
          'rows':2592,'subjects':9,'session':'0train','target_session_accessed':False,'LOSO_folds':9,
          'start_unix':start,'finish_unix':time.time(),'elapsed_seconds':time.time()-start}
    (OUT/'protocol/provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
    make_figures(X,y,meta,adaptive,scores,bvalid,subj)
    report(terminal,summary,paired,boundary_test,rows,prov)
    print(json.dumps({'terminal':terminal,'summary':summary,'paired':paired,'boundary_test':boundary_test,'elapsed_seconds':time.time()-start},indent=2))

def score_one_from_features(features,bounds,y,subjects,target):
    train=np.flatnonzero(subjects!=target);test=np.flatnonzero(subjects==target)
    p,audit=fit_score([features[i] for i in train],y[train],[features[i] for i in test],
        [np.diff(bounds[i]) for i in test])
    return test,p,audit

def make_figures(X,y,meta,adaptive,scores,bvalid,subj):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.adaptive_segmentation_v0 import probe_sequence
    fig,axs=plt.subplots(4,1,figsize=(12,9),sharex=True)
    ordering=meta.assign(_cls=y).sort_values(['_cls','sample_index'])
    chosen=[int(ordering.loc[ordering._cls==c,'sample_index'].iloc[0]) for c in CLASSES]
    for ax,i in zip(axs,chosen):
        x=X[i];_,v,centers=probe_sequence(x)
        change=np.r_[0,np.linalg.norm(np.diff(v,axis=0),axis=1)]
        ax.plot(centers,change,color='black',lw=1);ax.set_ylabel(str(y[i]))
        for tau in adaptive[i,1:-1]:ax.axvline(tau,color='tab:red',alpha=.7)
    axs[-1].set_xlabel('sample index (250 Hz)');fig.tight_layout();fig.savefig(OUT/'figures/representative_boundaries.png',dpi=150);plt.close(fig)
    bnorm=adaptive[:,1:-1]/1000
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(bnorm.ravel(),bins=np.linspace(0,1,21));ax.set(xlabel='normalized trial time',ylabel='boundary count');fig.tight_layout();fig.savefig(OUT/'figures/boundary_time_histogram.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for c in CLASSES:
        mask=np.asarray(y)==c;ax.hist((adaptive[mask,1:-1]/1000).ravel(),bins=np.linspace(0,1,21),histtype='step',label=c)
    ax.legend();ax.set(xlabel='normalized trial time',ylabel='boundary count');fig.tight_layout();fig.savefig(OUT/'figures/boundary_time_by_class.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(np.diff(adaptive,axis=1).ravel(),bins=np.arange(100,701,25));ax.set(xlabel='segment length (samples)',ylabel='count');fig.tight_layout();fig.savefig(OUT/'figures/adaptive_segment_lengths.png',dpi=150);plt.close(fig)
    order=list(scores);vals=[[scores[k][s] for s in sorted(scores[k])] for k in order]
    fig,ax=plt.subplots(figsize=(10,5));ax.boxplot(vals,tick_labels=order,showmeans=True);ax.set_ylabel('target subject balanced accuracy');fig.tight_layout();fig.savefig(OUT/'figures/loso_ba_comparison.png',dpi=150);plt.close(fig)
    for other,stem in [('FIXED5','adapt_vs_fixed'),('RANDOM_MATCHED','adapt_vs_random_matched')]:
        fig,ax=plt.subplots(figsize=(6,6));a=np.array([scores['ADAPT5'][s] for s in sorted(scores['ADAPT5'])]);b=np.array([scores[other][s] for s in sorted(scores[other])]);ax.scatter(b,a)
        ax.plot([0,1],[0,1],ls='--',color='gray');ax.set(xlabel=other+' BA',ylabel='ADAPT5 BA',xlim=(0,1),ylim=(0,1));fig.tight_layout();fig.savefig(OUT/f'figures/{stem}.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.scatter(subj.random_airm,subj.boundary_airm);ax.plot([0,ax.get_xlim()[1]],[0,ax.get_xlim()[1]],ls='--',color='gray');ax.set(xlabel='matched random AIRM',ylabel='adaptive boundary AIRM');fig.tight_layout();fig.savefig(OUT/'figures/boundary_airm_vs_random.png',dpi=150);plt.close(fig)

def report(terminal,summary,paired,boundary,rows,prov):
    d={r['condition']:r for r in summary}
    lines=['# Adaptive Covariance Segmentation falsification v0','',f"**Decision: `{terminal}`**",'',
        '## LOSO balanced accuracy (mean ± subject SD)','', '| Condition | LOSO BA | Replicates |','|---|---:|---:|']
    for name in ('GLOBAL','FIXED5','ADAPT5','RANDOM5','RANDOM_MATCHED'):
        r=d[name];lines.append(f"| {name} | {r['mean_loso_ba']:.4f} ± {r['sd_subject_ba']:.4f} | {r['replicates']} |")
    lines += ['', '## Paired exact sign-flip tests','', '| Comparison (first minus second) | Mean Δ BA | Median Δ BA | one-sided p |','|---|---:|---:|---:|']
    for p in paired:lines.append(f"| {p['comparison']} | {p['mean_difference']:.4f} | {p['median_difference']:.4f} | {p['p_one_sided']:.5f} |")
    lines += ['',f"Boundary AIRM: mean adaptive-minus-random subject effect {boundary['mean_difference']:.4f}, median {boundary['median_difference']:.4f}, exact one-sided p={boundary['p_one_sided']:.5f}.",
        '',f"Subjects=9; folds=9; trials=2592; implementation commit={prov['implementation_commit']}; elapsed={prov['elapsed_seconds']/60:.1f} min.",
        '', 'The trial-local probe is used only to select boundaries. OAS covariance is recomputed from each raw segment for the source-only classifier. Training standardization and multinomial logistic regression are fit within each outer source fold. Segment test probabilities are averaged with duration weights.',
        '', 'This exploratory falsification supports only the fixed BNCI2014_001 0train LOSO protocol and the preregistered boundary rule. No neural model is present. The decision is a deterministic engineering gate, not a population-level causal claim.',
        '', 'See subject tables, paired differences, boundary-validity tables and prespecified figures in this output directory.']
    (OUT/'report/adaptive_segmentation_v0.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    try:main()
    except Exception as exc:
        ensure_dirs()
        (OUT/'decisions/terminal.json').write_text(json.dumps({'terminal':'UNASSESSED_ADAPTIVE_BOUNDARY_V0','error_type':type(exc).__name__,'error':str(exc)},indent=2)+'\n')
        raise
