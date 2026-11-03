#!/usr/bin/env python3
"""Evaluate LLM SARIMA-class predictions in isolated experiment folders.

Run from project root, e.g.:
  python compare_sarima_llm.py --experiment features_30
  python compare_sarima_llm.py --experiment features_31
  python compare_sarima_llm.py --experiment raw_series --predictions sarima_llm_benchmark/predictions_raw_series.csv
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix)
from scipy.stats import binomtest, chi2

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / 'sarima_llm_benchmark'
KEY_CANDIDATES = [BENCH/'private_evaluation'/'answer_key_and_baselines.csv',
                  ROOT/'sarima_llm_private_evaluation'/'answer_key_and_baselines.csv']
CONFIG = {
 'features_30': {'default_prediction': BENCH/'predictions_features_30.csv', 'nn_col':'neural_class_id_features30', 'nn_label':'PyTorch MLP (30 features)'},
 'features_31': {'default_prediction': BENCH/'predictions_features_31.csv', 'nn_col':'neural_class_id_features31', 'nn_label':'PyTorch MLP (31 features)'},
 'raw_series': {'default_prediction': BENCH/'predictions_raw_series.csv', 'nn_col':'neural_class_id_features30', 'nn_label':'PyTorch MLP (30 features; reference only)'},
}

def args_parse():
 p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument('--experiment', choices=CONFIG, required=True, help='Experiment determines its own output folder and default prediction filename.')
 p.add_argument('--predictions', type=Path, help='Optional explicit LLM predictions CSV path.')
 p.add_argument('--key', type=Path, help='Optional private answer key path; auto-detected otherwise.')
 p.add_argument('--output-root', type=Path, default=Path('sarima_llm_benchmark/results'), help='Root for separated experiment outputs.')
 return p.parse_args()

def resolve(p):
 if p is None: return None
 return p if p.is_absolute() else ROOT/p

def first_file(paths): return next((x for x in paths if x.is_file()), None)

def paired_test(y, a, b, label):
 aok=np.asarray(a)==np.asarray(y); bok=np.asarray(b)==np.asarray(y)
 n10=int(np.sum(aok & ~bok)); n01=int(np.sum(~aok & bok)); n=n10+n01
 pv=float(binomtest(min(n10,n01),n,p=.5).pvalue) if n else 1.0
 stat=((abs(n10-n01)-1)**2/n) if n else 0.0
 return {'comparison':label,'llm_correct_baseline_wrong':n10,'llm_wrong_baseline_correct':n01,
         'discordant_pairs':n,'mcnemar_exact_p':pv,'mcnemar_chi2_cc_statistic':stat,
         'mcnemar_chi2_cc_p':float(chi2.sf(stat,1)) if n else 1.0}

def metric(name,y,p):
 return {'method':name,'n':len(y),'accuracy':accuracy_score(y,p),
  'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),
  'weighted_f1':f1_score(y,p,average='weighted',zero_division=0)}

def main():
 a=args_parse(); cfg=CONFIG[a.experiment]
 pred_path=resolve(a.predictions) or cfg['default_prediction']
 if not pred_path.is_file():
  raise FileNotFoundError(f"LLM predictions not found: {pred_path}\nSave predictions there or pass --predictions.")
 key_path=resolve(a.key) or first_file(KEY_CANDIDATES)
 if key_path is None: raise FileNotFoundError('Private answer_key_and_baselines.csv not found in benchmark/private_evaluation or sarima_llm_private_evaluation.')
 outroot=resolve(a.output_root) or ROOT/a.output_root
 out=outroot/a.experiment; out.mkdir(parents=True,exist_ok=True)
 pred=pd.read_csv(pred_path); key=pd.read_csv(key_path)
 for name,df in [('LLM predictions',pred),('private key',key)]:
  if not {'series_id'}.issubset(df.columns): raise ValueError(f'{name} must contain series_id')
  if df.series_id.isna().any() or df.series_id.astype(str).duplicated().any(): raise ValueError(f'{name} has missing or duplicate series_id values')
 needed={'series_id','true_class_id',cfg['nn_col']}
 if not needed.issubset(key.columns): raise ValueError(f'Private key missing columns: {sorted(needed-set(key.columns))}')
 if not {'predicted_class_id'}.issubset(pred.columns): raise ValueError('LLM CSV must contain predicted_class_id')
 pred=pred.copy(); key=key.copy(); pred['series_id']=pred.series_id.astype(str); key['series_id']=key.series_id.astype(str)
 pids,kids=set(pred.series_id),set(key.series_id)
 if pids!=kids: raise ValueError(f'ID mismatch: missing={len(kids-pids)} {sorted(kids-pids)[:5]}, extra={len(pids-kids)} {sorted(pids-kids)[:5]}')
 df=key.merge(pred,on='series_id',validate='one_to_one')
 for c in ['true_class_id',cfg['nn_col'],'predicted_class_id']:
  df[c]=pd.to_numeric(df[c],errors='raise').astype(int)
 for c in ['true_class_id',cfg['nn_col'],'predicted_class_id']:
  if not df[c].between(0,41).all(): raise ValueError(f'{c} contains class IDs outside 0..41')
 y=df.true_class_id.to_numpy(); llm=df.predicted_class_id.to_numpy(); nn=df[cfg['nn_col']].to_numpy()
 rows=[metric(f"LLM ({a.experiment})",y,llm),metric(cfg['nn_label'],y,nn)]
 if 'bic_class_id' in df and df.bic_class_id.notna().all():
  bic=pd.to_numeric(df.bic_class_id,errors='raise').astype(int).to_numpy(); rows.append(metric('Minimum BIC',y,bic))
 pd.DataFrame(rows).sort_values('accuracy',ascending=False).to_csv(out/'metrics_summary.csv',index=False)
 labels=np.arange(42)
 names=None
 if 'true_specification' in df:
  names=df[['true_class_id','true_specification']].drop_duplicates().set_index('true_class_id').true_specification.to_dict()
 target=[f'{i}: {names.get(i, f"class_{i}")}' if names else f'class_{i}' for i in labels]
 for arr,label,stem in [(llm,'LLM','llm'),(nn,cfg['nn_label'].replace(' ','_').lower(),'pytorch_baseline')]:
  pd.DataFrame(classification_report(y,arr,labels=labels,target_names=target,output_dict=True,zero_division=0)).T.to_csv(out/f'{stem}_classification_report.csv')
  pd.DataFrame(confusion_matrix(y,arr,labels=labels),index=target,columns=target).to_csv(out/f'{stem}_confusion_matrix.csv')
 pd.DataFrame([paired_test(y,llm,nn,f'LLM ({a.experiment}) vs {cfg["nn_label"]}')]).to_csv(out/'mcnemar_llm_vs_pytorch.csv',index=False)
 df['llm_correct']=llm==y; df['baseline_correct']=nn==y
 if 'confidence' in df:
  df['confidence']=pd.to_numeric(df.confidence,errors='coerce')
  if df.confidence.notna().any():
   bins=[-np.inf,.5,.6,.7,.8,.9,1.0000001]
   df['confidence_bin']=pd.cut(df.confidence,bins=bins,labels=['<=0.50','0.50-0.60','0.60-0.70','0.70-0.80','0.80-0.90','0.90-1.00'])
   df.groupby('confidence_bin',observed=False).agg(n=('llm_correct','size'),mean_confidence=('confidence','mean'),empirical_accuracy=('llm_correct','mean')).reset_index().to_csv(out/'llm_confidence_bins.csv',index=False)
 df.to_csv(out/'per_object_comparison.csv',index=False)
 if 'true_specification' in df:
  family=np.where(df.true_specification.astype(str).str.startswith('SARIMA'),'SARIMA','ARIMA')
  fam=[]
  for f in ['ARIMA','SARIMA']:
   m=family==f
   for label,arr in [('LLM',llm),(cfg['nn_label'],nn)]:
    fam.append({'family':f,'method':label,'n':int(m.sum()),'accuracy':accuracy_score(y[m],arr[m]) if m.any() else np.nan,
     'macro_f1':f1_score(y[m],arr[m],average='macro',zero_division=0) if m.any() else np.nan})
  pd.DataFrame(fam).to_csv(out/'metrics_by_family.csv',index=False)
 print(f'Experiment: {a.experiment}\nPredictions: {pred_path}\nPrivate key: {key_path}\nValidated {len(df)} unique test rows.')
 print(f'Results: {out.resolve()}')
 if a.experiment=='raw_series': print('NOTE: PyTorch baseline uses engineered features; treat this as a reference, not a same-input comparison.')
 print(pd.read_csv(out/'metrics_summary.csv').to_string(index=False))
if __name__=='__main__':
 try: main()
 except Exception as e: print(f'ERROR: {e}',file=sys.stderr); sys.exit(2)
