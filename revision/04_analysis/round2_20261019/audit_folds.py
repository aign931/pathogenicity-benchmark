import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path
import sys,json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,workspace_path('revision/04_analysis'))
from loop05_corrected_common import read_scenario,model_features
root=Path(workspace_path('revision/04_analysis/loop05_corrected_common5fold'));out=Path(workspace_path('revision/04_analysis/round2_20261019'))
rows=[]
for scenario in ['full519','reduced519','palb2_526','expert_panel','leave_one_gene_out']:
 df,folds,cfg=read_scenario(scenario);base=root/scenario
 for name,stem in [('VARITY','varity'),('MutScore','mutscore'),('MVP','mvp')]:
  features=model_features(name,cfg['panel'])
  md=json.loads((base/'logs'/(stem+'_run_metadata.json')).read_text())
  assert md['model_features']==features
  assert hashlib.sha256(cfg['data'].read_bytes()).hexdigest()==md['data_sha256']
  assert hashlib.sha256(cfg['folds'].read_bytes()).hexdigest()==md['fold_sha256']
  metrics=pd.read_csv(base/'tables'/(stem+'_metrics.csv')).set_index('fold')
  validation=pd.read_csv(base/'validation'/(stem+'_reload_validation.csv'))
  for k in range(5):
   tr=folds.fold.values!=k;va=~tr;X=df[features].apply(pd.to_numeric,errors='coerce')
   saved=pd.read_csv(base/'preprocess'/(stem+'_fold%d_median.csv'%k))
   assert saved.feature.tolist()==features
   med=X.loc[tr].median();delta=float(np.max(np.abs(med.values-saved.train_median.values)))
   assert np.allclose(med.values,saved.train_median.values,rtol=1e-12,atol=1e-12)
   pred=pd.read_csv(base/'predictions'/(stem+'_fold%d_predictions.csv'%k))
   assert np.array_equal(pred.row_id.values,folds.loc[va,'row_id'].values)
   assert len(pred)==va.sum() and metrics.loc[k,'n_fit']==tr.sum()==metrics.loc[k,'n_outer_train']
   shap=pd.read_csv(base/'shap'/(stem+'_fold%d_shap_values.csv'%k))
   assert np.array_equal(shap.row_id.values,pred.row_id.values)
   assert shap.columns[2:].tolist()==features
   v=validation[validation.fold==k].iloc[0]
   passing=[c for c in validation.columns if c.startswith('pass_')]
   assert all(bool(v[c]) for c in passing)
   rec={'scenario':scenario,'model':name,'fold':k,'n_train':int(tr.sum()),'n_valid':int(va.sum()),'feature_count':len(features),'median_max_abs_difference':delta,'prior_reload_pass':True}
   if name=='MVP':
    sc=pd.read_csv(base/'preprocess'/('mvp_fold%d_minmax_scaler.csv'%k));T=X.loc[tr].fillna(med)
    assert sc.feature.tolist()==features
    assert np.allclose(T.min().values,sc.data_min.values,rtol=1e-12,atol=1e-12)
    assert np.allclose(T.max().values,sc.data_max.values,rtol=1e-12,atol=1e-12)
    bg=pd.read_csv(base/'preprocess'/('mvp_fold%d_shap_background_rows.csv'%k))
    expected=np.random.RandomState(2025).choice(int(tr.sum()),min(100,int(tr.sum())),replace=False)
    assert np.array_equal(bg.training_array_index.values,expected)
    assert np.array_equal(bg.row_id.values,folds.loc[tr,'row_id'].values[expected])
    rec['background_n']=len(bg);rec['scaler_and_background_pass']=True
   rows.append(rec)
pd.DataFrame(rows).to_csv(out/'fold_preprocessing_audit.csv',index=False)
full=model_features('VARITY','full');mvp=model_features('MVP','full');red=model_features('MVP','base_annotation')
pd.DataFrame({'position_1_based':range(1,43),'feature':mvp}).to_csv(out/'mvp_feature_order_42.csv',index=False)
pd.DataFrame({'position_1_based':range(1,20),'feature':red}).to_csv(out/'reduced_feature_order_19.csv',index=False)
print('PASS: %d model-fold checks; 75 medians; 25 scalers/background sets; row/feature alignment and prior reload checks'%len(rows))
print(pd.DataFrame(rows).groupby(['scenario','model']).agg(n_train_min=('n_train','min'),n_train_max=('n_train','max'),n_valid_total=('n_valid','sum')).to_string())
