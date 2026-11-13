from __future__ import print_function
import sys as _path_sys
from pathlib import Path as _Path
for _parent in _Path(__file__).resolve().parents:
    if (_parent / 'workflow_paths.py').is_file():
        _path_sys.path.insert(0, str(_parent))
        break
from workflow_paths import workspace_path
import argparse,json,sys,hashlib,inspect
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,workspace_path('revision/04_analysis'))
from loop05_corrected_common import read_scenario,model_features
OUT=Path(workspace_path('revision/04_analysis/round2_20261019'))
parser=argparse.ArgumentParser();parser.add_argument('model');args=parser.parse_args()
name=args.model;stem=name.lower();root=Path(workspace_path('revision/04_analysis/loop05_corrected_common5fold/full519'))
df,folds,cfg=read_scenario('full519');features=model_features(name,'full')
import shap
result={'model':name,'shap_version':shap.__version__,'features':features,'checks':[]}
if name=='MVP':
 import os,tensorflow as tf,keras
 from keras import backend as K
 from sklearn.preprocessing import MinMaxScaler
 sys.path.insert(0,workspace_path('thesis/models/missense/code'))
 from models import CNN_Model
 result.update({'keras':keras.__version__,'tensorflow':tf.__version__})
for k in range(5):
 train=folds['fold'].values!=k;valid=~train
 med=pd.read_csv(str(root/'preprocess'/('{}{}_median.csv'.format(stem+'_fold',k)))).set_index('feature')['train_median'].reindex(features)
 X=df.loc[valid,features].apply(pd.to_numeric,errors='coerce').fillna(med)
 saved=pd.read_csv(str(root/'shap'/('{}{}_shap_values.csv'.format(stem+'_fold',k))))
 assert list(saved.columns[2:])==features
 assert np.array_equal(saved.row_id.values,folds.loc[valid,'row_id'].values)
 if name=='VARITY':
  from xgboost import XGBClassifier
  model=XGBClassifier();model.load_model(str(root/'models'/('varity_fold{}.model'.format(k))))
  ex=shap.TreeExplainer(model);sv=ex.shap_values(X)
  target=model.predict(X,output_margin=True);prob=model.predict_proba(X)[:,1]
  result['output_scale']='raw margin (log-odds)';result['expected_value_unit']='log-odds'
 elif name=='MutScore':
  import joblib
  model=joblib.load(str(root/'models'/('mutscore_fold{}.pkl'.format(k))))
  ex=shap.TreeExplainer(model);allsv=ex.shap_values(X)
  sv=allsv[:,:,1] if np.asarray(allsv).ndim==3 else allsv[1]
  target=model.predict_proba(X)[:,1];prob=target
  result['output_scale']='positive-class probability';result['expected_value_unit']='probability'
 else:
  K.clear_session()
  scaler=pd.read_csv(str(root/'preprocess'/('mvp_fold{}_minmax_scaler.csv'.format(k)))).set_index('feature').loc[features]
  Xs=(X.values*scaler.scale.values+scaler.min_offset.values).reshape((-1,len(features),1,1))
  model=CNN_Model(input_shape=(len(features),1,1),weights_path=None,nb_epoch=20,batch_size=16)
  model._init_model(verbose=False)
  K.get_session().run(tf.global_variables_initializer());model.model.load_weights(str(root/'models'/('mvp_fold{}.h5'.format(k))))
  bg=pd.read_csv(str(root/'preprocess'/('mvp_fold{}_shap_background_rows.csv'.format(k))))
  Xbg=df.loc[bg.row_id.values,features].apply(pd.to_numeric,errors='coerce').fillna(med)
  Xbg=(Xbg.values*scaler.scale.values+scaler.min_offset.values).reshape((-1,len(features),1,1))
  ex=shap.DeepExplainer(model.model,Xbg,session=K.get_session())
  sv=np.asarray(ex.shap_values(Xs)[0]).reshape((len(Xs),len(features)))
  target=model.model.predict(Xs,batch_size=16)[:,0];prob=target
  result['output_scale']='sigmoid positive-class probability';result['expected_value_unit']='probability'
  if k==0:
   result['architecture']=[{'name':l.name,'class':l.__class__.__name__,'output_shape':str(l.output_shape),'parameters':l.count_params(),'config':l.get_config()} for l in model.model.layers]
   result['total_parameters']=model.model.count_params();result['optimizer']=model.model.optimizer.get_config();result['loss']=model.model.loss
   (OUT/'mvp_architecture.json').write_text(model.model.to_json(),encoding='utf-8')
 ev=np.asarray(ex.expected_value)
 base=float(ev.reshape(-1)[1 if name=='MutScore' else 0])
 recorded=pd.read_csv(str(root/'predictions'/('{}{}_predictions.csv'.format(stem+'_fold',k))))['prob_pathogenic'].values
 entry={'fold':k,'n':len(X),'expected_value':base,'saved_shap_max_abs_difference':float(np.max(np.abs(np.asarray(sv)-saved[features].values))),'additivity_max_abs_residual':float(np.max(np.abs(base+np.asarray(sv).sum(axis=1)-target))),'saved_probability_max_abs_difference':float(np.max(np.abs(prob-recorded)))}
 if name!='MVP':
  result['TreeExplainer_signature']=str(inspect.signature(shap.TreeExplainer))
  entry.update({'feature_perturbation':str(getattr(ex,'feature_perturbation',None)),'model_output':str(getattr(ex,'model_output',None)),'tree_output':str(getattr(ex.model,'tree_output',None)),'background_data_is_none':getattr(ex,'data',None) is None})
 else:
  entry['background_n']=len(bg);entry['background_all_training']=bool(np.all(np.isin(bg.row_id.values,folds.loc[train,'row_id'].values)))
 result['checks'].append(entry);print(name,k,entry,flush=True)
(OUT/(stem+'_runtime_audit.json')).write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
