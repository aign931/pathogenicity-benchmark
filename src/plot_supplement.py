import sys,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import shap
from common import read_scenario,common_features,model_features
from summarize_models import correlation_components
from paths import RESULTS_DIR
ROOT=RESULTS_DIR/'full519'
OUT=RESULTS_DIR/'supplement';FIG=OUT/'figures';FIG.mkdir(parents=True,exist_ok=True)
df,folds,cfg=read_scenario('full519');features=common_features('full')
corr,edges,groups,mapping=correlation_components(df,features)
saved=pd.read_csv(ROOT/'tables/common_feature_spearman_correlation.csv',index_col=0)
assert np.allclose(corr.values,saved.loc[features,features].values,rtol=0,atol=1e-12)
oldgroups=pd.read_csv(ROOT/'tables/correlation_groups.csv')
assert groups[['group_id','members']].equals(oldgroups[['group_id','members']])
order=[x for m in groups.members for x in m.split(' | ')]
assert len(order)==len(set(order))==41
corr.loc[order,order].to_csv(OUT/'correlation_matrix_ordered.csv')
pd.DataFrame({'position':range(1,42),'group':[mapping[f] for f in order],'feature':order}).to_csv(OUT/'correlation_plot_order.csv',index=False)
def label(s):return s.replace('_score','').replace('_coding','').replace('dbNSFP_','')
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(11.5,10))
fig.subplots_adjust(left=.29,bottom=.29,right=.91,top=.94)
im=ax.imshow(corr.loc[order,order],vmin=-1,vmax=1,cmap='RdBu_r',interpolation='nearest')
ax.set_xticks(range(41));ax.set_xticklabels([label(f) for f in order],rotation=90,fontsize=10)
ax.set_yticks(range(41));ax.set_yticklabels([mapping[f]+'  '+label(f) for f in order],fontsize=10)
ax.tick_params(length=0,pad=3)
ends=np.cumsum(groups.n_features.values)
for x in ends[:-1]:ax.axhline(x-.5,color='white',lw=1);ax.axvline(x-.5,color='white',lw=1)
cax=fig.add_axes([.935,.36,.018,.48]);cb=fig.colorbar(im,cax=cax);cb.set_label('Spearman correlation',fontsize=10)
ax.set_title('41 shared features ordered by the manuscript correlation groups',fontsize=12,pad=12)
fig.savefig(FIG/'supp_correlation_41.pdf',bbox_inches='tight');fig.savefig(FIG/'supp_correlation_41.png',dpi=300,bbox_inches='tight');fig.savefig(FIG/'supp_correlation_41_preview.png',dpi=120,bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(18,7.4));fig.subplots_adjust(left=.15,right=.98,bottom=.13,top=.88,wspace=1.0)
ranks=[];sources=[]
for j,(name,stem,unit) in enumerate([('VARITY','varity','log-odds'),('MutScore','mutscore','probability'),('MVP','mvp','probability')]):
 ss=[];xx=[]
 for k in range(5):
  path=ROOT/'shap'/('%s_fold%d_shap_values.csv'%(stem,k));sh=pd.read_csv(path).set_index('row_id')
  ids=folds.loc[folds.fold==k,'row_id'].values
  assert np.array_equal(ids,sh.index.values)
  med=pd.read_csv(ROOT/'preprocess'/('%s_fold%d_median.csv'%(stem,k))).set_index('feature').train_median
  X=df.loc[ids,features].apply(pd.to_numeric,errors='coerce').fillna(med)
  ss.append(sh[features]);xx.append(X)
 S=pd.concat(ss).sort_index();X=pd.concat(xx).sort_index()
 assert np.array_equal(S.index.values,np.arange(519)) and np.array_equal(X.index.values,np.arange(519))
 S.to_csv(OUT/(stem+'_oof_shap_common41.csv'));X.to_csv(OUT/(stem+'_oof_feature_values_common41.csv'))
 means=S.abs().mean();ordered=means.sort_values(ascending=False)
 for rank,f in enumerate(ordered.index,1):ranks.append({'model':name,'rank':rank,'feature':f,'mean_abs_shap':means[f],'displayed':rank<=20,'unit':unit})
 plt.sca(axes[j]);np.random.seed(2025)
 shap.plots.beeswarm(shap.Explanation(values=S.values,data=X.values,feature_names=[label(f) for f in features]),max_display=20,show=False,plot_size=None,color_bar=False,ax=axes[j],group_remaining_features=False,s=10)
 axes[j].set_title('%s  %s-derived\n519 held-out variants'%(chr(65+j),name),fontsize=13,pad=14)
 axes[j].set_xlabel('SHAP value (%s)'%unit,fontsize=11)
 axes[j].tick_params(axis='y',labelsize=10);axes[j].tick_params(axis='x',labelsize=10)
 axes[j].axvline(0,color='#777777',lw=.6,zorder=0)
fig.suptitle('Five-fold held-out SHAP distributions: top 20 of 41 common features',fontsize=15,y=.985)
cax=fig.add_axes([.40,.045,.23,.012]);cb=fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap=shap.plots.colors.red_blue),cax=cax,orientation='horizontal');cb.set_ticks([0,1]);cb.set_ticklabels(['Low feature value','High feature value']);cb.ax.tick_params(labelsize=10,length=0)
fig.text(.72,.04,'Horizontal scales are model-specific',ha='center',fontsize=10)
fig.savefig(FIG/'supp_oof_shap_beeswarm.pdf');fig.savefig(FIG/'supp_oof_shap_beeswarm.png',dpi=300);fig.savefig(FIG/'supp_oof_shap_beeswarm_preview.png',dpi=110);plt.close(fig)
pd.DataFrame(ranks).to_csv(OUT/'beeswarm_feature_ranking.csv',index=False)
print('Supplementary figures saved to', FIG)
