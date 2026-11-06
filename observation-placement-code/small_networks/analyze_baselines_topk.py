"""Evaluate implementable designs and top-3 decisions on the same exact snapshots."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
import pandas as pd
import networkx as nx
OUT=Path(__file__).resolve().parent; STUDY=OUT.parents[2]
OLD=STUDY/'revision/partial_observation/experiments';TOL=1e-11
subsets=pd.read_csv(OLD/'all_subsets.csv.gz')
strategies=pd.read_csv(OLD/'strategies.csv.gz')
stage=pd.read_csv(STUDY/'revision/focused_revision/experiments/comparisons.csv')
KEY=['cohort','network','n','seed','time','m']
stage=stage[KEY+['infected_fraction','mean_infected']]
rows=[];krows=[];validate=[]
graphfiles={}
for cohort,folder in [('discovery',STUDY/'exact/raw'),('validation',STUDY/'revision/design/additional_raw')]:
 for p in folder.glob('*.npz'):
  net=p.stem.split('_')[0];seed=int(p.stem.split('seed')[1]);z=np.load(p);n=z['probabilities'].shape[1]
  if n not in [8,10]:continue
  g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from(z['edges'].tolist());dist=np.array(nx.floyd_warshall_numpy(g),int)
  radius={mask:int(dist[:,[v for v in range(n) if mask>>v&1]].min(1).max()) for mask in range(1,1<<n)}
  graphfiles[(cohort,net,n,seed)]=radius
for key,df in subsets.groupby(KEY,sort=True):
 cohort,net,n,seed,time,m=key
 if not 0<m<n:continue
 meta=dict(zip(KEY,key));df=df.sort_values('mask');a=df.top1.to_numpy();a3=df.top3.to_numpy();mi=df.mi.to_numpy();mask=df['mask'].to_numpy()
 ao=float(a.max());ar=float(a.mean());a3o=float(a3.max());a3r=float(a3.mean())
 mi_ties=mi>=mi.max()-TOL
 # Retain the existing conservative exact-hit convention for the chosen MI mask.
 candidates=np.flatnonzero(mi_ties);chosen=candidates[np.argmax(a[candidates])]
 krows.append(dict(**meta,MI_top3=float(a3[chosen]),MI_best_top3_tie=float(a3[mi_ties].max()),top3_optimum=a3o,top3_random=a3r,
     top3_regret=a3o-float(a3[chosen]),top3_relative_regret=(a3o-float(a3[chosen]))/a3o,
     top3_unavoidable_regret=a3o-float(a3[mi_ties].max()),top3_headroom=a3o-a3r,
     top3_gain_retention=(float(a3[chosen])-a3r)/(a3o-a3r) if a3o-a3r>1e-10 else np.nan))
 radii=graphfiles[(cohort,net,n,seed)];rs=np.array([radii[int(x)] for x in mask]);kmask=int(mask[rs==rs.min()].min());idx=int(np.flatnonzero(mask==kmask)[0])
 rows.append(dict(**meta,strategy='Exact k-center',mask=kmask,top1=float(a[idx]),top3=float(a3[idx]),radius=int(rs.min()),optimum=ao,random=ar))
for row in strategies.itertuples(index=False):
 if not 0<row.m<row.n or row.strategy not in ['MI optimum','A1 optimum','Uniform random','Output entropy','Greedy MI','Degree','Coverage']:continue
 rows.append({**{k:getattr(row,k) for k in KEY},'strategy':row.strategy,'mask':row.mask,'top1':row.top1,'top3':row.top3})
df=pd.DataFrame(rows)
opts=df[df.strategy=='A1 optimum'][KEY+['top1']].rename(columns={'top1':'accuracy_optimum'})
rnd=df[df.strategy=='Uniform random'][KEY+['top1']].rename(columns={'top1':'random_accuracy'})
df=df.merge(opts,on=KEY,validate='many_to_one').merge(rnd,on=KEY,validate='many_to_one').merge(stage,on=KEY,validate='many_to_one')
df['relative_regret']=(df.accuracy_optimum-df.top1)/df.accuracy_optimum
df['absolute_regret']=df.accuracy_optimum-df.top1
df['gain_headroom']=df.accuracy_optimum-df.random_accuracy
df['gain_retention']=np.where(df.gain_headroom>1e-10,(df.top1-df.random_accuracy)/df.gain_headroom,np.nan)
df['rho_band']=pd.cut(df.infected_fraction,[0,.25,.5,.75,1.000001],labels=['up to 0.25','0.25-0.50','0.50-0.75','0.75-1.00'])
assert df.relative_regret.min()>-2e-10
df.to_csv(OUT/'baseline_conditions.csv.gz',index=False)
summary=df.groupby(['rho_band','strategy'],observed=True).agg(conditions=('relative_regret','size'),mean_relative_regret=('relative_regret','mean'),median_relative_regret=('relative_regret','median'),worst_relative_regret=('relative_regret','max'),mean_accuracy=('top1','mean')).reset_index()
summary.to_csv(OUT/'baseline_by_stage.csv',index=False)
top=pd.DataFrame(krows).merge(stage,on=KEY,validate='one_to_one');top['rho_band']=pd.cut(top.infected_fraction,[0,.25,.5,.75,1.000001],labels=['up to 0.25','0.25-0.50','0.50-0.75','0.75-1.00'])
top.to_csv(OUT/'top3_conditions.csv',index=False)
ts=top.groupby('rho_band',observed=True).agg(conditions=('top3_regret','size'),mean_relative_regret=('top3_relative_regret','mean'),worst_relative_regret=('top3_relative_regret','max'),nonzero_unavoidable_regret=('top3_unavoidable_regret',lambda x:int((x>1e-10).sum()))).reset_index();ts.to_csv(OUT/'top3_by_stage.csv',index=False)
largest=top.loc[top.top3_relative_regret.idxmax()].to_dict()
report={'baseline_rows':len(df),'top3_conditions':len(top),'top3_largest_relative_regret':largest,'strategy_names':sorted(df.strategy.unique()),'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [OLD/'all_subsets.csv.gz',OLD/'strategies.csv.gz']},'conventions':'Five fixed ages, all proper budgets, descriptive condition summaries; no independent-observation standard errors. k-center ties use the smallest integer mask; no accuracy-based tie selection.'}
(OUT/'baseline_top3_summary.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps(report,indent=2,default=str))
