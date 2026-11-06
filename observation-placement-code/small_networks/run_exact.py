"""Exact SI snapshot inference; no product-marginal approximation.
Run: OPENBLAS_NUM_THREADS=1 python run_exact.py
"""
from pathlib import Path
import json,time,itertools
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.special import xlogy
ROOT=Path(__file__).resolve().parent
(ROOT/'raw').mkdir(exist_ok=True)
TIMES=[0,.05,.1,.2,.4,.8,1.6,3.2,6.4]

def generator(g):
 n=len(g); states=np.arange(1<<n); rows=[];cols=[];data=[]
 for x in states:
  rate=0
  for v in range(n):
   if not (x>>v)&1:
    r=sum((x>>u)&1 for u in g[v])
    if r:
     rows.append(x|1<<v);cols.append(x);data.append(r);rate+=r
  rows.append(x);cols.append(x);data.append(-rate)
 return csr_matrix((data,(rows,cols)),shape=(1<<n,1<<n),dtype=float)

def project(p,sensors):
 n=p.shape[0]; x=np.arange(p.shape[1]);codes=np.zeros(len(x),int)
 for k,v in enumerate(sensors):codes|=((x>>v)&1)<<k
 return np.stack([np.bincount(codes,weights=row,minlength=1<<len(sensors)) for row in p])

def mi(p):
 n=len(p); py=p.mean(axis=0)
 return float((xlogy(p,p)-xlogy(p,py)).sum()/n)

def metrics(p):
 n=len(p); info=mi(p)
 pairs=[np.minimum(p[i],p[j]).sum() for i in range(n) for j in range(i)]
 return {'top1':p.max(axis=0).sum()/n,'top3':np.sort(p,axis=0)[-3:].sum()/n,
 'mi_nats':info,'entropy_nats':np.log(n)-info,'mean_pair_overlap':np.mean(pairs)}

def mismatched(p,q):
 # Uniform random tie-breaking, allowing true probabilities to differ within ties.
 maxima=q.max(axis=0); ties=np.isclose(q,maxima[None,:],rtol=1e-10,atol=1e-14)
 acc=(p*ties/ties.sum(axis=0)).sum()/len(p)
 return float(acc)

def graph(kind,n,seed):
 rng=np.random.default_rng(seed+10000)
 for _ in range(10000):
  k=int(rng.integers(2**31))
  if kind=='ER':g=nx.gnp_random_graph(n,4/(n-1),seed=k)
  elif kind=='BA':g=nx.barabasi_albert_graph(n,2,seed=k)
  else:g=nx.watts_strogatz_graph(n,4,.15,seed=k)
  if nx.is_connected(g):return g
 raise RuntimeError('connected graph generation failed')

def run():
 rows=[];gr=[];checks=[];start=time.time()
 for n in [8,10]:
  for kind in ['ER','BA','WS']:
   for seed in range(8):
    name=f'{kind}_n{n}_seed{seed}';g=graph(kind,n,seed);q=generator(g)
    initial=np.zeros((1<<n,n));initial[1<<np.arange(n),np.arange(n)]=1
    ts=sorted(set(TIMES+[t*f for t in TIMES for f in [.5,2]]))
    pp={}
    for t in ts:
     p=expm_multiply(q*t,initial,traceA=q.diagonal().sum()*t).T if t else initial.T.copy()
     assert p.min()>-1e-12 and np.max(abs(p.sum(axis=1)-1))<1e-10
     pp[t]=np.maximum(p,0)
    np.savez_compressed(ROOT/'raw'/f'{name}.npz',times=ts,probabilities=np.stack([pp[t] for t in ts]),edges=np.array(g.edges()))
    gr.append(dict(network=kind,n=n,seed=seed,edges=g.number_of_edges(),mean_degree=2*g.number_of_edges()/n,clustering=nx.average_clustering(g),path_length=nx.average_shortest_path_length(g)))
    rng=np.random.default_rng(20000+seed+n*100)
    randomorders=[list(rng.permutation(n)) for _ in range(12)]
    degreeorder=sorted(g,key=lambda v:(-g.degree[v],v))
    full_infos=[]
    for t in TIMES:
     p=pp[t]; cache={}
     def get(s):
      key=tuple(sorted(s))
      if key not in cache:cache[key]=project(p,key)
      return cache[key]
     greedy=[]
     for _ in range(n):
      v=max((v for v in range(n) if v not in greedy),key=lambda v:mi(get(greedy+[v])))
      greedy.append(v)
     methods=[('degree',0,degreeorder),('greedy_MI',0,greedy)]+[('random',r,o) for r,o in enumerate(randomorders)]
     for method,r,order in methods:
      for m in range(n+1):
       s=order[:m];obs=get(s);met=metrics(obs)
       row=dict(network=kind,n=n,seed=seed,time=t,method=method,placement=r,m=m,sensors=' '.join(map(str,s)),**met)
       for factor in [.5,2.]:
        assumed=project(pp[t*factor],sorted(s));row['mismatch_top1_'+str(factor)]=mismatched(obs,assumed)
       rows.append(row)
     full_infos.append(mi(p))
    checks.append(dict(graph=name,max_increase_full_mi=float(np.max(np.diff(full_infos)))))
    print(name,'done',round(time.time()-start,1),'s',flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'metrics.csv',index=False)
 pd.DataFrame(gr).to_csv(ROOT/'graphs.csv',index=False)
 pd.DataFrame(checks).to_csv(ROOT/'checks.csv',index=False)
 json.dump(dict(times=TIMES,sizes=[8,10],seeds=list(range(8)),beta=1,random_orders_per_graph=12,model='Continuous-time SI; per-SI-edge exponential hazard beta; known time, single uniform source; exact labelled joint snapshot',graph_generation='Connected ER p=4/(n-1); BA m=2; WS k=4,p=.15. Mean degrees not matched; topology comparisons are descriptive.',runtime_seconds=time.time()-start),open(ROOT/'config.json','w'),indent=2)
 print('DONE',time.time()-start,flush=True)
if __name__=='__main__':run()
