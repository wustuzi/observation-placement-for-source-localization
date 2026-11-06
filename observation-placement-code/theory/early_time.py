"""Exact CTMC checks of the early-time graph-cut mechanism for sensor design."""
from run_design import *
from scipy.sparse.linalg import expm_multiply
sys.path.insert(0,str(ROOT.parent.parent/'exact'))
from run_exact import generator,project

def run():
  config=[('cycle10_m2',nx.cycle_graph(10),2),('petersen10_m2',nx.petersen_graph(),2),('cycle12_m3',nx.cycle_graph(12),3),('path8_m3',nx.path_graph(8),3),('star8_m3',nx.star_graph(7),3)]
  tt=[.0001,.0003,.001,.003,.01,.03,.1]
  allrows=[];summary=[]
  for name,g,m in config:
    n=len(g);L=n-m;qq=generator(g);initial=np.zeros((1<<n,n));initial[1<<np.arange(n),np.arange(n)]=1
    dmat=np.array([[nx.shortest_path_length(g,i,j) for j in range(n)] for i in range(n)],float)
    baseI=np.log(n)-(L/n)*np.log(L);baseA=(m+1)/n
    for t in tt:
      p=expm_multiply(qq*t,initial,traceA=qq.diagonal().sum()*t).T;rows=[]
      for sens in itertools.combinations(range(n),m):
        M=set(sens);ein=g.subgraph(M).number_of_edges();cut=sum(1 for u,v in g.edges() if (u in M)!=(v in M));dmin=min(sum(v in M for v in g[u]) for u in g if u not in M)
        obs=project(p,sens);info,acc,acc3,hop=metric(obs,dmat)
        predI=baseI-cut*t*np.log(1/t)/n+t*(cut*(np.log(L)-1)-2*ein*np.log(2))/n
        predA=baseA-(ein+dmin)*t/n
        rows.append(dict(graph=name,n=n,m=m,time=t,mask=sum(1<<v for v in M),induced_edges=ein,cut=cut,min_external_sensor_degree=dmin,mi=info,top1=acc,mi_prediction=predI,accuracy_prediction=predA,mi_remainder=info-predI,accuracy_remainder=acc-predA))
      d=pd.DataFrame(rows);allrows.extend(rows)
      maxi=d.mi.max();maxa=d.top1.max();ii=d[abs(d.mi-maxi)<=1e-12];aa=d[abs(d.top1-maxa)<=1e-12]
      emax=d.induced_edges.max();emin=d.induced_edges.min();regular=len(set(dict(g.degree()).values()))==1
      theorem=regular and m*(g.degree[0]+1)<n and emax>emin
      summary.append(dict(graph=name,n=n,m=m,time=t,regular_theorem_applicable=theorem,induced_edges_max=emax,induced_edges_min=emin,mi_optimal_induced_edges_min=ii.induced_edges.min(),mi_optimal_induced_edges_max=ii.induced_edges.max(),top1_optimal_induced_edges_min=aa.induced_edges.min(),top1_optimal_induced_edges_max=aa.induced_edges.max(),top1_regret=maxa-ii.top1.max(),predicted_regret=t*(emax-emin)/n if theorem else np.nan,max_abs_mi_remainder=d.mi_remainder.abs().max(),max_abs_accuracy_remainder=d.accuracy_remainder.abs().max(),scaled_mi_remainder=d.mi_remainder.abs().max()/(t*t*abs(np.log(t))),scaled_accuracy_remainder=d.accuracy_remainder.abs().max()/(t*t)))
    print(name,'done',flush=True)
  D=pd.DataFrame(allrows);S=pd.DataFrame(summary);D.to_csv(ROOT/'early_time_subsets.csv.gz',index=False,compression='gzip');S.to_csv(ROOT/'early_time_summary.csv',index=False)
  # Conditions tested at sufficiently early times, not a theorem asserting all finite t.
  z=S[S.regular_theorem_applicable&(S.time<=.01)]
  assert (z.mi_optimal_induced_edges_min==z.induced_edges_max).all()
  assert (z.top1_optimal_induced_edges_max==z.induced_edges_min).all()
  assert (z.top1_regret>0).all()
  json.dump(dict(subset_conditions=len(D),graph_time_conditions=len(S),regular_theorem_checked_cases=len(z),all_pass=True,max_scaled_mi_remainder=float(S.scaled_mi_remainder.max()),max_scaled_accuracy_remainder=float(S.scaled_accuracy_remainder.max())),open(ROOT/'early_time_validation.json','w'),indent=2)
if __name__=='__main__':run()
