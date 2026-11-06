"""Cross-check independently generated FPP samples against the full CTMC."""
import json,subprocess,time
from datetime import datetime,timezone
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from run import ROOT,TIMES,nx,write_input,read_binary,metrics,accuracy_bounds

def exact(g):
    n=len(g);rows=[];cols=[];values=[]
    for x in range(1<<n):
        escape=0
        for v in g:
            if x>>v&1:continue
            rate=sum(x>>u&1 for u in g[v])
            if rate:rows.append(x|(1<<v));cols.append(x);values.append(rate);escape+=rate
        rows.append(x);cols.append(x);values.append(-escape)
    q=coo_matrix((values,(rows,cols)),shape=(1<<n,1<<n)).tocsr()
    init=np.zeros((1<<n,n));init[1<<np.arange(n),np.arange(n)]=1
    return np.array([expm_multiply(q*t,init,traceA=q.diagonal().sum()*t).T for t in TIMES])

def main():
    started=datetime.now(timezone.utc).isoformat();start=time.perf_counter()
    (ROOT/'validation').mkdir(exist_ok=True)
    results=[]
    graphs=[('cycle8',nx.cycle_graph(8)),('path8',nx.path_graph(8)),('BA8',nx.barabasi_albert_graph(8,2,seed=910321))]
    for name,g in graphs:
        pairs=[(0,1),(0,4),(1,5),(2,7)]
        c=dict(graph_id='validation_'+name,layouts=[dict(u=u,v=v) for u,v in pairs])
        inp=write_input(g,c);out=ROOT/'validation'/f'{name}.bin';R=4096;B=16
        subprocess.run([str(ROOT/'fpp_mc'),str(inp),str(out),str(B),str(R),'901239'],stderr=subprocess.DEVNULL,check=True)
        counts,sums,ss,R=read_binary(out);empirical=counts.sum(axis=0)/(B*R);full=exact(g)
        projected=[]
        for u,v in pairs:
            states=np.arange(1<<len(g));mapping=((states>>u)&1)+2*((states>>v)&1)
            projected.append(np.stack([full[:,:,mapping==y].sum(axis=-1) for y in range(4)],axis=-1))
        truth=np.array(projected);err=float(abs(empirical-truth).max())
        # Union bound across graph-specific channel cells; dependence between
        # cells is allowed, independence is required only across edge samples.
        bound=float(np.sqrt(np.log(2*truth.size/.001)/(2*B*R)))
        mi,a=metrics(counts.sum(axis=0));mit,at=metrics(truth)
        al,au=accuracy_bounds(counts.sum(axis=0),B*R)
        infection_truth=(full*np.array([x.bit_count() for x in range(1<<len(g))])).sum(axis=-1).mean(axis=-1)
        r=dict(graph=name,n=len(g),samples=B*R,max_channel_abs_error=err,
               simultaneous_hoeffding_bound_99_9pct=bound,passed=err<bound,
               max_MI_abs_error=float(abs(mi-mit).max()),max_A1_abs_error=float(abs(a-at).max()),
               max_mean_infected_abs_error=float(abs(sums.sum(axis=0)/(B*R)-infection_truth).max()),
               row_sum_error=float(abs(empirical.sum(axis=-1)-1).max()))
        r['all_exact_A1_inside_CP_bounds']=bool(np.all((al<=at)&(at<=au)))
        results.append(r)
    assert all(r['passed'] and r['row_sum_error']==0 for r in results)
    (ROOT/'validation'/'validation.json').write_text(json.dumps(dict(status='passed',checks=results,method='independent full-state CTMC vs exponential-edge FPP',seed=901239,started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),runtime_seconds=time.perf_counter()-start),indent=2)+'\n')
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
