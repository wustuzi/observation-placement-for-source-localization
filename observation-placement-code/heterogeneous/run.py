"""Reproducible local SI channels; no external simulation outputs are imported."""
import argparse,csv,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.special import betaincinv
ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parents[2]
sys.path.insert(0,str(STUDY/'revision/partial_observation/dependencies'))
import networkx as nx
TIMES=np.array([.1,.5,1.,2.,4.])
MASTER=202610301317

def graph_config(n,family,rep):
    seed=int(np.random.SeedSequence([MASTER,n,{'ER':0,'BA':1,'WS':2}[family],rep]).generate_state(1)[0])
    rng=np.random.default_rng(seed)
    if family=='ER':g=nx.fast_gnp_random_graph(n,4/(n-1),seed=seed)
    elif family=='BA':g=nx.barabasi_albert_graph(n,2,seed=seed)
    else:g=nx.watts_strogatz_graph(n,4,.1,seed=seed)
    original_nodes=sorted(max(nx.connected_components(g),key=len))
    g=nx.convert_node_labels_to_integers(g.subgraph(original_nodes).copy(),ordering='sorted')
    edges=sorted(tuple(sorted(e)) for e in g.edges());actual=len(g)
    cluster=edges[int(rng.integers(len(edges)))]
    # Two-sweep pseudodiameter, independent of simulated outcomes.
    v=int(rng.integers(actual));best=None
    for _ in range(4):
        ds=nx.single_source_shortest_path_length(g,v);d=max(ds.values());u=min(x for x in ds if ds[x]==d)
        candidate=(d,tuple(sorted((v,u))))
        if best is None or candidate[0]>best[0]:best=candidate
        v=u
    pairs=[('cluster',cluster),('spread',best[1])]
    for j in range(4):pairs.append((f'random_{j}',tuple(sorted(rng.choice(actual,2,replace=False).tolist()))))
    layouts=[]
    for label,(u,v) in pairs:
        internal=int(g.has_edge(u,v));boundary=g.degree[u]+g.degree[v]-2*internal
        outside=[x for x in g if x not in (u,v)]
        amin=min(int(g.has_edge(x,u))+int(g.has_edge(x,v)) for x in outside)
        layouts.append(dict(layout=label,u=u,v=v,e_M=internal,c_M=boundary,a_M=amin,
                            distance=nx.shortest_path_length(g,u,v),degree_u=g.degree[u],degree_v=g.degree[v]))
    return g,dict(graph_id=f'{family}_{n}_{rep}',family=family,requested_n=n,n=actual,replicate=rep,
                  edges=len(edges),mean_degree=2*len(edges)/actual,graph_seed=seed,
                  mc_seed=seed+124728914,original_nodes=original_nodes,layouts=layouts)

def write_input(g,c):
    p=ROOT/'graphs'/f"{c['graph_id']}.txt";p.parent.mkdir(exist_ok=True)
    lines=[f"{len(g)} {g.number_of_edges()} {len(c['layouts'])}"]
    lines.extend(f'{u} {v}' for u,v in sorted(g.edges()))
    lines.extend(f"{x['u']} {x['v']}" for x in c['layouts'])
    p.write_text('\n'.join(lines)+'\n');c['graph_input_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def read_binary(p):
    with p.open('rb') as f:
        n,k,nt,B,R,obs=np.fromfile(f,dtype=np.int32,count=6)
        counts=[];sums=[];sumsq=[]
        for _ in range(B):
            counts.append(np.fromfile(f,dtype=np.uint32,count=k*nt*n*4).reshape(k,nt,n,4))
            sums.append(np.fromfile(f,dtype=np.float64,count=nt));sumsq.append(np.fromfile(f,dtype=np.float64,count=nt))
        assert f.read()==b''
    return np.array(counts),np.array(sums),np.array(sumsq),int(R)

def metrics(counts):
    p=counts/counts.sum(axis=-1,keepdims=True);q=p.mean(axis=-2,keepdims=True)
    with np.errstate(divide='ignore',invalid='ignore'):
        mi=np.where(p>0,p*np.log(np.divide(p,q,out=np.zeros_like(p),where=q>0)),0).sum(axis=(-2,-1))/p.shape[-2]
    a=p.max(axis=-2).sum(axis=-1)/p.shape[-2]
    return mi,a

def accuracy_bounds(total,samples,alpha=.025):
    """Clopper-Pearson + union bound across 4N channel entries per layout/time.

    Source-conditional Bernoulli counts are independent over edge realizations;
    no independence between sources is required for this simultaneous bound.
    Each design/time has >=97.5% coverage, so a fixed two-design contrast has
    >=95% coverage by a second union bound. Not a global 225-condition band.
    """
    n=total.shape[-2];tail=alpha/(2*4*n)
    lower=np.zeros(total.shape);upper=np.ones(total.shape)
    pos=total>0;nonfull=total<samples
    lower[pos]=betaincinv(total[pos],samples-total[pos]+1,tail)
    upper[nonfull]=betaincinv(total[nonfull]+1,samples-total[nonfull],1-tail)
    return lower.max(axis=-2).sum(axis=-1)/n,upper.max(axis=-2).sum(axis=-1)/n

def analyze(c,p):
    counts,sums,sumsq,R=read_binary(p);B=len(counts);n=c['n'];total=counts.sum(axis=0);samples=B*R
    mi,a=metrics(total);mi_half,a_half=metrics(counts[:B//2].sum(axis=0))
    a_lower,a_upper=accuracy_bounds(total,samples)
    loo=[metrics(total-x) for x in counts];loo_mi=np.array([x[0] for x in loo]);loo_a=np.array([x[1] for x in loo])
    mi_jack=B*mi-(B-1)*loo_mi.mean(axis=0)
    se_mi=np.sqrt((B-1)/B*((loo_mi-loo_mi.mean(axis=0))**2).sum(axis=0))
    se_a=np.sqrt((B-1)/B*((loo_a-loo_a.mean(axis=0))**2).sum(axis=0))
    train=counts[:B//2].sum(axis=0);test=counts[B//2:];rules=train.argmax(axis=-2)
    eval_batches=np.zeros((len(test),len(c['layouts']),len(TIMES)))
    for b,block in enumerate(test):
        for h in range(len(c['layouts'])):
            for t in range(len(TIMES)):
                eval_batches[b,h,t]=sum(block[h,t,rules[h,t,y],y] for y in range(4))/(R*n)
    heldout=eval_batches.mean(axis=0);se_heldout=eval_batches.std(axis=0,ddof=1)/np.sqrt(len(test))
    infection=sums.sum(axis=0)/samples
    inf_se=np.sqrt(np.maximum(0,sumsq.sum(axis=0)/samples-infection**2)/(samples-1))
    rows=[]
    for h,layout in enumerate(c['layouts']):
        for t,tau in enumerate(TIMES):
            row={k:c[k] for k in ['graph_id','family','requested_n','n','replicate','edges','mean_degree','graph_seed','mc_seed']}
            row.update(layout,tau=tau,mc_samples=samples,training_samples=(B//2)*R,test_samples=len(test)*R,
                       information_plugin=mi[h,t],information_jackknife=mi_jack[h,t],information_se=se_mi[h,t],
                       information_half_samples=mi_half[h,t],accuracy_plugin=a[h,t],accuracy_plugin_se=se_a[h,t],
                       accuracy_lower_97_5=a_lower[h,t],accuracy_upper_97_5=a_upper[h,t],
                       accuracy_half_samples=a_half[h,t],accuracy_heldout=heldout[h,t],accuracy_heldout_se=se_heldout[h,t],
                       scaled_accuracy_plugin=n*a[h,t],scaled_accuracy_heldout=n*heldout[h,t],
                       mean_infected=infection[t],mean_infected_se=inf_se[t],infected_fraction=infection[t]/n)
            rows.append(row)
    contrasts=[]
    for t,tau in enumerate(TIMES):
        # Paired coupling retained in all uncertainty calculations.
        dloo=loo_mi[:,0,t]-loo_mi[:,1,t]
        dse=np.sqrt((B-1)/B*((dloo-dloo.mean())**2).sum())
        da_loo=loo_a[:,1,t]-loo_a[:,0,t]
        da_se=np.sqrt((B-1)/B*((da_loo-da_loo.mean())**2).sum())
        held=eval_batches[:,1,t]-eval_batches[:,0,t]
        contrasts.append(dict(graph_id=c['graph_id'],family=c['family'],requested_n=c['requested_n'],n=n,replicate=c['replicate'],tau=tau,
            delta_I_cluster_minus_spread=mi[0,t]-mi[1,t],delta_I_jackknife=mi_jack[0,t]-mi_jack[1,t],delta_I_se=dse,
            delta_A_spread_minus_cluster=a[1,t]-a[0,t],delta_A_plugin_se=da_se,scaled_delta_A=n*(a[1,t]-a[0,t]),
            delta_A_lower_95=a_lower[1,t]-a_upper[0,t],delta_A_upper_95=a_upper[1,t]-a_lower[0,t],
            delta_A_heldout=held.mean(),delta_A_heldout_se=held.std(ddof=1)/np.sqrt(len(held)),
            random_information_mean=mi[2:,t].mean() if len(c['layouts'])>2 else float('nan'),
            random_accuracy_plugin_mean=a[2:,t].mean() if len(c['layouts'])>2 else float('nan'),
            random_accuracy_heldout_mean=heldout[2:,t].mean() if len(c['layouts'])>2 else float('nan'),
            mean_infected=infection[t],infected_fraction=infection[t]/n))
    return rows,contrasts

def write_csv(path,rows):
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sizes',nargs='+',type=int,default=[100,250,500,1000,2000]);ap.add_argument('--replicates',type=int,default=3);ap.add_argument('--batches',type=int,default=16);ap.add_argument('--per-batch',type=int,default=512);ap.add_argument('--families',nargs='+',default=['ER','BA','WS']);a=ap.parse_args()
    ROOT.mkdir(exist_ok=True,parents=True);(ROOT/'raw').mkdir(exist_ok=True);(ROOT/'configs').mkdir(exist_ok=True)
    subprocess.run(['clang++','-O3','-std=c++17',str(ROOT/'fpp_mc.cpp'),'-o',str(ROOT/'fpp_mc')],check=True)
    engine_hash=hashlib.sha256((ROOT/'fpp_mc.cpp').read_bytes()).hexdigest()
    allrows=[];allcontrast=[]
    for n in a.sizes:
        for family in a.families:
            for rep in range(a.replicates):
                g,c=graph_config(n,family,rep);inp=write_input(g,c);out=ROOT/'raw'/f"{c['graph_id']}.bin"
                c.update(batches=a.batches,per_batch=a.per_batch,times=TIMES.tolist(),rate=1,method='exponential-edge first-passage SI, all-source coupled channels',random_baseline_pairs=4)
                c['engine_source_sha256']=engine_hash
                if not out.exists():
                    start=time.perf_counter()
                    with (ROOT/'raw'/f"{c['graph_id']}.log").open('w') as log:
                        subprocess.run([str(ROOT/'fpp_mc'),str(inp),str(out),str(a.batches),str(a.per_batch),str(c['mc_seed'])],stderr=log,check=True)
                    c['runtime_seconds']=time.perf_counter()-start
                else:
                    prev=ROOT/'configs'/f"{c['graph_id']}.json"
                    if not prev.exists():raise RuntimeError(f'Cached counts lack configuration: {out}')
                    saved=json.loads(prev.read_text())
                    for key in ['graph_input_sha256','graph_seed','mc_seed','n','edges','layouts','batches','per_batch','times','rate']:
                        if saved.get(key)!=c[key]:raise RuntimeError(f'Cached counts incompatible with current {key}: {out}')
                    old_engine=saved.get('engine_source_sha256')
                    if old_engine is None:
                        old_engine=json.loads((ROOT/'run_manifest.json').read_text())['code_sha256']['fpp_mc.cpp']
                    if old_engine!=engine_hash:raise RuntimeError(f'Cached simulation engine differs: {out}')
                    if hashlib.sha256(out.read_bytes()).hexdigest()!=saved['raw_sha256']:raise RuntimeError(f'Cached data hash mismatch: {out}')
                    c['runtime_seconds']=saved.get('runtime_seconds')
                    old=read_binary(out);assert len(old[0])==a.batches and old[3]==a.per_batch,'incompatible cached samples'
                c['raw_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
                (ROOT/'configs'/f"{c['graph_id']}.json").write_text(json.dumps(c,indent=2)+'\n')
                rows,contrasts=analyze(c,out);allrows+=rows;allcontrast+=contrasts
                write_csv(ROOT/'conditions.csv',allrows);write_csv(ROOT/'contrasts.csv',allcontrast)
                print(c['graph_id'],c.get('runtime_seconds'),flush=True)
    (ROOT/'run_manifest.json').write_text(json.dumps(dict(master_seed=MASTER,arguments=vars(a),graphs=len(allrows)//30,python=sys.version,numpy=np.__version__,networkx=nx.__version__,code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'run.py',ROOT/'fpp_mc.cpp']}),indent=2)+'\n')
if __name__=='__main__':main()
