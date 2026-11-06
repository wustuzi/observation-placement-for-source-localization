"""Local deterministic cycle SI channels; independent Poisson-front projection.

The two fronts advance at unit rate until they meet. For a source u, vertex v
is infected iff (v-u) mod N <= R or (u-v) mod N <= L, where L,R are independent
Poisson(tau) variables. This remains valid after meeting: the union is then V.
Partitioning each Poisson variable at the observation distances gives the
whole channel without state-space enumeration, trajectory sampling or tails.
"""
from pathlib import Path
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'mplcache'))
import sys, json, time, hashlib, math
import numpy as np
import scipy
from scipy.stats import poisson
from scipy.special import xlogy
from scipy.optimize import brentq
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

SIZES=[100,200,500,1000,2000]
TIMES=[.1,.5,1.,2.,4.,8.,12.,20.]

def front_channel(n, observations, tau):
    """Return P(report | source), rows ordered by all N source vertices."""
    obs=np.array(observations,dtype=int)
    assert len(set(obs))==len(obs) and np.all((obs>=0)&(obs<n))
    m=len(obs); u=np.arange(n)[:,None]
    clockwise=(obs[None,:]-u)%n
    counterclockwise=(u-obs[None,:])%n
    bits=1<<np.arange(m)
    def partition(dist):
        thresholds=np.sort(dist,axis=1)
        lo=np.concatenate((np.zeros((n,1)),thresholds),axis=1)
        hi=np.concatenate((thresholds,np.full((n,1),np.inf)),axis=1)
        mass=poisson.sf(lo-1,tau)-poisson.sf(hi-1,tau)
        code=((dist[:,None,:]<=lo[:,:,None])*bits).sum(axis=2)
        return mass,code
    pl,cl=partition(counterclockwise);pr,cr=partition(clockwise)
    p=np.zeros((n,1<<m))
    rows=np.arange(n)
    for j in range(m+1):
        for k in range(m+1):
            np.add.at(p,(rows,cl[:,j]|cr[:,k]),pl[:,j]*pr[:,k])
    assert p.min()>=0 and np.max(abs(p.sum(axis=1)-1))<2e-14
    return p

def metrics(p):
    n=len(p)
    # Rescale each report column before computing KL: averaging a subnormal
    # column directly can underflow to zero and create a spurious infinity.
    scale=p.max(axis=0);active=scale>0
    q=p[:,active]/scale[active]
    mi=float(((xlogy(q,q)-xlogy(q,q.mean(axis=0))).sum(axis=0)*scale[active]).sum()/n)
    accuracy=float(p.max(axis=0).sum()/n)
    assert np.isfinite(mi) and np.isfinite(accuracy)
    return mi,accuracy

def expected_infected(n,tau):
    mu=2*tau
    return float(1+mu*poisson.cdf(n-3,mu)+(n-1)*poisson.sf(n-2,mu))

def masks(n):
    return {'cluster2':[0,1],'spread2':[0,n//2],
            'cluster3':[0,1,2],'spread3':[0,n//3,(2*n)//3]}

def full_generator(n):
    states=np.arange(1<<n,dtype=np.int64);rr=[];cc=[];vv=[]
    escape=np.zeros(len(states))
    for v in range(n):
        rate=(((states>>((v-1)%n))&1)+((states>>((v+1)%n))&1))*(1-((states>>v)&1))
        good=rate>0
        rr.extend((states[good]|(1<<v)).tolist());cc.extend(states[good].tolist());vv.extend(rate[good].tolist())
        escape+=rate
    rr.extend(states.tolist());cc.extend(states.tolist());vv.extend((-escape).tolist())
    return coo_matrix((vv,(rr,cc)),shape=(1<<n,1<<n)).tocsr()

def full_projection(full,obs):
    states=np.arange(full.shape[1]);code=sum(((states>>v)&1)<<j for j,v in enumerate(obs))
    return np.stack([np.bincount(code,weights=row,minlength=1<<len(obs)) for row in full])

def verify():
    records=[]
    for n in [4,6,8,10,12]:
        q=full_generator(n);init=np.zeros((1<<n,n));init[1<<np.arange(n),np.arange(n)]=1
        state_sizes=np.array([i.bit_count() for i in range(1<<n)])
        for tau in [.0001,.1,1.,4.,20.]:
            full=expm_multiply(q*tau,init,traceA=float(q.diagonal().sum()*tau)).T
            full=np.maximum(full,0)
            for name,obs in masks(n).items():
                p=front_channel(n,obs,tau);ref=full_projection(full,obs)
                mi,a=metrics(p);mi0,a0=metrics(ref)
                err=float(np.max(abs(p-ref)))
                mean_err=float(abs((full@state_sizes).mean()-expected_infected(n,tau)))
                assert err<2e-12 and abs(mi-mi0)<2e-12 and abs(a-a0)<2e-12 and mean_err<2e-11
                records.append(dict(n=n,tau=tau,layout=name,max_channel_error=err,
                    MI_error=abs(mi-mi0),A1_error=abs(a-a0),expected_infected_error=mean_err))
    return records

def write_csv(path,rows):
    import csv
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def crossing(n,m=2):
    choices=masks(n)
    def diff(t):
        return metrics(front_channel(n,choices[f'cluster{m}'],t))[0]-metrics(front_channel(n,choices[f'spread{m}'],t))[0]
    # Prespecified positive root search; no assumption about its location/sign.
    grid=np.geomspace(1e-5/n,20.,180)
    roots=[];last_t=grid[0];last_f=diff(last_t)
    for t in grid[1:]:
        f=diff(t)
        if last_f>0 and f<0:
            root=brentq(diff,last_t,t,xtol=1e-13,rtol=1e-12)
            assert abs(diff(root))<1e-11 and diff(root*.9)>0 and diff(root*1.1)<0
            roots.append(dict(n=n,m=m,tau_c=root,N_tau_c=n*root,
                delta_MI=diff(root),delta_MI_below=diff(root*.9),delta_MI_above=diff(root*1.1),
                bracket_low=last_t,bracket_high=t))
        last_t,last_f=t,f
    return roots

def main():
    start=time.perf_counter();(ROOT/'channels').mkdir(exist_ok=True)
    validation=verify();write_csv(ROOT/'full_state_crosscheck.csv',validation)
    print('full-state validation passed',len(validation),flush=True)
    rows=[];contrasts=[]
    for n in SIZES:
        for tau in TIMES:
            values={}
            for name,obs in masks(n).items():
                p=front_channel(n,obs,tau);mi,a=metrics(p);values[name]=(mi,a)
                mean=expected_infected(n,tau)
                np.savez_compressed(ROOT/'channels'/f'N{n}_{name}_tau{tau:g}.npz',channel=p,observations=obs,n=n,tau=tau)
                m=len(obs);internal=sum((v+1)%n in obs for v in obs)
                rows.append(dict(n=n,tau=tau,m=m,layout=name,observations=';'.join(map(str,obs)),
                    internal_edges=internal,boundary_edges=2*m-2*internal,
                    pair_distance=min((obs[1]-obs[0])%n,(obs[0]-obs[1])%n) if m==2 else '',
                    mean_degree=2,MI=mi,A1=a,N_A1=n*a,mean_infected=mean,infected_fraction=mean/n,
                    normalization_error=float(abs(p.sum(1)-1).max())))
            for m in [2,3]:
                ic,ac=values[f'cluster{m}'];is_,as_=values[f'spread{m}']
                contrasts.append(dict(n=n,tau=tau,m=m,delta_MI_cluster_minus_spread=ic-is_,
                    delta_A1_spread_minus_cluster=as_-ac,N_delta_MI=n*(ic-is_),N_delta_A1=n*(as_-ac)))
        print('fixed-time results complete N=',n,flush=True)
    write_csv(ROOT/'results.csv',rows);write_csv(ROOT/'contrasts.csv',contrasts)
    roots=[]
    for n in [10,20,50]+SIZES:
        for m in [2,3]: roots.extend(crossing(n,m))
        print('crossing search complete N=',n,flush=True)
    write_csv(ROOT/'crossovers.csv',roots)
    # A common scaled-time grid to visualize the finite-size crossover.
    scaling=[]
    for n in SIZES:
        for x in np.geomspace(.01,10,100):
            tau=x/n
            for m in [2,3]:
                obs=masks(n)
                ic,ac=metrics(front_channel(n,obs[f'cluster{m}'],tau))
                is_,as_=metrics(front_channel(n,obs[f'spread{m}'],tau))
                scaling.append(dict(n=n,m=m,N_tau=x,tau=tau,
                    N2_delta_MI=n*n*(ic-is_),N2_delta_A1=n*n*(as_-ac)))
    write_csv(ROOT/'scaled_time.csv',scaling)
    summary={'status':'passed','method':'deterministic two-Poisson-front threshold partition',
        'source_prior':'uniform over all N vertices','tail_truncation':'none; exact Poisson CDF/SF partition',
        'floating_point_arithmetic':'IEEE float64, scipy Poisson survival functions',
        'network_sizes':SIZES,'fixed_absolute_times':TIMES,'layouts':list(masks(100)),
        'results_rows':len(rows),'contrast_rows':len(contrasts),'full_state_comparisons':len(validation),
        'max_channel_error':max(r['max_channel_error'] for r in validation),
        'max_MI_error':max(r['MI_error'] for r in validation),'max_A1_error':max(r['A1_error'] for r in validation),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'runtime_seconds':time.perf_counter()-start,'random_seed':None,
        'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'local_execution':True,'cloud_results_used':False}
    (ROOT/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
