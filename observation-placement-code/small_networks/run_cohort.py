"""Matched-edge independent graph cohort, exact SI channels and exhaustive masks.

The graph distributions, times, budgets, and tie/filter rules are fixed in this
script. The pilot is a timing check on the first three streams per family;
it is not a prospective holdout and does not determine model parameters.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(STUDY/'revision/partial_observation/dependencies'))
import numpy as np
import pandas as pd
import networkx as nx
import scipy
from scipy.sparse.linalg import expm_multiply

HELPER=STUDY/'revision/focused_revision/experiments/run_geometry.py'
spec=importlib.util.spec_from_file_location('focused_helpers',HELPER)
engine=importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
old=engine.old
N=10
EDGE_COUNT=16
TIMES=[.05,.2,.8]
FAMILIES=['ER_Gnm_connected','BA_m2','WS_k4_p015_delete4_connected']
TOL=1e-11
ENTROPY=2026102808161603


def draw_graph(family, sequence):
    rng=np.random.default_rng(sequence)
    for attempt in range(10000):
        nx_seed=int(rng.integers(0,2**63-1))
        deleted=[]
        if family==FAMILIES[0]:
            graph=nx.gnm_random_graph(N,EDGE_COUNT,seed=nx_seed)
        elif family==FAMILIES[1]:
            graph=nx.barabasi_albert_graph(N,2,seed=nx_seed)
        else:
            graph=nx.watts_strogatz_graph(N,4,.15,seed=nx_seed)
            edge_list=sorted(tuple(sorted(edge)) for edge in graph.edges())
            remove=rng.choice(len(edge_list),4,replace=False)
            deleted=[edge_list[int(j)] for j in remove]
            graph.remove_edges_from(deleted)
        if nx.is_connected(graph):
            assert len(graph)==N and graph.number_of_edges()==EDGE_COUNT
            return graph,dict(attempts=attempt+1,nx_seed=nx_seed,removed_edges=deleted,
                              seed_entropy=ENTROPY,spawn_key=list(sequence.spawn_key))
    raise RuntimeError('Connected graph rejection limit exceeded')


def run(count,pilot=False):
    begin=time.perf_counter()
    folder=ROOT/('pilot' if pilot else 'results')
    folder.mkdir(exist_ok=True)
    raw=folder/'raw'
    raw.mkdir(exist_ok=True)
    state=np.arange(1<<N)
    sizes=np.array([int(x).bit_count() for x in state])
    initial=np.zeros((1<<N,N))
    initial[1<<np.arange(N),np.arange(N)]=1
    family_sequences=np.random.SeedSequence(ENTROPY).spawn(len(FAMILIES))
    summaries=[]
    graph_rows=[]
    generation=[]
    checks=[]
    for family,family_sequence in zip(FAMILIES,family_sequences):
        # Always allocate 100 child streams, so pilot and full run have identical
        # first three graphs regardless of requested cohort size.
        sequences=family_sequence.spawn(100)
        for replicate,sequence in enumerate(sequences[:count]):
            graph,meta=draw_graph(family,sequence)
            graph_id=f'{family}_{replicate:03d}'
            edges=np.array(sorted(tuple(sorted(edge)) for edge in graph.edges()),dtype=np.int16)
            generation.append(dict(graph_id=graph_id,family=family,replicate=replicate,
                                   n=N,edge_count=EDGE_COUNT,edges=edges.tolist(),**meta))
            degree=np.array([graph.degree(v) for v in range(N)])
            graph_rows.append(dict(graph_id=graph_id,family=family,replicate=replicate,
                                   n=N,edge_count=len(edges),mean_degree=degree.mean(),
                                   max_degree=degree.max(),degree_sd=degree.std(),
                                   clustering=nx.average_clustering(graph),
                                   mean_distance=nx.average_shortest_path_length(graph)))
            q=engine.generator(N,edges)
            probabilities=[]
            arrays=[]
            for tau in TIMES:
                p=expm_multiply(q*tau,initial,traceA=float(q.diagonal().sum()*tau)).T
                norm_error=float(abs(p.sum(axis=1)-1).max())
                minimum=float(p.min())
                assert norm_error<1e-11 and minimum>=-1e-14
                if minimum<0:
                    p=np.maximum(p,0)
                channels=old.all_channels(p)
                arr=np.array([engine.joint_metrics(channels[int(mask)]) for mask in state])
                projection_error=max(float(abs(channels[k]-old.direct_project(p,k)).max())
                                     for k in [0,1,3,341,1022,1023])
                min_gain=np.full(2,np.inf)
                for v in range(N):
                    sub=state[(state&(1<<v))==0]
                    min_gain=np.minimum(min_gain,(arr[sub|(1<<v)]-arr[sub]).min(axis=0))
                assert projection_error<2e-12 and min_gain.min()>-1e-11
                uniform_error=None
                uniform_tail=None
                if replicate==0:
                    independent,uniform_tail=old.uniformization(N,edges,tau)
                    uniform_error=float(abs(independent-p).max())
                    assert uniform_error<3e-12
                checks.append(dict(graph_id=graph_id,family=family,replicate=replicate,tau=tau,
                                   normalization_error=norm_error,min_probability=minimum,
                                   direct_projection_error=projection_error,
                                   min_MI_coordinate_gain=min_gain[0],min_A_coordinate_gain=min_gain[1],
                                   uniformization_error=uniform_error,uniformization_tail=uniform_tail))
                expected=float((p@sizes).mean())
                for m in range(1,N):
                    eligible=state[sizes==m]
                    mi=arr[eligible,0]
                    accuracy=arr[eligible,1]
                    information_ties=np.flatnonzero(mi>=mi.max()-TOL)
                    information_selected=information_ties[np.argmax(accuracy[information_ties])]
                    selected_mask=int(eligible[information_selected])
                    a_best=float(accuracy.max())
                    a_info=float(accuracy[information_selected])
                    a_random=float(accuracy.mean())
                    regret=max(0.,a_best-a_info)
                    gain=max(0.,a_best-a_random)
                    retention=(a_info-a_random)/gain if gain>TOL else np.nan
                    relative_gain=gain/a_best
                    sensitivity=[]
                    for tolerance in [1e-12,1e-10]:
                        tied=mi>=mi.max()-tolerance
                        sensitivity.append(max(0.,a_best-float(accuracy[tied].max())))
                    summaries.append(dict(graph_id=graph_id,family=family,replicate=replicate,
                        n=N,edges=EDGE_COUNT,tau=tau,m=m,fraction=m/N,
                        rho=expected/N,expected_events=expected-1,
                        saturation_probability=float(p[:,-1].mean()),
                        full_MI=float(arr[-1,0]),full_A=float(arr[-1,1]),
                        information_optimum=float(mi.max()),A_opt_A=a_best,MI_opt_A=a_info,
                        random_A=a_random,delta_A=regret,relative_regret=regret/a_best,
                        design_gain=gain,relative_gain=relative_gain,gain_retention=retention,
                        retention_numerically_defined=bool(gain>TOL),
                        retention_primary_eligible=bool(relative_gain>=.05),
                        MI_ties=len(information_ties),MI_opt_mask=selected_mask,
                        A_ties=int((accuracy>=a_best-TOL).sum()),
                        optima_overlap=bool((accuracy[information_ties]>=a_best-TOL).any()),
                        delta_A_tol1e12=sensitivity[0],delta_A_tol1e10=sensitivity[1],
                        budget_gap=max(0.,float(arr[-1,1])-a_best)))
                probabilities.append(p)
                arrays.append(arr)
            np.savez_compressed(raw/f'{graph_id}.npz',edges=edges,times=np.array(TIMES),
                                probabilities=np.array(probabilities),mask_metrics=np.array(arrays))
            if (replicate+1)%10==0 or pilot:
                print(f'{family} {replicate+1}/{count}, elapsed {time.perf_counter()-begin:.1f}s',flush=True)
        # Checkpoint completed families without overwriting any older experiment.
        pd.DataFrame(summaries).to_csv(folder/'conditions.csv',index=False)
        pd.DataFrame(graph_rows).to_csv(folder/'graphs.csv',index=False)
        pd.DataFrame(checks).to_csv(folder/'checks.csv',index=False)
        (folder/'graph_manifest.json').write_text(json.dumps(generation,indent=2)+'\n')
    config=dict(graph_count=len(graph_rows),graphs_per_family=count,pilot=pilot,
        n=N,edge_count=EDGE_COUNT,mean_degree=2*EDGE_COUNT/N,times=TIMES,
        budgets=list(range(1,N)),families=FAMILIES,graph_seed_entropy=ENTROPY,
        graph_streams='SeedSequence master -> independent family child -> independent replicate child; RNG continues within each rejection sampler',
        graph_distributions={FAMILIES[0]:'Uniform G(10,16), conditioned on connectivity',
            FAMILIES[1]:'NetworkX Barabasi-Albert with attachment m=2 and default initial graph',
            FAMILIES[2]:'WS n=10 k=4 p=.15; remove 4 uniformly chosen edges, then condition resulting graph on connectivity; redraw whole proposal on rejection'},
        no_outbreak_monte_carlo=True,source_prior='uniform on all 10 vertices',
        raw_channels='Exact CTMC matrix exponential; all 1024 masks including endpoints retained in raw arrays',
        condition_count=len(summaries),evaluated_proper_mask_conditions=len(graph_rows)*len(TIMES)*(2**N-2),
        tie_tolerance=TOL,regret_rule='best actual accuracy among all MI tolerance ties',
        primary_retention_filter='(A_opt-A_random)/A_opt >= .05',
        gain_zero_tolerance=TOL,stages='Fixed tau grid with exact expected infected fraction reported; not equal-rho sampling',
        pilot_scope='timing check only, first three predetermined streams; no held-out or preregistered claim',
        reused_helpers={str(HELPER.relative_to(STUDY)):hashlib.sha256(HELPER.read_bytes()).hexdigest(),
            str(Path(old.__file__).relative_to(STUDY)):hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest()},
        runtime_seconds=time.perf_counter()-begin,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,networkx=nx.__version__,pandas=pd.__version__))
    (folder/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps(config,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--count',type=int,default=100)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    if not 1<=args.count<=100:
        parser.error('count must be between 1 and 100')
    run(args.count,args.pilot)
