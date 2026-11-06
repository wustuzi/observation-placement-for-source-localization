"""M2/M3 analysis of existing exact channels. No graph or outbreak randomness."""
from pathlib import Path
import hashlib,importlib.util,json,platform,time
import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import xlogy
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent;STUDY=ROOT.parents[2]
PRIOR=STUDY/'revision/focused_revision/experiments';TOL=1e-11
spec=importlib.util.spec_from_file_location('prior_exact',STUDY/'revision/partial_observation/experiments/run.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
KEY=['cohort','network','n','seed'];CELL=KEY+['time','m']
BUDGET_LABELS=['(0,0.25]','(0.25,0.50]','(0.50,0.75]','(0.75,1)']
EVENT_LABELS=['[0,0.1]','(0.1,0.5]','(0.5,1]','(1,2]','(2,4]','(4,inf)']
RHO_LABELS=['[0,0.20]','(0.20,0.40]','(0.40,0.60]','(0.60,0.80]','(0.80,0.95]','(0.95,1]']

def source_file(cohort,network,n,seed):
    folder=STUDY/('exact/raw' if cohort=='discovery' else 'revision/design/additional_raw')
    return folder/f'{network}_n{n}_seed{seed}.npz'

def full_law(cohort,network,n,seed,t):
    path=source_file(cohort,network,n,seed);z=np.load(path);i=np.flatnonzero(np.isclose(z['times'],t,atol=1e-14,rtol=0))
    if not len(i):
        path=PRIOR/'raw'/f'{cohort}_{network}_n{n}_seed{seed}.npz';z=np.load(path);i=np.flatnonzero(np.isclose(z['times'],t,atol=1e-14,rtol=0))
    assert len(i)==1
    return z['probabilities'][i[0]],z['edges'],path

def summarize(q):
    out=dict(conditions=len(q),graphs=len(q[KEY].drop_duplicates()),seed_blocks=q.seed.nunique(),
       relative_regret_max=q.relative_regret.max(),relative_regret_median=q.relative_regret.median(),absolute_regret_max=q.delta_A.max(),
       expected_events_min=q.expected_events.min(),expected_events_max=q.expected_events.max(),rho_min=q.rho.min(),rho_max=q.rho.max(),
       numerical_gain_undefined=int((~q.retention_numerically_defined).sum()))
    for pct in [1,5,10]:
        a=q[q.relative_gain>=pct/100];r=a.gain_retention
        out.update({f'eligible_{pct}pct':len(a),f'negative_{pct}pct':int((r< -TOL).sum()),f'negative_fraction_{pct}pct':float((r< -TOL).mean()) if len(a) else np.nan,
            f'min_retention_{pct}pct':r.min(),f'median_retention_{pct}pct':r.median(),f'below_half_{pct}pct':int((r<.5-TOL).sum())})
    for label,threshold in [('0p1pp',.001),('1pp',.01)]:
        a=q[q.design_gain>=threshold]
        out.update({f'eligible_{label}':len(a),f'negative_{label}':int((a.gain_retention<-TOL).sum()),f'min_retention_{label}':a.gain_retention.min()})
    return pd.Series(out)

def run():
    start=time.time();c=pd.read_csv(PRIOR/'comparisons.csv');sub=pd.read_csv(PRIOR/'all_subsets.csv.gz');graphs=pd.read_csv(PRIOR/'graphs.csv')
    mapping=[];hashes={};orbit_rows=[];perm_rows=[];permutation_records=[];mask_orbit_tables=[]
    for key,q in c.groupby(KEY):
        cohort,network,n,seed=key;n=int(n);seed=int(seed)
        for t in sorted(q.time.unique()):
            p,edges,path=full_law(cohort,network,n,seed,t);counts=np.array([x.bit_count() for x in range(1<<n)])
            expected=float((p@counts).mean());rho=expected/n;nu=expected-1
            assert abs(q[q.time==t].mean_infected.iloc[0]-expected)<1e-12
            mapping.append(dict(zip(KEY,key))|dict(time=t,expected_infected=expected,rho=rho,expected_events=nu,
                      probability_saturated=float(p[:,-1].mean()),full_law_file=str(path.relative_to(STUDY))))
            hashes[str(path.relative_to(STUDY))]=hashlib.sha256(path.read_bytes()).hexdigest()
        # Exact graph automorphism orbits of masks, independent of time.
        G=nx.Graph();G.add_nodes_from(range(n));G.add_edges_from(edges.tolist());states=np.arange(1<<n);canonical=states.copy();automorphisms=0;burnside=0
        for perm in nx.algorithms.isomorphism.GraphMatcher(G,G).isomorphisms_iter():
            assert {tuple(sorted((perm[u],perm[v]))) for u,v in G.edges()}=={tuple(sorted(e)) for e in G.edges()}
            permutation_records.append(dict(zip(KEY,key))|dict(mapping=[perm[v] for v in range(n)]))
            cycles=0;seen=set()
            for u in range(n):
                if u not in seen:
                    cycles+=1;v=u
                    while v not in seen:seen.add(v);v=perm[v]
            burnside+=2**cycles
            mapped=np.zeros(1<<n,dtype=int)
            for v in range(n):mapped|=((states>>v)&1)<<perm[v]
            canonical=np.minimum(canonical,mapped);automorphisms+=1
        assert len(np.unique(canonical))*automorphisms==burnside
        mask_orbit_tables.append(pd.DataFrame(dict(**{k:np.repeat(v,1<<n) for k,v in zip(KEY,key)},mask=states,canonical=canonical,m=[v.bit_count() for v in states])))
        geom=sub[(sub.cohort==cohort)&(sub.network==network)&(sub.n==n)&(sub.seed==seed)&(sub.time==.001)].set_index('mask')
        for m in range(1,n):
            z=geom[geom.m==m];L=z.e+z.a;cut=z[z.c==z.c.min()];lex=cut[cut.e==cut.e.min()]
            for label,h in [('min_e_plus_a',z[L==L.min()]),('lex_cut_edges',lex)]:
                _,orbit_sizes=np.unique(canonical[h.index.to_numpy()],return_counts=True);k=len(h)
                orbit_rows.append(dict(zip(KEY,key))|dict(m=m,fraction=m/n,geometry_class=label,class_size=k,
                    automorphisms=automorphisms,orbits=len(orbit_sizes),largest_orbit=orbit_sizes.max(),single_orbit=len(orbit_sizes)==1,
                    symmetry_pair_fraction=float(sum(s*(s-1) for s in orbit_sizes)/(k*(k-1))) if k>1 else np.nan,
                    distinct_e_a_pairs=len(h[['e','a']].drop_duplicates()),distinct_cut_values=h.c.nunique(),
                    max_a=float(h.a.max()),min_a=float(h.a.min())))
        perm_rows.append(dict(zip(KEY,key))|dict(automorphisms=automorphisms,edges=G.number_of_edges(),max_degree=max(dict(G.degree()).values())))
        print(cohort,network,n,seed,'automorphisms',automorphisms,round(time.time()-start,1),flush=True)
    (ROOT/'graph_automorphisms.json').write_text(json.dumps(permutation_records,indent=2))
    pd.concat(mask_orbit_tables,ignore_index=True).to_csv(ROOT/'all_mask_orbits.csv.gz',index=False)
    physical=pd.DataFrame(mapping);physical.to_csv(ROOT/'physical_time_map.csv',index=False)
    orbit=pd.DataFrame(orbit_rows);orbit.to_csv(ROOT/'geometry_symmetry_orbits.csv',index=False);pd.DataFrame(perm_rows).to_csv(ROOT/'graph_symmetries.csv',index=False)
    c=c.merge(physical,on=KEY+['time'],validate='many_to_one')
    c['relative_regret']=c.delta_A/c.A_opt_A;c['design_gain']=c.A_opt_A-c.random_A;c['relative_gain']=c.design_gain/c.A_opt_A
    assert c.design_gain.min()>-1e-12
    c['retention_numerically_defined']=c.design_gain>TOL
    c['gain_retention']=np.where(c.retention_numerically_defined,(c.MI_opt_A-c.random_A)/c.design_gain,np.nan)
    c['retention_identity_error']=abs(c.gain_retention-(1-c.delta_A/c.design_gain))
    assert c.loc[c.retention_numerically_defined,'retention_identity_error'].max()<1e-7
    c['budget_band']=pd.cut(c.fraction,[0,.25,.5,.75,1],labels=BUDGET_LABELS,include_lowest=True)
    c['event_stage']=pd.cut(c.expected_events,[-1e-12,.1,.5,1,2,4,np.inf],labels=EVENT_LABELS)
    c['rho_stage']=pd.cut(c.rho,[0,.2,.4,.6,.8,.95,1+1e-12],labels=RHO_LABELS,include_lowest=True)
    c.to_csv(ROOT/'relative_conditions.csv',index=False)
    for name,keys in [('by_rho_budget',['rho_stage','budget_band']),('by_events_budget',['event_stage','budget_band']),
                      ('by_exact_budget',['n','m','rho_stage']),('by_cohort_rho',['cohort','rho_stage']),('by_time',['time']),
                      ('by_cohort',['cohort']),('by_rho',['rho_stage']),('by_events',['event_stage'])]:
        rows=[]
        for labels,group in c.groupby(keys,observed=True):
            if not isinstance(labels,tuple):labels=(labels,)
            rows.append(dict(zip(keys,labels))|summarize(group).to_dict())
        pd.DataFrame(rows).to_csv(ROOT/f'{name}.csv',index=False)
    # Per-graph maxima avoid treating repeated times as independent observations.
    pergraph=c.groupby(KEY+['rho_stage','budget_band'],observed=True).agg(max_relative_regret=('relative_regret','max'),
        max_absolute_regret=('delta_A','max'),evaluated_conditions=('delta_A','size')).reset_index()
    eligible=c[c.relative_gain>=.05]
    rr=eligible.groupby(KEY+['rho_stage','budget_band'],observed=True).agg(min_gain_retention=('gain_retention','min'),eligible_conditions=('gain_retention','size')).reset_index()
    pergraph.merge(rr,on=KEY+['rho_stage','budget_band'],how='left').to_csv(ROOT/'graph_stage_worstcases.csv',index=False)
    worst=[]
    for stagekey,group in c.groupby(['rho_stage','budget_band'],observed=True):
        for metric,select in [('relative_regret',lambda g:g[metric].idxmax()),('delta_A',lambda g:g[metric].idxmax()),('gain_retention',lambda g:g[metric].idxmin())]:
            pool=group if metric!='gain_retention' else group[group.relative_gain>=.05]
            if len(pool):worst.append(pool.loc[select(pool)].to_dict()|dict(extreme_metric=metric))
    pd.DataFrame(worst).to_csv(ROOT/'stage_worstcase_identities.csv',index=False)
    # Main adverse case, detailed posterior anatomy.
    case=c.loc[c.delta_A.idxmax()];cohort,network,n,seed=[case[k] for k in KEY];n=int(n);seed=int(seed);t=float(case.time);m=int(case.m)
    p,edges,path=full_law(cohort,network,n,seed,t);G=nx.Graph();G.add_nodes_from(range(n));G.add_edges_from(edges.tolist())
    candidates=sub[(sub.cohort==cohort)&(sub.network==network)&(sub.n==n)&(sub.seed==seed)&(sub.time==t)&(sub.m==m)].copy()
    candidates['MI_rank']=[1+int((candidates.mi>x+TOL).sum()) for x in candidates.mi]
    candidates['A_rank']=[1+int((candidates.accuracy>x+TOL).sum()) for x in candidates.accuracy]
    candidates.to_csv(ROOT/'exception_all_masks.csv',index=False)
    patterns=[];source_rows=[];summary=[];saved=dict(edges=edges,full_channel=p,time=t)
    for design,mask in [('MI',int(case.mi_mask)),('Accuracy',int(case.a_mask))]:
        sensors=[v for v in range(n) if mask>>v&1];q=old.direct_project(p,mask);joint=q/n;py=joint.sum(0);post=np.divide(joint,py,out=np.zeros_like(joint),where=py>0)
        H=-xlogy(post,post).sum(0);maximum=post.max(0);confusion=np.zeros((n,n))
        for y in range(1<<m):
            best=q[:,y].max();ties=np.flatnonzero(q[:,y]>=best-1e-12)
            confusion[:,ties]+=q[:,y,None]/len(ties)
            patterns.append(dict(design=design,mask=mask,pattern=y,bits=''.join(str(y>>j&1) for j in range(m)),
                infected_sensors=y.bit_count(),probability=py[y],posterior_entropy=H[y],posterior_max=maximum[y],
                accuracy_contribution=float(joint[:,y].max()),information_contribution=float(py[y]*(np.log(n)-H[y])),
                MAP_nodes=' '.join(map(str,ties)),posterior=' '.join(f'{v:.16g}' for v in post[:,y])))
        accuracy=q.max(0).sum()/n;mi=(xlogy(q,q)-xlogy(q,q.mean(0))).sum()/n
        assert abs(np.trace(confusion)/n-accuracy)<3e-12
        r=candidates[candidates['mask']==mask].iloc[0]
        summary.append(dict(design=design,mask=mask,sensors=' '.join(map(str,sensors)),outside=' '.join(str(v) for v in range(n) if v not in sensors),
            information=mi,accuracy=accuracy,conditional_entropy=float((py*H).sum()),MI_rank=int(r.MI_rank),A_rank=int(r.A_rank),
            e=int(r.e),c=int(r.c),a=int(r.a),degree_volume=sum(dict(G.degree())[v] for v in sensors),
            exact_MI_optima=int((candidates.mi>=candidates.mi.max()-TOL).sum()),exact_A_optima=int((candidates.accuracy>=candidates.accuracy.max()-TOL).sum())))
        for u in range(n):source_rows.append(dict(design=design,source=u,degree=G.degree(u),observed=u in sensors,correct_probability=confusion[u,u]))
        saved[design+'_posterior']=post;saved[design+'_probabilities']=py;saved[design+'_confusion']=confusion;saved[design+'_sensors']=np.array(sensors)
    np.savez_compressed(ROOT/'exception_posterior.npz',**saved)
    pd.DataFrame(summary).to_csv(ROOT/'exception_designs.csv',index=False);pp=pd.DataFrame(patterns);pp.to_csv(ROOT/'exception_patterns.csv',index=False)
    pp.groupby(['design','infected_sensors']).agg(probability=('probability','sum'),accuracy_contribution=('accuracy_contribution','sum'),
        information_contribution=('information_contribution','sum')).to_csv(ROOT/'exception_pattern_groups.csv')
    pd.DataFrame(source_rows).to_csv(ROOT/'exception_source_success.csv',index=False)
    (ROOT/'input_manifest.json').write_text(json.dumps({'full_laws':hashes,'comparison_sha256':hashlib.sha256((PRIOR/'comparisons.csv').read_bytes()).hexdigest(),
        'all_subsets_sha256':hashlib.sha256((PRIOR/'all_subsets.csv.gz').read_bytes()).hexdigest()},indent=2))
    cfg=dict(no_new_randomness=True,no_new_channels=True,source_graphs=96,source_conditions=len(c),physical_channels=len(physical),
        primary_gain_filter='(A_opt-A_random)/A_opt >= 0.05; fixed reporting convention, not preregistered',
        gain_filter_sensitivity=[.01,.05,.1],numeric_gain_zero_tolerance=TOL,rank_ties='Case rank is1+number of masks exceeding the score by1e-11; exact optimum multiplicities use1e-11',
        MAP_confusion_ties='Uniform allocation among conditional likelihood entries within1e-12 of max; accuracy separately uses actual maxima',
        runtime_seconds=time.time()-start,python=platform.python_version(),networkx=nx.__version__)
    (ROOT/'config.json').write_text(json.dumps(cfg,indent=2));print(json.dumps(cfg,indent=2))
if __name__=='__main__':run()
