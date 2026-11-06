"""Focused deterministic exact-channel analysis. See PROTOCOL.md before rerunning."""
from pathlib import Path
import hashlib, importlib.util, json, platform, sys, time
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.special import xlogy
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parent; STUDY=ROOT.parents[2]
spec=importlib.util.spec_from_file_location('archived',STUDY/'revision/partial_observation/experiments/run.py')
old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
TIMES=np.array([.001,.003,.01,.03,.05,.1,.2,.4,.8,1.6,3.2]); TOL=1e-11

def generator(n,edges):
    # Independently constructed column generator using vectorized bit states.
    states=np.arange(1<<n); rows=[]; cols=[]; data=[]; escape=np.zeros(1<<n)
    for v in range(n):
        neighbors=[int(b) if int(a)==v else int(a) for a,b in edges if v in (int(a),int(b))]
        rate=sum((states>>u)&1 for u in neighbors)*(((states>>v)&1)==0)
        good=np.flatnonzero(rate)
        rows.extend((good|(1<<v)).tolist());cols.extend(good.tolist());data.extend(rate[good].tolist());escape+=rate
    rows.extend(states.tolist());cols.extend(states.tolist());data.extend((-escape).tolist())
    return coo_matrix((data,(rows,cols)),shape=(1<<n,1<<n)).tocsr()

def joint_metrics(p):
    py=p.mean(0)
    return np.array([(xlogy(p,p)-xlogy(p,py)).sum()/len(p),p.max(0).sum()/len(p)])

def tied_rank(x,tol=TOL):
    # Numerically equal levels have mean ranks, even if symmetry leaves ulp noise.
    order=np.argsort(x); labels=np.zeros(len(x),int); level=0; anchor=x[order[0]]
    for k in order:
        if x[k]-anchor>tol: level+=1;anchor=x[k]
        labels[k]=level
    return rankdata(labels,method='average')

def spearman(x,y):
    rx=tied_rank(x);ry=tied_rank(y)
    if np.ptp(rx)==0 or np.ptp(ry)==0:return np.nan
    return float(np.corrcoef(rx,ry)[0,1])

def optimum(x,tol=TOL):return np.flatnonzero(x>=x.max()-tol)

def summary_class(meta,name,pred,truth,target,acc,score,mi):
    intersection=np.intersect1d(pred,truth)
    return dict(**meta,predictor=name,class_size=len(pred),true_optima=len(truth),
        overlap=bool(len(intersection)),contains_all_true=bool(np.isin(truth,pred).all()),
        all_pred_are_true=bool(np.isin(pred,truth).all()),true_coverage=len(intersection)/len(truth),
        jaccard=len(intersection)/len(np.union1d(pred,truth)),spearman=spearman(score,target),
        best_objective_regret=max(0.,target.max()-target[pred].max()),
        worst_objective_regret=max(0.,target.max()-target[pred].min()),
        best_accuracy_regret=max(0.,acc.max()-acc[pred].max()),
        worst_accuracy_regret=max(0.,acc.max()-acc[pred].min()),
        exact_MI_span=float(np.ptp(mi[pred])),exact_accuracy_span=float(np.ptp(acc[pred])))

def run():
    begin=time.time();(ROOT/'raw').mkdir(exist_ok=True)
    comparisons=[];prediction_rows=[];checks=[];graphs=[];manifest=[];ties=[];first=True
    archived=pd.read_csv(STUDY/'revision/partial_observation/experiments/all_subsets.csv.gz',
       usecols=['cohort','network','n','seed','time','mask','mi','top1']).set_index(['cohort','network','n','seed','time'])
    for cohort,folder in [('discovery',STUDY/'exact/raw'),('validation',STUDY/'revision/design/additional_raw')]:
      for path in sorted(folder.glob('*.npz')):
        z=np.load(path);n=z['probabilities'].shape[1]
        if n not in (8,10):continue
        network=path.stem.split('_')[0];seed=int(path.stem.split('seed')[1]);edges=z['edges']
        base=dict(cohort=cohort,network=network,n=n,seed=seed)
        manifest.append(dict(**base,path=str(path.relative_to(STUDY)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        masks=np.arange(1<<n);sizes=np.array([int(k).bit_count() for k in masks]);g=old.geometry(n,edges,masks)
        degree=np.bincount(edges.ravel(),minlength=n);ell=g[:,0]+g[:,2]
        graphs.append(dict(**base,edges=len(edges),degree_mean=degree.mean(),degree_std=degree.std(),degree_cv=degree.std()/degree.mean(),
                           degree_min=degree.min(),degree_max=degree.max()))
        missing=[t for t in TIMES if not np.any(np.isclose(z['times'],t,atol=1e-14,rtol=0))]
        newpath=ROOT/'raw'/f'{cohort}_{path.name}'
        if newpath.exists():
            zz=np.load(newpath);assert np.array_equal(zz['edges'],edges) and np.array_equal(zz['times'],missing)
            new=dict(zip(zz['times'],zz['probabilities']))
        else:
            Q=generator(n,edges);initial=np.zeros((1<<n,n));initial[1<<np.arange(n),np.arange(n)]=1
            new={t:expm_multiply(Q*t,initial,traceA=float(Q.diagonal().sum()*t)).T for t in missing}
            np.savez_compressed(newpath,times=np.array(missing),probabilities=np.array(list(new.values())),edges=edges)
        manifest.append(dict(**base,path=str(newpath.relative_to(STUDY)),sha256=hashlib.sha256(newpath.read_bytes()).hexdigest(),role='new_missing_ages'))
        graph_subsets=[]
        for t in TIMES:
            at=np.flatnonzero(np.isclose(z['times'],t,atol=1e-14,rtol=0));p=z['probabilities'][at[0]] if len(at) else new[t]
            cc=old.all_channels(p);arr=np.array([joint_metrics(cc[k]) for k in masks]);fullmi,fulla=arr[-1]
            meta=dict(**base,time=float(t));state_size=np.array([int(k).bit_count() for k in masks])
            mean_infected=float((p@state_size).mean());maxnorm=float(abs(p.sum(1)-1).max())
            projection=max(float(abs(cc[k]-old.direct_project(p,k)).max()) for k in [0,1,3,(1<<n)//3,(1<<n)-2,(1<<n)-1])
            min_gains=np.full(2,np.inf)
            for v in range(n):
                sub=masks[(masks&(1<<v))==0];min_gains=np.minimum(min_gains,(arr[sub|(1<<v)]-arr[sub]).min(0))
            legacy=np.nan
            if t in [.1,.4,.8,1.6,3.2]:
                ref=archived.loc[(cohort,network,n,seed,t)].sort_values('mask')
                legacy=float(abs(arr-ref[['mi','top1']].to_numpy()).max());assert legacy<1e-12
            independent=np.nan;tail=np.nan
            if t in [.001,.03,.2] and seed in (0,8) and n==8:
                independent_p,tail=old.uniformization(n,edges,t);independent=float(abs(independent_p-p).max());assert independent<2e-12
            assert maxnorm<1e-11 and projection<1e-12 and min_gains.min()>-1e-11 and p.min()>-1e-14
            # Direct independent geometry check on a varied fixed mask list.
            geomerr=0
            for k in [1,3,(1<<n)//3,(1<<n)-2]:
                observed={v for v in range(n) if k&(1<<v)};unobserved=set(range(n))-observed
                ee=sum(u in observed and v in observed for u,v in edges)
                cut=sum((u in observed)!=(v in observed) for u,v in edges)
                aa=min(sum((u==v and w in observed) or (w==v and u in observed) for u,w in edges) for v in unobserved)
                geomerr=max(geomerr,float(abs(g[k]-[ee,cut,aa]).max()))
            assert geomerr==0
            checks.append(dict(**meta,max_normalization_error=maxnorm,direct_projection_error=projection,legacy_error=legacy,
                               min_MI_inclusion_gain=min_gains[0],min_A_inclusion_gain=min_gains[1],uniformization_error=independent,
                               uniformization_tail=tail,independent_geometry_error=geomerr))
            graph_subsets.append(pd.DataFrame(dict(**{k:np.repeat(v,len(masks)) for k,v in meta.items()},mask=masks,m=sizes,
                              mi=arr[:,0],accuracy=arr[:,1],e=g[:,0],c=g[:,1],a=g[:,2])))
            for m in range(1,n):
                eligible=masks[sizes==m];mi=arr[eligible,0];a=arr[eligible,1];e=g[eligible,0];c=g[eligible,1];aa=g[eligible,2];L=ell[eligible]
                I0=np.log(n)-(n-m)/n*np.log(n-m)
                mi_score=I0-c/n*t*np.log(1/t)+t/n*(c*(np.log(n-m)-1)-2*e*np.log(2))
                a_score=(m+1)/n-L/n*t
                miopt=optimum(mi);aopt=optimum(a)
                # Conservative MI accuracy is the highest actual accuracy in its MI tolerance class.
                im=miopt[np.argmax(a[miopt])]
                # Stabilize the representative of the accuracy-optimum class.
                # Secondary MI values within TOL are numerical ties, then use
                # the smallest mask. Conservative MI accuracy above stays exact.
                secondary_MI_max=mi[aopt].max()
                a_secondary=aopt[mi[aopt]>=secondary_MI_max-TOL]
                ia=int(a_secondary[np.argmin(eligible[a_secondary])])
                delta=max(0.,a.max()-a[im]);budget=max(0.,fulla-a.max());total=budget+delta
                cut=optimum(-c);lex=cut[e[cut]==e[cut].min()];finite=optimum(mi_score);accuracy=optimum(-L)
                meta_m=dict(**meta,m=m,fraction=m/n)
                for name,pred,truth,target,score in [('min_cut',cut,miopt,mi,-c),('lex_cut_edges',lex,miopt,mi,-c*(len(edges)+1)-e),
                                ('MI_first_order',finite,miopt,mi,mi_score),('min_e_plus_a',accuracy,aopt,a,-L)]:
                    prediction_rows.append(summary_class(meta_m,name,pred,truth,target,a,score,mi))
                lo=t/n*(L[lex].min()-L.min());hi=t/n*(L[lex].max()-L.min())
                flo=t/n*(L[finite].min()-L.min());fhi=t/n*(L[finite].max()-L.min())
                oracle=t/n*(L[im]-L[ia]);a_residual=float(abs(a-a_score).max());mi_residual=float(abs(mi-mi_score).max())
                comparisons.append(dict(**meta_m,subsets=len(eligible),mean_infected=mean_infected,infected_fraction=mean_infected/n,
                    degree_cv=degree.std()/degree.mean(),degree_mean=degree.mean(),full_MI=fullmi,full_A=fulla,random_MI=mi.mean(),random_A=a.mean(),
                    MI_opt_MI=mi.max(),MI_opt_A=a[im],A_opt_A=a.max(),A_opt_MI=mi[ia],mi_mask=eligible[im],a_mask=eligible[ia],
                    MI_ties=len(miopt),A_ties=len(aopt),optima_overlap=bool(len(np.intersect1d(miopt,aopt))),
                    delta_A=delta,delta_A_worst_MI_tie=max(0.,a.max()-a[miopt].min()),budget_gap=budget,total_gap=total,
                    objective_fraction=delta/total if total>=.01 else np.nan,objective_fraction_1e3=delta/total if total>=.001 else np.nan,
                    objective_fraction_1e6=delta/total if total>=.000001 else np.nan,
                    geometry_low=lo,geometry_high=hi,geometry_width=hi-lo,geometry_interval_miss=max(lo-delta,delta-hi,0.),
                    geometry_interval_contains=bool(lo-TOL<=delta<=hi+TOL),finite_geometry_low=flo,finite_geometry_high=fhi,
                    finite_geometry_miss=max(flo-delta,delta-fhi,0.),
                    oracle_conditioned=oracle,oracle_error=delta-oracle,oracle_e=t/n*(e[im]-e[ia]),oracle_a=t/n*(aa[im]-aa[ia]),
                    MI_e=e[im],MI_a=aa[im],MI_c=c[im],A_e=e[ia],A_a=aa[ia],A_c=c[ia],
                    mean_a=aa.mean(),positive_a_fraction=(aa>0).mean(),a_opt_min=aa[accuracy].min(),a_opt_max=aa[accuracy].max(),
                    min_L=L.min(),lex_min_L=L[lex].min(),lex_max_L=L[lex].max(),
                    max_A_linear_residual=a_residual,max_MI_expansion_residual=mi_residual,
                    oracle_error_bound=2*a_residual+TOL,geometry_bound_expanded_contains=bool(lo-2*a_residual-TOL<=delta<=hi+2*a_residual+TOL),
                    residual_A_scaled=a_residual/t**2,residual_MI_scaled=mi_residual/(t*t*(1+abs(np.log(t)))),
                    budget_intercept=(n-m-1)/n,budget_first_order=(n-m-1)/n-(len(edges)-L.min())*t/n))
                assert abs(delta-oracle)<=2*a_residual+2*TOL
                assert abs((fulla-a[im])-total)<2e-11
                for tol in [1e-12,1e-10]:
                    mm=optimum(mi,tol);zz=optimum(a,tol);d=max(0.,a.max()-a[mm].max())
                    ties.append(dict(**meta_m,tolerance=tol,delta_A=d,MI_ties=len(mm),A_ties=len(zz),optima_overlap=bool(len(np.intersect1d(mm,zz)))))
        pd.concat(graph_subsets,ignore_index=True).to_csv(ROOT/'all_subsets.csv.gz',mode='w' if first else 'a',index=False,header=first,compression='gzip');first=False
        print(cohort,path.stem,round(time.time()-begin,1),'s',flush=True)
    for name,data in [('comparisons',comparisons),('predictor_validity',prediction_rows),('validation',checks),('graphs',graphs),('tie_sensitivity',ties)]:
        pd.DataFrame(data).to_csv(ROOT/f'{name}.csv',index=False)
    json.dump(manifest,open(ROOT/'input_manifest.json','w'),indent=2)
    json.dump(dict(times=TIMES.tolist(),source_prior='uniform over vertices',dynamics='continuous-time SI, beta=1',
        original_cohorts={'discovery':list(range(8)),'validation':list(range(8,16))},sizes=[8,10],families=['ER','BA','WS'],
        new_graph_seeds=None,new_randomness=False,tie_tolerance=TOL,rank_ties='Adjacent sorted scores grouped from a fixed group anchor within absolute tolerance, average ranks',
        geometry_classes='Exact integer cut/edge/ell ties; no arbitrary theoretical tie-breaking',
        MI_regret='Accuracy maximum minus exact best accuracy among masks within tolerance of maximum MI',
        A_representative='Within absolute 1e-11 of max accuracy, then within absolute 1e-11 of best MI in that class, then smallest integer mask; secondary MI ulp ties do not select arbitrary representatives',
        oracle_conditioned='Uses actual exact optimizing masks, not a prospective prediction',
        statistical_scope='Descriptive condition-weighted summaries; no graph-independent uncertainty intervals; seed coupling audited in prior revision',
        subset_rows=int(sum((1<<g['n'])*len(TIMES) for g in graphs)),conditions=len(comparisons),full_channels=len(checks),
        new_full_channels=sum(len(np.load(p)['times']) for p in (ROOT/'raw').glob('*.npz')),
        runtime_seconds=time.time()-begin,python=sys.version,platform=platform.platform(),
        versions={'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__}),open(ROOT/'config.json','w'),indent=2)

if __name__=='__main__':run()
