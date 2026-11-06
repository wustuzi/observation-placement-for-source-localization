"""Graph-level summaries and percentile bootstrap for the matched-edge cohort."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
BOOTSTRAPS=2000
BOOTSTRAP_SEED=2026102810219507
TOL=1e-11


def mean_ci(values,indices):
    a=np.asarray(values,dtype=float)
    valid=np.isfinite(a)
    if not valid.any():
        return np.nan,np.nan,np.nan,0
    samples=a[indices]
    counts=np.isfinite(samples).sum(axis=1)
    means=np.divide(np.nansum(samples,axis=1),counts,
                    out=np.full(len(indices),np.nan),where=counts>0)
    bounds=np.nanquantile(means,[.025,.975])
    return float(a[valid].mean()),float(bounds[0]),float(bounds[1]),int(valid.sum())


def summarize_rows(frame,grouping,metrics,rng):
    rows=[]
    for key,group in frame.groupby(grouping,observed=True,sort=True):
        if not isinstance(key,tuple):
            key=(key,)
        assert not group.graph_id.duplicated().any(),(key,'non-independent rows')
        indices=rng.integers(0,len(group),size=(BOOTSTRAPS,len(group)))
        row=dict(zip(grouping,key))|dict(graphs=len(group))
        for metric in metrics:
            mean,lower,upper,valid=mean_ci(group[metric],indices)
            row.update({metric+'_mean':mean,metric+'_ci025':lower,
                        metric+'_ci975':upper,metric+'_valid_graphs':valid})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    c=pd.read_csv(RESULTS/'conditions.csv')
    graphs=pd.read_csv(RESULTS/'graphs.csv')
    checks=pd.read_csv(RESULTS/'checks.csv')
    generation=json.loads((RESULTS/'graph_manifest.json').read_text())
    rng=np.random.default_rng(BOOTSTRAP_SEED)
    assert len(graphs)==300 and graphs.graph_id.nunique()==300
    assert c.groupby('graph_id').size().eq(27).all()
    assert graphs.edge_count.eq(16).all() and graphs.mean_degree.eq(3.2).all()
    assert len({tuple(g['spawn_key']) for g in generation})==300
    assert c.design_gain.min()>-TOL and c.relative_regret.min()>-TOL
    defined=c.retention_numerically_defined
    identity=abs(c.loc[defined,'gain_retention']-(1-c.loc[defined,'delta_A']/c.loc[defined,'design_gain']))
    assert identity.max()<1e-7
    c['eligible_retention']=c.gain_retention.where(c.relative_gain>=.05)
    c['eligible_negative_retention']=np.where(c.relative_gain>=.05,(c.gain_retention < -TOL).astype(float),np.nan)
    c['rho_stage']=pd.cut(c.rho,[0,.2,.4,.6,.8,1.0000001],labels=['(0,.2]','(.2,.4]','(.4,.6]','(.6,.8]','(.8,1]'])
    c['budget_band']=pd.cut(c.fraction,[0,.25,.5,.75,1],labels=['(0,.25]','(.25,.5]','(.5,.75]','(.75,1)'])
    exact=summarize_rows(c,['family','tau','m'],
                         ['delta_A','relative_regret','eligible_retention','eligible_negative_retention'],rng)
    descriptive=c.groupby(['family','tau','m'],observed=True).agg(
        max_absolute_regret=('delta_A','max'),max_relative_regret=('relative_regret','max'),
        median_relative_regret=('relative_regret','median'),min_eligible_retention=('eligible_retention','min'),
        median_eligible_retention=('eligible_retention','median'),
        gain_undefined=('retention_numerically_defined',lambda x:int((~x).sum())),
        rho_mean=('rho','mean'),rho_min=('rho','min'),rho_max=('rho','max')).reset_index()
    exact.merge(descriptive,on=['family','tau','m'],validate='one_to_one').to_csv(RESULTS/'by_family_time_budget.csv',index=False)

    grouped=c.groupby(['graph_id','family','tau'],observed=True).agg(
        mean_relative_regret=('relative_regret','mean'),
        max_relative_regret=('relative_regret','max'),
        mean_absolute_regret=('delta_A','mean'),max_absolute_regret=('delta_A','max'),
        mean_eligible_retention=('eligible_retention','mean'),
        min_eligible_retention=('eligible_retention','min'),
        eligible_budgets=('eligible_retention','count'),
        rho=('rho','first'),expected_events=('expected_events','first')).reset_index()
    grouped.to_csv(RESULTS/'graph_time_summary.csv',index=False)
    times=summarize_rows(grouped,['family','tau'],
                        ['mean_relative_regret','max_relative_regret','mean_absolute_regret',
                         'max_absolute_regret','mean_eligible_retention','min_eligible_retention','rho','expected_events'],rng)
    times.to_csv(RESULTS/'by_family_time.csv',index=False)

    # Each graph contributes one statistic per occupied physical-stage/budget cell.
    stage=c.groupby(['graph_id','family','rho_stage','budget_band'],observed=True).agg(
        max_relative_regret=('relative_regret','max'),max_absolute_regret=('delta_A','max'),
        min_eligible_retention=('eligible_retention','min'),
        eligible_conditions=('eligible_retention','count'),
        conditions=('relative_regret','size'),rho_min=('rho','min'),rho_max=('rho','max')).reset_index()
    stage.to_csv(RESULTS/'graph_stage_budget.csv',index=False)
    summarize_rows(stage,['family','rho_stage','budget_band'],
                   ['max_relative_regret','max_absolute_regret','min_eligible_retention'],rng).to_csv(
                       RESULTS/'by_family_physical_stage_budget.csv',index=False)

    overall=c.groupby(['graph_id','family'],observed=True).agg(
        mean_relative_regret=('relative_regret','mean'),max_relative_regret=('relative_regret','max'),
        mean_absolute_regret=('delta_A','mean'),max_absolute_regret=('delta_A','max'),
        mean_eligible_retention=('eligible_retention','mean'),min_eligible_retention=('eligible_retention','min'),
        eligible_conditions=('eligible_retention','count')).reset_index()
    overall.to_csv(RESULTS/'per_graph_summary.csv',index=False)
    family_summary=summarize_rows(overall,['family'],
                   ['mean_relative_regret','max_relative_regret','mean_absolute_regret',
                    'max_absolute_regret','mean_eligible_retention','min_eligible_retention'],rng)
    family_summary.to_csv(RESULTS/'by_family.csv',index=False)

    sensitivity=[]
    for family,group in c.groupby('family'):
        for threshold in [.01,.05,.1]:
            subset=group[group.relative_gain>=threshold]
            sensitivity.append(dict(family=family,relative_gain_threshold=threshold,
                eligible_conditions=len(subset),eligible_graphs=subset.graph_id.nunique(),
                negative_retention=int((subset.gain_retention < -TOL).sum()),
                negative_retention_fraction=float((subset.gain_retention < -TOL).mean()) if len(subset) else np.nan,
                min_retention=subset.gain_retention.min(),median_retention=subset.gain_retention.median()))
    pd.DataFrame(sensitivity).to_csv(RESULTS/'retention_filter_sensitivity.csv',index=False)

    worst=[]
    for family,group in c.groupby('family'):
        for metric in ['delta_A','relative_regret','eligible_retention']:
            index=group[metric].idxmin() if metric=='eligible_retention' else group[metric].idxmax()
            worst.append(group.loc[index].to_dict()|dict(extreme_metric=metric))
    pd.DataFrame(worst).to_csv(RESULTS/'observed_extremes.csv',index=False)

    duplicates={}
    for family in graphs.family.unique():
        signatures=[tuple(tuple(e) for e in g['edges']) for g in generation if g['family']==family]
        duplicates[family]=len(signatures)-len(set(signatures))
    summary=dict(graphs=300,graphs_per_family=100,conditions=len(c),full_channels=len(checks),
        exact_labelled_duplicate_draws_retained=duplicates,
        all_seed_spawn_keys_unique=True,
        graph_generation_rejections=sum(g['attempts']-1 for g in generation),
        max_normalization_error=float(checks.normalization_error.max()),
        max_direct_projection_error=float(checks.direct_projection_error.max()),
        max_independent_uniformization_error=float(checks.uniformization_error.max()),
        min_MI_coordinate_gain=float(checks.min_MI_coordinate_gain.min()),
        min_A_coordinate_gain=float(checks.min_A_coordinate_gain.min()),
        max_tie_sensitivity=float(np.maximum(abs(c.delta_A-c.delta_A_tol1e12),abs(c.delta_A-c.delta_A_tol1e10)).max()),
        rho_min=float(c.rho.min()),rho_max=float(c.rho.max()),
        max_saturation_probability=float(c.saturation_probability.max()),
        max_absolute_regret=float(c.delta_A.max()),max_relative_regret=float(c.relative_regret.max()),
        relative_gain_filter=.05,eligible_retention_conditions=int(c.eligible_retention.notna().sum()),
        negative_retention_conditions=int((c.eligible_retention < -TOL).sum()),
        min_eligible_retention=float(c.eligible_retention.min()),
        undefined_gain_conditions=int((~c.retention_numerically_defined).sum()),
        bootstrap_resamples=BOOTSTRAPS,bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_unit='whole graph, independently within each family; never masks or times',
        bootstrap_interval='pointwise percentile interval [2.5%,97.5%] for the displayed graph-average statistic; not simultaneous',
        stage_estimand='mean of per-graph extrema over the sampled tau points and budgets in an occupied physical-stage cell; conditional on stage coverage',
        retention_estimand='mean among graphs with at least one eligible condition when per-graph conditional means/extrema are used',
        no_prospective_holdout_claim=True,
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__))
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((RESULTS/'raw').glob('*.npz'))}
    (RESULTS/'raw_sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    print(family_summary[['family','mean_absolute_regret_mean','mean_absolute_regret_ci025','mean_absolute_regret_ci975',
                          'mean_relative_regret_mean','max_relative_regret_mean','min_eligible_retention_mean']].to_string(index=False))


if __name__=='__main__':
    main()
