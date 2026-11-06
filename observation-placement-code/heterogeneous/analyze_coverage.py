"""Read-only analysis of the locally generated MC observation channels.

This is a diagnostic decomposition, not a new experiment or causal analysis.
Neither the raw samples nor the random-network simulator/configs are changed.
"""
from pathlib import Path
import sys,os,csv,json,hashlib
ROOT=Path(__file__).resolve().parent
MC=ROOT.parent/'random_mc'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'mplcache'))
sys.path.insert(0,str(MC))
from run import read_binary,TIMES,metrics
import numpy as np
from scipy.special import xlogy
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def decomposition(counts):
    p=counts/counts.sum(axis=-1,keepdims=True)
    q=p.mean(axis=-2)
    Hout=-xlogy(q,q).sum(axis=-1)
    Hcond=-xlogy(p,p).sum(axis=-1).mean(axis=-1)
    contrib=(xlogy(p,p)-xlogy(p,q[...,None,:])).mean(axis=-2)
    nonempty=1-q[...,0]
    K=np.divide(contrib[...,1:].sum(axis=-1),nonempty,out=np.zeros_like(nonempty),where=nonempty>0)
    return dict(Hout=Hout,Hcond=Hcond,I=Hout-Hcond,q_nonempty=nonempty,p11=q[...,3],
        sum_infected_marginals=q[...,1]+q[...,2]+2*q[...,3],
        overlap_fraction=np.divide(q[...,3],nonempty,out=np.zeros_like(nonempty),where=nonempty>0),
        I_empty=contrib[...,0],I_nonempty=contrib[...,1:].sum(axis=-1),K_nonempty=K)

def csvwrite(name,rows):
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    rawhash={};configs={};rows=[];cs=[];errors=[]
    original={(r['graph_id'],r['layout'],float(r['tau'])):r for r in csv.DictReader((MC/'conditions.csv').open())}
    for config in sorted((MC/'configs').glob('*.json')):
        c=json.loads(config.read_text());raw=MC/'raw'/f"{c['graph_id']}.bin"
        rawhash[c['graph_id']]=hashlib.sha256(raw.read_bytes()).hexdigest()
        configs[c['graph_id']]=hashlib.sha256(config.read_bytes()).hexdigest()
        assert rawhash[c['graph_id']]==c['raw_sha256']
        counts,_,_,R=read_binary(raw);B=len(counts);total=counts.sum(axis=0)
        d=decomposition(total);loo=[decomposition(total-block) for block in counts]
        jack={k:B*d[k]-(B-1)*np.mean([v[k] for v in loo],axis=0) for k in ['Hout','Hcond','I']}
        assert np.max(abs(d['I']-metrics(total)[0]))<3e-14
        for j,layout in enumerate(c['layouts']):
            for t,tau in enumerate(TIMES):
                base={key:c[key] for key in ['graph_id','family','requested_n','n','replicate']}
                base.update(layout,tau=float(tau),samples=B*R,degree_sum=layout['degree_u']+layout['degree_v'])
                base.update({key:float(value[j,t]) for key,value in d.items()})
                base.update({key+'_jackknife':float(value[j,t]) for key,value in jack.items()})
                old=original[(c['graph_id'],layout['layout'],float(tau))]
                err=abs(base['I']-float(old['information_plugin']))
                errj=abs(base['I_jackknife']-float(old['information_jackknife']))
                assert max(err,errj)<2e-13
                assert abs(base['sum_infected_marginals']-base['q_nonempty']-base['p11'])<1e-14
                errors.extend([err,errj]);rows.append(base)
        cluster,spread=c['layouts'][:2]
        for t,tau in enumerate(TIMES):
            qc,qs=d['q_nonempty'][:2,t];kc,ks=d['K_nonempty'][:2,t]
            coverage=(qc-qs)*(kc+ks)/2
            specificity=(kc-ks)*(qc+qs)/2
            delta_empty=d['I_empty'][0,t]-d['I_empty'][1,t]
            delta=d['I'][0,t]-d['I'][1,t]
            assert abs(delta-delta_empty-coverage-specificity)<3e-14
            r={key:c[key] for key in ['graph_id','family','requested_n','n','replicate']}
            r.update(tau=float(tau),delta_I=float(delta),
                delta_I_jackknife=float(jack['I'][0,t]-jack['I'][1,t]),
                delta_Hout=float(d['Hout'][0,t]-d['Hout'][1,t]),
                delta_Hcond=float(d['Hcond'][0,t]-d['Hcond'][1,t]),
                delta_Hout_jackknife=float(jack['Hout'][0,t]-jack['Hout'][1,t]),
                delta_Hcond_jackknife=float(jack['Hcond'][0,t]-jack['Hcond'][1,t]),
                delta_q_nonempty=float(qc-qs),delta_K_nonempty=float(kc-ks),
                delta_p11=float(d['p11'][0,t]-d['p11'][1,t]),
                delta_sum_infected_marginals=float(d['sum_infected_marginals'][0,t]-d['sum_infected_marginals'][1,t]),
                delta_I_empty=float(delta_empty),coverage_component=float(coverage),
                specificity_component=float(specificity),
                cluster_q=float(qc),spread_q=float(qs),cluster_K=float(kc),spread_K=float(ks),
                cluster_degree_sum=cluster['degree_u']+cluster['degree_v'],
                spread_degree_sum=spread['degree_u']+spread['degree_v'],
                delta_degree_sum=cluster['degree_u']+cluster['degree_v']-spread['degree_u']-spread['degree_v'],
                delta_cut=cluster['c_M']-spread['c_M'],cluster_distance=cluster['distance'],spread_distance=spread['distance'])
            r['delta_internal_edges']=cluster['e_M']-spread['e_M']
            r['first_order_delta_I']=float(tau/c['n']*(r['delta_cut']*(np.log((c['n']-2)*tau)-1)-2*r['delta_internal_edges']*np.log(2)))
            r['first_order_delta_q']=float(r['delta_cut']*tau/c['n'])
            for key in ['I','Hout','Hcond','q_nonempty','p11','K_nonempty']:
                delta_loo=np.array([z[key][0,t]-z[key][1,t] for z in loo])
                r['delta_'+key+'_jackknife_se']=float(np.sqrt((B-1)/B*np.square(delta_loo-delta_loo.mean()).sum()))
            cs.append(r)
    csvwrite('layout_decomposition.csv',rows);csvwrite('paired_decomposition.csv',cs)
    summaries=[];corrs=[]
    mean_keys=['delta_I','delta_I_jackknife','delta_Hout','delta_Hcond','delta_q_nonempty','delta_K_nonempty',
        'delta_p11','delta_I_empty','coverage_component','specificity_component','delta_degree_sum','delta_cut']
    for fam in ['ER','BA','WS']:
        for tau in TIMES:
            sub=[r for r in cs if r['family']==fam and r['tau']==tau]
            s=dict(family=fam,tau=float(tau),graphs=len(sub),
                positive_delta_I=sum(r['delta_I']>0 for r in sub),
                positive_delta_q=sum(r['delta_q_nonempty']>0 for r in sub),
                positive_delta_K=sum(r['delta_K_nonempty']>0 for r in sub),
                positive_delta_Hout=sum(r['delta_Hout']>0 for r in sub),
                positive_delta_Hcond=sum(r['delta_Hcond']>0 for r in sub))
            s['first_order_information_sign_agreement']=sum(np.sign(r['first_order_delta_I'])==np.sign(r['delta_I']) for r in sub)
            s['first_order_information_abs_error_mean']=float(np.mean([abs(r['first_order_delta_I']-r['delta_I']) for r in sub]))
            s.update({key+'_mean':float(np.mean([r[key] for r in sub])) for key in mean_keys})
            summaries.append(s)
            for x,y in [('delta_q_nonempty','delta_I'),('delta_degree_sum','delta_I'),('delta_cut','delta_I'),('delta_degree_sum','delta_q_nonempty')]:
                rho=float(spearmanr([r[x] for r in sub],[r[y] for r in sub]).statistic)
                corrs.append(dict(family=fam,tau=float(tau),graphs=len(sub),x=x,y=y,spearman_rho=rho,
                    interpretation='descriptive across 15 size/seed cases, not causal and not an independent population test'))
    csvwrite('family_time_summary.csv',summaries);csvwrite('descriptive_correlations.csv',corrs)
    selected=[r for r in cs if r['requested_n']==1000]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(9.3,5.6),sharex=True,layout='constrained')
    defs=[('delta_Hout','Output entropy','#377eb8'),('delta_Hcond','Conditional entropy','#e41a1c'),('delta_I','Information','black')]
    defs2=[('coverage_component','Coverage component','#4daf4a'),('specificity_component','Specificity component','#984ea3'),('delta_I_empty','Empty-report component','#ff7f00'),('delta_I','Total','black')]
    for col,fam in enumerate(['ER','BA','WS']):
        sub=[r for r in selected if r['family']==fam]
        for row,items in enumerate([defs,defs2]):
            for key,label,color in items:
                v=np.array([[r[key] for r in sub if r['tau']==t] for t in TIMES])
                axes[row,col].plot(TIMES,v.mean(axis=1),'-o',color=color,ms=3,label=label)
            axes[row,col].axhline(0,color='.6',lw=.7);axes[row,col].set_xscale('log')
            axes[row,col].set_xticks(TIMES,labels=['0.1','0.5','1','2','4'])
            axes[row,col].text(.02,1.02,'('+chr(97+row*3+col)+')',va='bottom',transform=axes[row,col].transAxes)
        axes[0,col].set_title(fam);axes[1,col].set_xlabel(r'$\tau$')
    axes[0,0].set_ylabel('Cluster minus spread (nats)');axes[1,0].set_ylabel('Contribution to information difference (nats)')
    axes[0,2].legend(frameon=False,fontsize=7,loc='best');axes[1,2].legend(frameon=False,fontsize=6.6,loc='best')
    fig.savefig(ROOT/'coverage_decomposition.pdf');fig.savefig(ROOT/'coverage_decomposition.png',dpi=190);plt.close(fig)
    summary={'local_samples_only':True,'analysis_type':'post hoc descriptive decomposition; no causal attribution',
        'graphs':len(rawhash),'layout_time_rows':len(rows),'paired_time_rows':len(cs),
        'max_original_MI_reconstruction_error':max(errors),'raw_sha256':rawhash,'config_sha256':configs,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'coverage_identity':'I=I_empty+q*K_nonempty; delta=q_avg*delta_K+K_avg*delta_q+delta_I_empty',
        'entropy_identity':'delta_I=delta_Hout-delta_Hcond','figure_requested_N':1000,
        'figure_graphs_per_family':3,'figure_estimators':'empirical-channel plug-in, no clipping or sign selection'}
    (ROOT/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('diagnostics complete',len(rows),len(cs),'max MI discrepancy',max(errors))

if __name__=='__main__':main()
