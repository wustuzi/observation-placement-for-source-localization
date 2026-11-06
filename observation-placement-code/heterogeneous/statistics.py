import csv,json
import numpy as np
from run import ROOT,TIMES,write_csv

def read(p):
    rows=list(csv.DictReader(p.open()))
    for row in rows:
        for k,v in row.items():
            if k not in ['graph_id','family','layout']:
                try:row[k]=float(v)
                except ValueError:pass
    return rows

def main():
    cs=read(ROOT/'contrasts.csv');rows=read(ROOT/'conditions.csv');summary=[]
    for fam in ['ER','BA','WS']:
        for n in [100,250,500,1000,2000]:
            for tau in TIMES:
                sub=[x for x in cs if x['family']==fam and x['requested_n']==n and x['tau']==tau]
                r=dict(family=fam,requested_n=n,tau=tau,graphs=len(sub))
                for key in ['n','delta_I_cluster_minus_spread','delta_I_jackknife','scaled_delta_A','delta_A_heldout','mean_infected','infected_fraction']:
                    values=np.array([x[key] for x in sub]);r[key+'_mean']=values.mean();r[key+'_min']=values.min();r[key+'_max']=values.max()
                r['positive_delta_A_CP95']=sum(x['delta_A_lower_95']>0 for x in sub)
                # Approximate precision screen; not a finite-sample MI bound.
                r['positive_delta_I_precision_screen']=sum(x['delta_I_jackknife']>1.96*x['delta_I_se']+abs(x['delta_I_jackknife']-x['delta_I_cluster_minus_spread']) for x in sub)
                r['negative_delta_I_precision_screen']=sum(-x['delta_I_jackknife']>1.96*x['delta_I_se']+abs(x['delta_I_jackknife']-x['delta_I_cluster_minus_spread']) for x in sub)
                summary.append(r)
    write_csv(ROOT/'family_size_time_summary.csv',summary)
    uncertain=[x for x in cs if x['delta_A_lower_95']<=0]
    convergence=[]
    p=ROOT/'convergence'/'contrasts.csv'
    if p.exists():
        for new in read(p):
            old=next(x for x in cs if x['graph_id']==new['graph_id'] and x['tau']==new['tau'])
            convergence.append(dict(graph_id=new['graph_id'],tau=new['tau'],delta_I_8192=old['delta_I_jackknife'],delta_I_32768=new['delta_I_jackknife'],
                delta_I_difference=new['delta_I_jackknife']-old['delta_I_jackknife'],
                combined_delta_I_se=np.hypot(old['delta_I_se'],new['delta_I_se']),
                scaled_delta_A_8192=old['scaled_delta_A'],scaled_delta_A_32768=new['scaled_delta_A'],
                delta_A_CP95_positive_32768=new['delta_A_lower_95']>0))
        write_csv(ROOT/'convergence'/'comparison.csv',convergence)
    out=dict(accuracy_uncertain_comparisons=uncertain,convergence_comparisons=convergence,
             mi_precision_screen='abs(jackknife delta I) > 1.96*paired jackknife SE + abs(jackknife correction); descriptive sensitivity, not rigorous coverage',
             positive_information_screen=sum(x['positive_delta_I_precision_screen'] for x in summary),
             negative_information_screen=sum(x['negative_delta_I_precision_screen'] for x in summary))
    (ROOT/'statistical_audit.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
    print('CP unresolved:',[(x['graph_id'],x['tau']) for x in uncertain])
    print('MI sign precision screens:',out['positive_information_screen'],out['negative_information_screen'])
    print('Convergence conditions:',len(convergence))
if __name__=='__main__':main()
