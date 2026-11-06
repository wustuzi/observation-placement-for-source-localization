import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,TIMES,read_binary,analyze,write_csv

def rebuild():
    rows=[];contrasts=[]
    for p in sorted((ROOT/'configs').glob('*.json')):
        c=json.loads(p.read_text());r,d=analyze(c,ROOT/'raw'/f"{c['graph_id']}.bin");rows+=r;contrasts+=d
    write_csv(ROOT/'conditions.csv',rows);write_csv(ROOT/'contrasts.csv',contrasts)
    return rows,contrasts

def main():
    rows,cs=rebuild();figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    colors=plt.get_cmap('viridis')(np.linspace(.05,.88,5));sizes=[100,250,500,1000,2000]
    fig,axs=plt.subplots(2,3,figsize=(10,5.8),sharex=True)
    for j,fam in enumerate(['ER','BA','WS']):
        for n,color in zip(sizes,colors):
            subset=[x for x in cs if x['family']==fam and x['requested_n']==n]
            for row,key in [(0,'delta_I_jackknife'),(1,'scaled_delta_A')]:
                vals=np.array([[x[key] for x in subset if x['tau']==tau] for tau in TIMES])
                axs[row,j].plot(TIMES,vals.mean(axis=1),'-o',ms=3,lw=1,color=color,label=str(n))
                axs[row,j].fill_between(TIMES,vals.min(axis=1),vals.max(axis=1),color=color,alpha=.12,lw=0)
        for row in range(2):
            axs[row,j].axhline(0,color='.4',ls=':',lw=.8);axs[row,j].set_xscale('log');axs[row,j].set_xticks(TIMES,labels=['0.1','0.5','1','2','4'])
        axs[0,j].set_title(fam);axs[1,j].set_xlabel(r'$\tau$')
    axs[0,0].set_ylabel(r'$I_{\rm cluster}-I_{\rm spread}$ (nats)')
    axs[1,0].set_ylabel(r'$N(A_{\rm spread}-A_{\rm cluster})$')
    axs[0,2].legend(title='Requested N',fontsize=7,title_fontsize=8,frameon=False,ncol=2)
    fig.tight_layout();fig.savefig(figdir/'topology_comparison.pdf');fig.savefig(figdir/'topology_comparison.png',dpi=200);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(10,5.8),sharex=True)
    for j,fam in enumerate(['ER','BA','WS']):
        subset=[x for x in rows if x['family']==fam and x['requested_n']==1000]
        for label,color in [('cluster','#c13b36'),('spread','#276bb0'),('random','#777777')]:
            for rr,key in [(0,'information_jackknife'),(1,'scaled_accuracy_plugin')]:
                vals=[]
                for tau in TIMES:
                    byrep=[]
                    for rep in range(3):
                        selected=[x[key] for x in subset if x['tau']==tau and x['replicate']==rep and (x['layout']==label or label=='random' and x['layout'].startswith('random_'))]
                        byrep.append(np.mean(selected))
                    vals.append(byrep)
                vals=np.array(vals);axs[rr,j].plot(TIMES,vals.mean(axis=1),'-o',ms=3,color=color,label=label)
                axs[rr,j].fill_between(TIMES,vals.min(axis=1),vals.max(axis=1),alpha=.12,color=color,lw=0)
        for rr in range(2):axs[rr,j].set_xscale('log');axs[rr,j].set_xticks(TIMES,labels=['0.1','0.5','1','2','4'])
        axs[0,j].set_title(fam);axs[1,j].set_xlabel(r'$\tau$')
    axs[0,0].set_ylabel('Source information (nats)');axs[1,0].set_ylabel(r'$N A_1$ (plug-in)')
    axs[0,2].legend(frameon=False);fig.tight_layout();fig.savefig(figdir/'topology_baselines.pdf');fig.savefig(figdir/'topology_baselines.png',dpi=200);plt.close(fig)
    summary={}
    for fam in ['ER','BA','WS']:
        x=[r for r in cs if r['family']==fam]
        reversing=[]
        for gid in sorted(set(r['graph_id'] for r in x)):
            z=sorted([r for r in x if r['graph_id']==gid],key=lambda r:r['tau'])
            signs=[int(np.sign(r['delta_I_jackknife'])) for r in z]
            if any(signs[i]<0 and signs[j]>0 and signs[k]<0 for i in range(5) for j in range(i+1,5) for k in range(j+1,5)):
                reversing.append(gid)
        summary[fam]=dict(graphs=len(set(r['graph_id'] for r in x)),conditions=len(x),
            positive_delta_I=sum(r['delta_I_jackknife']>0 for r in x),negative_delta_I=sum(r['delta_I_jackknife']<0 for r in x),
            positive_delta_A_plugin=sum(r['delta_A_spread_minus_cluster']>0 for r in x),
            positive_delta_A_heldout=sum(r['delta_A_heldout']>0 for r in x),
            delta_A_lower95_positive=sum(r['delta_A_lower_95']>0 for r in x),
            minus_plus_minus_information_graphs=reversing)
    summary['overall']=dict(conditions=len(rows),contrasts=len(cs),
        n_range=[min(r['n'] for r in rows),max(r['n'] for r in rows)],
        max_abs_MI_jackknife_correction=max(abs(r['information_jackknife']-r['information_plugin']) for r in rows),
        max_abs_MI_half_full=max(abs(r['information_half_samples']-r['information_plugin']) for r in rows),
        max_abs_scaled_A_half_full=max(r['n']*abs(r['accuracy_half_samples']-r['accuracy_plugin']) for r in rows),
        max_abs_scaled_plugin_heldout=max(r['n']*abs(r['accuracy_plugin']-r['accuracy_heldout']) for r in rows),
        information_sign_change_after_jackknife=sum(np.sign(r['delta_I_jackknife'])!=np.sign(r['delta_I_cluster_minus_spread']) for r in cs),
        max_NA=max(r['scaled_accuracy_plugin'] for r in rows),
        all_A_bounds_valid=all(0<=r['accuracy_lower_97_5']<=r['accuracy_plugin']<=r['accuracy_upper_97_5']<=4/r['n']+1e-14 for r in rows))
    encoded=json.dumps(summary,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))
    (ROOT/'summary.json').write_text(encoded+'\n');print(encoded)
if __name__=='__main__':main()
