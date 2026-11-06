"""Independent 80-digit Decimal check of cycle information crossings.

Poisson tails are evaluated by recurrence, not scipy CDF/SF. The finite
recurrence K=80 has a recorded geometric upper bound on omitted mass.
"""
from decimal import Decimal as D, localcontext
from pathlib import Path
import csv,json
from run_cycles import masks,front_channel,metrics
ROOT=Path(__file__).resolve().parent

def decimal_metrics(n,obs,tau):
    K=80;m=len(obs)
    mass=[(-tau).exp()]
    for k in range(1,K+1):mass.append(mass[-1]*tau/k)
    next_mass=mass[-1]*tau/(K+1)
    tail_bound=next_mass/(1-tau/(K+2))
    tails=[D(0)]*(K+1)
    for k in range(K-1,-1,-1):tails[k]=tails[k+1]+mass[k+1]
    def sf(k):
        if k<0:return D(1)
        if k>=K:return D(0)
        return tails[k]
    def part(ds):
        boundaries=sorted(ds)
        lo=[0]+boundaries;hi=boundaries+[10**9]
        return [(sf(a-1)-sf(b-1),sum(1<<j for j,d in enumerate(ds) if d<=a)) for a,b in zip(lo,hi)]
    rows=[]
    for u in range(n):
        left=part([(u-v)%n for v in obs]);right=part([(v-u)%n for v in obs])
        p=[D(0)]*(1<<m)
        for pl,cl in left:
            for pr,cr in right:p[cl|cr]+=pl*pr
        rows.append(p)
    py=[sum(row[y] for row in rows)/n for y in range(1<<m)]
    info=sum(p*(p/py[y]).ln() for row in rows for y,p in enumerate(row) if p)/n
    a=sum(max(row[y] for row in rows) for y in range(1<<m))/n
    return info,a,tail_bound

def main():
    roots=list(csv.DictReader((ROOT/'crossovers.csv').open()))
    records=[]
    with localcontext() as ctx:
        ctx.prec=80
        for row in roots:
            n=int(row['n']);m=int(row['m'])
            if n not in [100,2000]:continue
            center=D(row['tau_c'])
            for scale in [D('.999'),D('1'),D('1.001')]:
                tau=center*scale
                ic,ac,tail=decimal_metrics(n,masks(n)[f'cluster{m}'],tau)
                isp,asp,_=decimal_metrics(n,masks(n)[f'spread{m}'],tau)
                dc=ic-isp;da=asp-ac
                f_ic,f_ac=metrics(front_channel(n,masks(n)[f'cluster{m}'],float(tau)))
                f_is,f_as=metrics(front_channel(n,masks(n)[f'spread{m}'],float(tau)))
                error=abs(D(str(f_ic-f_is))-dc)
                assert error<D('2e-14')
                assert scale==1 or (dc>0 if scale<1 else dc<0)
                records.append(dict(n=n,m=m,tau=str(tau),root_multiplier=str(scale),
                    delta_MI_decimal=str(dc),delta_A1_decimal=str(da),
                    float_delta_MI_error=str(error),per_front_omitted_mass_bound=str(tail)))
    (ROOT/'high_precision_check.json').write_text(json.dumps({'status':'passed','decimal_precision':80,
        'recurrence_cutoff':80,'checks':records},indent=2)+'\n')
    print('80-digit independent checks passed:',len(records))

if __name__=='__main__':main()
