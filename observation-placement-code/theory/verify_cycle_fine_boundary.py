"""90-digit independent Poisson-front check near the C12 finite-time crossover."""
from decimal import Decimal as D
from itertools import combinations
import json
import math

import networkx as nx
import numpy as np

from verify_spectrum import OUT, channel, decimal_project, decimal_s, geometry
from make_spectrum_figure import local_projection, values


def decimal_cycle_channel(n,tau):
    p=[[D(0)]*(1<<n) for _ in range(n)]
    prefactor=(-2*tau).exp()
    for source in range(n):
        for left in range(n-1):
            for right in range(n-1-left):
                infected={((source-j)%n) for j in range(left+1)} | {((source+j)%n) for j in range(right+1)}
                state=sum(1<<v for v in infected)
                probability=prefactor*tau**(left+right)/D(math.factorial(left)*math.factorial(right))
                p[source][state]+=probability
        p[source][-1]=1-sum(p[source])
    return p


def main():
    n,m=12,3
    tau=D('.0001')
    summary=json.loads((OUT/'figure_crossover_summary.json').read_text())
    alpha=D(str([row for row in summary if row['n']==12][0]['exact_extremal_class_crossing']))
    masks=list(combinations(range(n),m))
    g=nx.cycle_graph(n)
    e=np.array([geometry(g,mask)[0] for mask in masks])
    full=channel(g,float(tau))
    s=values(local_projection(full,masks),float(alpha))
    chosen={int(k):masks[np.flatnonzero(e==k)[np.argmax(s[e==k])]] for k in np.unique(e)}
    independent=decimal_cycle_channel(n,tau)
    crosscheck=float(abs(np.array(independent,dtype=float)-full).max())
    assert crosscheck<1e-14
    scores={k:decimal_s(decimal_project(independent,mask),alpha) for k,mask in chosen.items()}
    advantage=scores[1]-max(scores[0],scores[2])
    assert advantage>D('1e-12')
    out=dict(n=n,m=m,tau=str(tau),alpha=str(alpha),
        representative_masks={str(k):list(v) for k,v in chosen.items()},
        exact_poisson_front_S={str(k):str(v) for k,v in scores.items()},
        intermediate_edge_class_advantage=str(advantage),
        maximum_channel_difference=crosscheck,
        interpretation='Finite-time second-order corrections produce a narrow intermediate geometry; fixed-alpha asymptotic claims are unaffected.')
    (OUT/'fine_boundary_high_precision.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))


if __name__=='__main__':
    main()
