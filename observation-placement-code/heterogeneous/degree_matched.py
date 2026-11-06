"""Post hoc diagnostic matching both endpoint degrees exactly on original graphs."""
import csv,json,subprocess,time,hashlib
import numpy as np
from run import ROOT,nx,analyze,write_csv

def load_graph(gid):
    lines=(ROOT/'graphs'/f'{gid}.txt').read_text().splitlines();n,e,k=map(int,lines[0].split())
    g=nx.Graph();g.add_nodes_from(range(n));g.add_edges_from(tuple(map(int,l.split())) for l in lines[1:e+1]);return g

def matched_pair(g,cluster):
    du,dv=sorted([g.degree[cluster['u']],g.degree[cluster['v']]])
    us=[u for u in g if g.degree[u]==du];vs={v for v in g if g.degree[v]==dv}
    best=None;count=0
    for u in us:
        eligible=[v for v in vs if v!=u and not g.has_edge(u,v) and (du!=dv or v>u)]
        if not eligible:continue
        ds=nx.single_source_shortest_path_length(g,u)
        for v in eligible:
            count+=1;pair=tuple(sorted((u,v)));key=(-ds[v],pair)
            if best is None or key<best[0]:best=(key,pair)
    return (None,0) if best is None else (best[1],count)

def main():
    dest=ROOT/'degree_matched';dest.mkdir(exist_ok=True);records=[];rows=[];contrasts=[]
    for cp in sorted((ROOT/'configs').glob('*.json')):
        old=json.loads(cp.read_text());gid=old['graph_id'];g=load_graph(gid);cluster=old['layouts'][0];pair,count=matched_pair(g,cluster)
        record=dict(graph_id=gid,family=old['family'],requested_n=old['requested_n'],n=len(g),matching_nonadjacent_pairs=count,matched=pair is not None)
        if pair is None:records.append(record);continue
        u,v=pair;layout=dict(layout='spread',u=u,v=v,e_M=0,c_M=g.degree[u]+g.degree[v],
                           a_M=min(int(g.has_edge(x,u))+int(g.has_edge(x,v)) for x in g if x not in pair),
                           distance=nx.shortest_path_length(g,u,v),degree_u=g.degree[u],degree_v=g.degree[v])
        c=old.copy();c['layouts']=[cluster,layout];c['mc_seed']+=498234123;c['batches']=16;c['per_batch']=512
        c['random_baseline_pairs']=0
        c['diagnostic']='Post hoc exact endpoint degree-multiset matching; maximally distant eligible nonadjacent pair, lexicographic ties.'
        inp=dest/f'{gid}.txt';inp.write_text('\n'.join([f'{len(g)} {g.number_of_edges()} 2']+[f'{u} {v}' for u,v in sorted(g.edges())]+[f"{x['u']} {x['v']}" for x in c['layouts']])+'\n')
        out=dest/f'{gid}.bin';start=time.perf_counter()
        with (dest/f'{gid}.log').open('w') as log:subprocess.run([str(ROOT/'fpp_mc'),str(inp),str(out),'16','512',str(c['mc_seed'])],stderr=log,check=True)
        c['runtime_seconds']=time.perf_counter()-start
        c['raw_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
        c['graph_input_sha256']=hashlib.sha256(inp.read_bytes()).hexdigest()
        r,d=analyze(c,out);rows+=r;contrasts+=d
        record.update(distance=layout['distance'],degree_1=layout['degree_u'],degree_2=layout['degree_v'])
        records.append(record);(dest/f'{gid}.json').write_text(json.dumps(c,indent=2)+'\n')
        write_csv(dest/'conditions.csv',rows);write_csv(dest/'contrasts.csv',contrasts)
        (dest/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
        print(gid,'distance',layout['distance'],'pairs',count,'seconds',c['runtime_seconds'],flush=True)
    (dest/'manifest.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()
