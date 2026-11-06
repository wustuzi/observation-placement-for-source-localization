#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <utility>
#include <vector>
// Exact-in-law SI simulation using independent exponential undirected edge lengths.
// Each metric realization couples all source-conditional channels; it does not
// approximate the uniform source prior by a subset of candidate sources.
int main(int argc,char**argv){
 if(argc!=6){std::cerr<<"graph.txt output.bin batches per_batch seed\n";return 2;}
 int B=std::stoi(argv[3]),R=std::stoi(argv[4]);uint64_t seed=std::stoull(argv[5]);
 std::ifstream f(argv[1]);int n,e,k;f>>n>>e>>k;
 std::vector<std::vector<std::pair<int,int>>> adj(n);
 for(int j=0;j<e;j++){int u,v;f>>u>>v;adj[u].push_back({v,j});adj[v].push_back({u,j});}
 std::vector<std::pair<int,int>> pairs(k);std::vector<int> obs;
 for(auto &p:pairs){f>>p.first>>p.second;obs.push_back(p.first);obs.push_back(p.second);}
 std::sort(obs.begin(),obs.end());obs.erase(std::unique(obs.begin(),obs.end()),obs.end());
 for(auto &p:pairs){p.first=std::lower_bound(obs.begin(),obs.end(),p.first)-obs.begin();p.second=std::lower_bound(obs.begin(),obs.end(),p.second)-obs.begin();}
 const std::vector<double> times={.1,.5,1.,2.,4.};int nt=times.size();
 std::mt19937_64 rng(seed);std::exponential_distribution<double> exp(1.);
 std::uniform_int_distribution<int> source(0,n-1);
 std::vector<double> weights(e);
 auto search=[&](int start){
  std::vector<double>d(n,std::numeric_limits<double>::infinity());d[start]=0.;
  using Item=std::pair<double,int>;std::priority_queue<Item,std::vector<Item>,std::greater<Item>>pq;pq.push({0.,start});
  while(!pq.empty()){auto[du,u]=pq.top();pq.pop();if(du!=d[u])continue;
   for(auto[v,j]:adj[u]){double dv=du+weights[j];if(dv<=times.back()&&dv<d[v]){d[v]=dv;pq.push({dv,v});}}
  }return d;
 };
 std::ofstream out(argv[2],std::ios::binary);
 int32_t head[6]={n,k,nt,B,R,(int)obs.size()};out.write((char*)head,sizeof(head));
 for(int b=0;b<B;b++){
  std::vector<uint32_t>counts((size_t)k*nt*n*4,0);
  std::vector<double>sums(nt,0.),sumsq(nt,0.);
  for(int r=0;r<R;r++){
   for(double&w:weights)w=exp(rng);
   std::vector<std::vector<double>> dist;for(int v:obs)dist.push_back(search(v));
   for(int h=0;h<k;h++)for(int t=0;t<nt;t++)for(int u=0;u<n;u++){
    int state=(dist[pairs[h].first][u]<=times[t]) + 2*(dist[pairs[h].second][u]<=times[t]);
    counts[(((size_t)h*nt+t)*n+u)*4+state]++;
   }
   // Infection-size diagnostic: a fresh uniform source, independent of layout.
   auto ds=search(source(rng));
   for(int t=0;t<nt;t++){int size=0;for(double x:ds)size+=(x<=times[t]);sums[t]+=size;sumsq[t]+=(double)size*size;}
  }
  out.write((char*)counts.data(),counts.size()*sizeof(uint32_t));
  out.write((char*)sums.data(),nt*sizeof(double));out.write((char*)sumsq.data(),nt*sizeof(double));
  std::cerr<<"batch "<<b+1<<"/"<<B<<"\n";
 }
}
