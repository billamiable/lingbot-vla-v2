"""Summarize CPU phases and disjoint GPU timeline intervals from Kineto traces."""
import bisect
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics

ROOT=Path('/root/experiments/lingbot-profile-20261108')


def union(intervals):
    output=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if output and a<=output[-1][1]:output[-1]=(output[-1][0],max(b,output[-1][1]))
        else:output.append((a,b))
    return output


def length(intervals):
    return sum(b-a for a,b in union(intervals))/1e6


def overlap(a,b):
    a,b=union(a),union(b);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return total/1e6


def analyze_trace(path):
    with gzip.open(path,'rt') as f:trace=json.load(f)
    events=[e for e in trace['traceEvents'] if e.get('ph')=='X' and 'dur' in e]
    updates=sorted([e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('update/')],key=lambda e:e['ts'])
    assert len(updates)==3,(path,len(updates))
    phases=sorted([e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('phase/')],key=lambda e:e['ts'])
    starts=[e['ts'] for e in phases]
    launches={}
    for e in events:
        if e.get('cat') in ['cuda_runtime','cuda_driver'] and 'correlation' in e.get('args',{}):
            launches[e['args']['correlation']]=e
    kernels=[e for e in events if e.get('cat')=='kernel']
    assert kernels,('No CUDA kernels',path)
    samples=[]
    for update in updates:
        start=update['ts'];end=start+update['dur']
        gpu=[e for e in events if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset'] and start<=e['ts']<end]
        compute=[];comm=[];memory=[];counts=Counter();tops=defaultdict(lambda:[0,0.0]);phasegpu=defaultdict(float);mapped=0
        payload=defaultdict(lambda:dict(input_bytes=0,output_bytes=0,kernels_with_metadata=0))
        phase_comm=defaultdict(Counter)
        for e in gpu:
            interval=(e['ts'],min(e['ts']+e['dur'],end))
            if e.get('cat')!='kernel':memory.append(interval);continue
            name=e['name'];lower=name.lower()
            if 'nccl' in lower:
                comm.append(interval)
                kind='allgather' if 'allgather' in lower else 'reducescatter' if 'reducescatter' in lower else 'allreduce' if 'allreduce' in lower else 'alltoall' if 'all_to_all' in str(e.get('args',{}).get('Collective name','')) else 'other_nccl'
                counts[kind]+=1
                args=e.get('args',{});size={'Float':4,'BFloat16':2,'Half':2,'Double':8,'Long':8,'Int':4}.get(args.get('dtype'))
                if size and 'In msg nelems' in args:
                    payload[kind]['input_bytes']+=int(args['In msg nelems'])*size
                    payload[kind]['output_bytes']+=int(args['Out msg nelems'])*size
                    payload[kind]['kernels_with_metadata']+=1
            else:compute.append(interval)
            tops[name][0]+=1;tops[name][1]+=e['dur']/1e6
            launch=launches.get(e.get('args',{}).get('correlation'))
            if launch:
                index=bisect.bisect_right(starts,launch['ts'])-1
                if index>=0 and phases[index]['ts']<=launch['ts']<=phases[index]['ts']+phases[index]['dur']:
                    phasegpu[phases[index]['name']]+=e['dur']/1e6;mapped+=1
                    if 'nccl' in lower:phase_comm[phases[index]['name']][kind]+=1
        wall=update['dur']/1e6
        active=compute+comm+memory
        compute_s=length(compute);comm_s=length(comm);both=overlap(compute,comm)
        memory_only=length(active)-length(compute+comm)
        phasewall=defaultdict(float)
        for e in phases:
            if start<=e['ts']<end:phasewall[e['name']]+=e['dur']/1e6
        samples.append(dict(step=int(update['name'].split('/')[1]),wall_s=wall,
                            compute_union_s=compute_s,comm_union_s=comm_s,comm_compute_overlap_s=both,
                            comm_without_compute_s=comm_s-both,memory_only_s=memory_only,
                            no_gpu_activity_s=wall-length(active),kernel_counts=dict(counts),
                            collective_logical_payload=dict(payload),
                            phase_nccl_kernel_counts={k:dict(v) for k,v in phase_comm.items()},
                            all_kernel_count=sum(v[0] for v in tops.values()),
                            phase_cpu_wall_s=dict(phasewall),phase_kernel_sum_s=dict(phasegpu),
                            mapped_kernel_count=mapped,
                            top_kernels=[dict(name=n,count=v[0],sum_s=v[1]) for n,v in sorted(tops.items(),key=lambda x:-x[1][1])[:30]]))
    categories=Counter(e.get('cat') for e in events)
    collectives=[dict(name=e['name'],args=e.get('args')) for e in kernels if 'Collective name' in e.get('args',{})][:12]
    return dict(file=path.name,categories=dict(categories),kernel_args_example=kernels[0].get('args'),collective_metadata_examples=collectives,updates=samples)


def timing(path):
    ranks=[list(map(json.loads,(path/f'rank{i}.jsonl').read_text().splitlines())) for i in range(8)]
    n=min(map(len,ranks));times=[max(rank[i]['elapsed_s'] for rank in ranks) for i in range(n)]
    pre=times[10:29];post=times[36:60]
    finite=all(math.isfinite(row[k]) for rank in ranks for row in rank for k in ['loss','grad_norm','elapsed_s'])
    return dict(records=list(map(len,ranks)),finite=finite,
                pre_11_29_s=statistics.mean(pre) if pre else None,
                post_37_60_s=statistics.mean(post) if post else None,
                unprofiled_mean_s=statistics.mean(pre+post) if pre+post else None,
                profiled_31_33_s=statistics.mean(times[30:33]) if len(times)>=33 and not path.name.startswith('validate_') else None)


def main():
    output=[]
    for path in sorted((ROOT/'runs').glob('*')):
        if not (path/'rank0.jsonl').exists():continue
        summary=dict(run=path.name,timing=timing(path),ranks=[])
        for trace in sorted(path.glob('trace_rank*.json.gz')):
            destination=trace.with_name(trace.name.replace('.json.gz','.analysis.json'))
            if destination.exists():result=json.loads(destination.read_text())
            else:
                result=analyze_trace(trace)
                destination.write_text(json.dumps(result,indent=2))
            summary['ranks'].append(result)
        output.append(summary)
        print(summary['run'],'records',summary['timing']['records'],'traces',len(summary['ranks']),flush=True)
    (ROOT/'analysis.json').write_text(json.dumps(output,indent=2))


if __name__=='__main__':main()
