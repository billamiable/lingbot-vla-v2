"""Build the completed profiling attribution report from archived measurements."""
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'evidence'
rows=json.loads((ROOT/'analysis.json').read_text())
cells={r['run']:r for r in rows if not r['run'].startswith('validate_')}
assert len(cells)==6
metrics={}
for name,r in cells.items():
    assert r['timing']['records']==[60]*8 and r['timing']['finite']
    assert len(r['ranks'])==8
    updates=[u for rank in r['ranks'] for u in rank['updates']]
    assert len(updates)==24
    keys=['wall_s','compute_union_s','comm_without_compute_s','memory_only_s','no_gpu_activity_s','all_kernel_count']
    m={k:st.mean(x[k] for x in updates) for k in keys}
    m['counts']={k:sorted(set(x['kernel_counts'].get(k,0) for x in updates)) for k in ['allgather','reducescatter','allreduce','alltoall']}
    assert all(len(v)==1 for v in m['counts'].values())
    m['counts']={k:v[0] for k,v in m['counts'].items()}
    m['payload']={};m['payload_complete_observations']={}
    for k in ['allgather','reducescatter','alltoall']:
        complete=[x for x in updates if x['collective_logical_payload'][k]['kernels_with_metadata']==x['kernel_counts'][k]]
        assert len(complete)>=22
        m['payload_complete_observations'][k]=len(complete)
        m['payload'][k]={field:st.median(x['collective_logical_payload'][k][field] for x in complete) for field in ['input_bytes','output_bytes']}
    assert abs(m['wall_s']-sum(m[k] for k in ['compute_union_s','comm_without_compute_s','memory_only_s','no_gpu_activity_s']))<1e-6
    metrics[name]=m

validations={}
benchmark=json.loads((ROOT/'benchmark_aggregate.json').read_text())
for code in ['main','opt']:
    p=ROOT/'runs'/f'validate_{code}_ac16'
    ranks=[list(map(json.loads,(p/f'rank{i}.jsonl').read_text().splitlines())) for i in range(8)]
    assert list(map(len,ranks))==[80]*8
    times=[max(r[i]['elapsed_s'] for r in ranks) for i in range(30,80)]
    validations[code]=dict(mean_31_80_s=st.mean(times),median_31_80_s=st.median(times),records=[80]*8)

comparisons={}
for code in ['main','opt']:
    b=metrics[code+'_noac16'];c=metrics[code+'_ac16'];a=metrics[code+'_ac32']
    comparisons[code]=dict(endpoint={k:a[k]-b[k] for k in ['wall_s','compute_union_s','comm_without_compute_s','memory_only_s','no_gpu_activity_s']},
                           checkpoint_fixed_mb={k:c[k]-b[k] for k in ['wall_s','compute_union_s','comm_without_compute_s']},
                           batch_fixed_ac={k:a[k]-c[k] for k in ['wall_s','compute_union_s','comm_without_compute_s']})
(ROOT/'attribution.json').write_text(json.dumps(dict(metrics=metrics,comparisons=comparisons,unprofiled_controls=validations),indent=2))

fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
components=[('compute_union_s','Non-NCCL kernels','#376fa7'),('comm_without_compute_s','NCCL without compute overlap','#d27a2b'),('memory_only_s','Memory only','#70b7a1'),('no_gpu_activity_s','No recorded GPU activity','#a7afb8')]
for ax,code,title in zip(axes,['main','opt'],['Official main','Optimized']):
    names=[code+'_'+x for x in ['noac16','ac16','ac32']]
    base=[0.0]*3
    for key,label,color in components:
        vals=[metrics[n][key] for n in names]
        ax.bar(range(3),vals,bottom=base,label=label,color=color,width=.58)
        base=[x+y for x,y in zip(base,vals)]
    for i,v in enumerate(base):ax.text(i,v+.05,f'{v:.3f}',ha='center')
    ax.set_xticks(range(3),['AC off\n16 / GA2','AC on\n16 / GA2','AC on\n32 / GA1'])
    ax.set_title(title);ax.set_ylabel('Seconds per update\n(8 ranks x 3 steps mean)')
    ax.set_ylim(0,max(m['wall_s'] for m in metrics.values())*1.15);ax.spines[['top','right']].set_visible(False)
fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,-.12),ncol=2,frameon=False)
fig.savefig(ROOT/'attribution.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'attribution.svg',bbox_inches='tight');plt.close(fig)

def table(headers,data):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in data])

order=[code+'_'+recipe for code in ['main','opt'] for recipe in ['noac16','ac16','ac32']]
labels={'noac16':'不重算16 GA2','ac16':'重算16 GA2','ac32':'重算32 GA1'}
def label(n):
    code,recipe=n.split('_',1);return ('原版' if code=='main' else '优化版')+' '+labels[recipe]

parts=['# 激活重算为什么能带来更高吞吐：Profiling归因',
       '## 结论',
       '这次收益来自重算允许使用更大的microbatch，从而把GA2改为GA1，减少每次参数更新的FSDP通信轮次和暴露出来的通信/同步时间。重算并没有减少计算成本。固定mbs16、GA2时，开启重算的控制组反而更慢。',
       '优化版的直接证据：AllGather从222次降至111次，ReduceScatter从366次降至183次；DistMuon优化器的106次All-to-All保持不变。原版的对应计数与耗时见下表。',
       '## 时间线分解',
       '每组8个rank、每rank第31–33次更新，共24个观测的均值。以下是采集窗口的解释性数据，不替代[此前无Profiler的200步吞吐结果](../report.md)。',
       '![计算与暴露通信的时间分解](attribution.png)',
       table(['配置','窗口耗时 秒','非NCCL计算活动 秒','未被计算覆盖的NCCL 秒','仅内存操作 秒','无GPU活动 秒'],
             [[label(n)]+[f'{metrics[n][k]:.4f}' for k in ['wall_s','compute_union_s','comm_without_compute_s','memory_only_s','no_gpu_activity_s']] for n in order]),
       '先对各类GPU事件取时间区间并集，再计算重叠。四个互斥区间之和等于CPU更新标记的持续时间。这里的“计算活动”表示非NCCL kernel执行区间，不代表硬件算术单元利用率；“未被计算覆盖的NCCL”包含传输、同步及等待其他rank，不能全部解释成纯网络传输时间。无GPU活动区间也不能全部归因于CPU调度。',
       '## 通信次数与逻辑消息量',
       table(['配置','AllGather','ReduceScatter','AllReduce','All-to-All','AllGather输出 GiB/rank','ReduceScatter输入 GiB/rank'],
             [[label(n)]+[metrics[n]['counts'][k] for k in ['allgather','reducescatter','allreduce','alltoall']]+[f"{metrics[n]['payload']['allgather']['output_bytes']/2**30:.3f}",f"{metrics[n]['payload']['reducescatter']['input_bytes']/2**30:.3f}"] for n in order]),
       '次数为trace中NCCL kernel事件数，24个观测逐项一致。逻辑消息量来自Kineto的Collective name、In/Out msg nelems和dtype元数据；它不是实际链路传输字节数。优化版重算32有2个观测各缺1个AllGather消息大小，优化版不重算16有2个观测各缺1个ReduceScatter消息大小。消息量取对应元数据完整观测的中位数（至少22/24）；不补造缺失值。次数和时间仍使用全部24个观测。SendRecv kernel按元数据识别为all_to_allv，不能混算为FSDP收集或归约。',
       '## 控制组把因果链拆开']
for code in ['main','opt']:
    c=comparisons[code]
    parts.append(table([('原版' if code=='main' else '优化版')+' 配置变化','计算活动变化 秒','暴露NCCL变化 秒','更新耗时变化 秒'],
                       [[name]+[f"{c[key][k]:+.4f}" for k in ['compute_union_s','comm_without_compute_s','wall_s']] for name,key in [('固定16 GA2，关闭→开启重算','checkpoint_fixed_mb'),('固定开启重算，16 GA2→32 GA1','batch_fixed_ac'),('最终端点，不重算16→重算32','endpoint')]]))
parts += ['负数表示耗时减少。固定16 GA2的控制组保持了通信轮次，显示重算额外计算的净成本；固定开启重算的比较同时改变microbatch与累积轮次，显示完整执行配置的收益，不能把计算活动的缩短全部单独归因于GEMM效率。',
          '## 不开启Profiler的独立复核',
          '新增的重算16控制组另用原始计时harness运行80次更新，完全不开启Profiler。每一步取8个rank最大时间，统计第31–80次；原始日志和配置一并保存。',
          table(['代码','原不重算16 GA2均值 秒','新增重算16 GA2均值 秒','原重算32 GA1均值 秒'],[[code,f"{benchmark['cells'][code+'_noac16']['windows']['31-80']['mean_s']:.4f}",f"{validations[code]['mean_31_80_s']:.4f}",f"{benchmark['cells'][code+'_ac32']['windows']['31-80']['mean_s']:.4f}"] for code in ['main','opt']]),
          '本表统一使用31–80均值；原端点来自此前两轮的平均，新增控制组为本次单次80步运行。这些均为无Profiler测量，未与采集窗口耗时混合。',
          table(['采集运行','采集前11–29均值 秒','采集后37–60均值 秒','采集31–33最大rank均值 秒'],[[label(n)]+[f"{cells[n]['timing'][k]:.4f}" for k in ['pre_11_29_s','post_37_60_s','profiled_31_33_s']] for n in order]),
          '导出trace会扰动相邻步骤，因此采集后的比较排除第34–36次。保留所有慢步，不为获得好看的均值删除异常记录。采集有额外开销，最终吞吐引用无Profiler结果；阶段分解用于解释执行路径。',
          '## 采集与复核方法',
          '同一台8卡RTX PRO 5000，main be969b8、优化版71a41bc；沿用上一轮的模型、数据、teacher模式、global batch 256、seed、学习率、NCCL、NUMA及expandable_segments。六组各60次更新，第30次Profiler预热，第31–33次记录CPU+CUDA，全部8个rank保存trace。record_shapes、with_stack、profile_memory均关闭；没有新增每个阶段的CUDA同步。',
          '通过AST包装既有计时入口，增加microbatch、prepare/teacher/prologue、forward、backward、after_backward/reshard、optimizer标记，不改训练计算。Kineto同时输出CPU和GPU注释，统计只用CPU user_annotation作为更新边界，避免重复计数。',
          '原始执行逻辑与trace一致：每个microbatch都backward；优化版每次backward后调用after_backward，排空在途ReduceScatter并重新分片。第二个microbatch再次收集参数。PR2改善了microbatch内部调度，并未让两份microbatch共享一次梯度归约或跨microbatch保留完整参数。',
          '完整原始trace保存在原机器 `/root/experiments/lingbot-profile-20261108` 和本地归档。`analysis.json`保存每rank每步事件统计，`attribution.json`保存表格与差值，`analyze.py`是原始trace分析入口。',
          '## 结论边界',
          '证据支持本硬件、本代码与这两套配置下，减少暴露的FSDP通信/同步是净提速的主要来源。时间线分解是观测到的执行区间，并非对每个机制独立干预后的唯一因果百分比。没有测量实际链路带宽、Tensor Core利用率或最终模型效果，也不能推广为“单独打开重算就会加速”。']
(ROOT/'profiling.md').write_text('\n\n'.join(parts)+'\n')
print(ROOT/'profiling.md')
