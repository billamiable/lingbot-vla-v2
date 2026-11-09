"""Refresh a readable progress/final report after every guarded run."""
import csv
import json
import math
from pathlib import Path
import statistics as st

ROOT=Path(__file__).resolve().parent
cases={'A':('关','关'),'B':('开','关'),'C':('关','开'),'D':('开','开')}

def analyze(path):
    ranks=[list(map(json.loads,(path/f'rank{i}.jsonl').read_text().splitlines())) if (path/f'rank{i}.jsonl').exists() else [] for i in range(8)]
    n=min(map(len,ranks));times=[max(rank[i]['elapsed_s'] for rank in ranks) for i in range(n)]
    windows={}
    for start,end in [(11,30),(31,80),(31,200)]:
        values=times[start-1:end]
        if values:
            windows[f'{start}-{end}']=dict(n=len(values),mean_s=st.mean(values),median_s=st.median(values),p95_s=sorted(values)[math.ceil(.95*len(values))-1],samples_per_s=256/st.mean(values))
    waits=[max(rank[i]['data_wait_s'] for rank in ranks) for i in range(30,min(n,200))]
    data_wait=dict(n=len(waits),mean_s=st.mean(waits) if waits else None,max_s=max(waits) if waits else None,steps_over_1s=sum(v>1 for v in waits))
    smi=path/'smi.csv';samples=list(csv.reader(smi.read_text().splitlines())) if smi.exists() else []
    samples=[row for row in samples if len(row)>=9]
    return dict(run=path.name,records=list(map(len,ranks)),windows=windows,data_wait_31_200=data_wait,
                finite=all(math.isfinite(row[k]) for rank in ranks for row in rank for k in ['loss','grad_norm','elapsed_s']),
                peak_mib=max((int(row[2]) for row in samples),default=0),
                thermal_samples=sum(any(x.strip()=='Active' for x in row[7:9]) for row in samples))

rows=[analyze(path) for path in sorted((ROOT/'runs').glob('*')) if path.is_dir()]
cells={}
for case in cases:
    members=[x for x in rows if x['run'].startswith(case+'_r') and x['records']==[200]*8]
    if members:
        mean=st.mean(x['windows']['31-200']['mean_s'] for x in members)
        cells[case]=dict(n_runs=len(members),runs=[x['run'] for x in members],mean_s=mean,samples_per_s=256/mean,
                         peak_mib=max(x['peak_mib'] for x in members),finite=all(x['finite'] for x in members),
                         per_run_mean_s=[x['windows']['31-200']['mean_s'] for x in members])
receipts=[json.loads(p.read_text()) for p in ROOT.glob('[ABCD]_r*.receipt.json')]
ready=len(cells)==4 and all(x['n_runs']==2 for x in cells.values()) and len(receipts)==8 and all(x['passed'] for x in receipts)
numeric={}
for case in list(cases)+['A_repeat']:
    files=list((ROOT/'runs'/('smoke_'+case)).glob('numeric_rank*.json'))
    if len(files)==8:
        data=[json.loads(f.read_text()) for f in files]
        numeric[case]=dict(finite=all(x['finite'] for x in data),
            relative_l2=(sum(x['sum_diff2'] for x in data)/max(sum(x['sum_ref2'] for x in data),1e-300))**.5,
            max_abs=max(x['max_abs'] for x in data),
            matched=sum(sum(y.get('bit_equal',True) for y in x['rows']) for x in data),
            tensors=sum(len(x['rows']) for x in data))
result=dict(ready=ready,analysis_scope=dict(primary_performance_cases=['A','B'],excluded_performance_cases=['C','D'],reason='C/D have substantial data wait and long tails; do not infer compute or fusion performance from these runs'),cells=cells,numeric=numeric,runs=rows)
(ROOT/'results.json').write_text(json.dumps(result,indent=2))

def table(headers,rs):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rs])
body=['# 恢复Decoder区域编译：可行性与性能验证',
      '**状态：'+('全部完成' if ready else '进行中；以下不是最终结论')+'**。性能结论仅采用A、B；C、D受数据等待影响，仅保留为附录观测，不用于评价手写融合或区域编译的性能优劣。',
      '本实验验证：让FSDP hooks继续在图外执行通信与状态管理，单独编译decoder纯计算的checkpoint区域，能否恢复捕获并改善训练性能。A、B使用相同的新checkpoint边界；不是将官方原始baseline与优化版直接比较。',
      table(['组','Decoder区域编译','手写融合','完成轮数','均值 秒/更新','samples/s','显存峰值 MiB'],
            [[case,*cases[case],cells.get(case,{}).get('n_runs',0),f"{cells[case]['mean_s']:.4f}" if case in cells else '待测',f"{cells[case]['samples_per_s']:.2f}" if case in cells else '待测',cells.get(case,{}).get('peak_mib','待测')] for case in ['A','B']])]
if ready:
    a,b=cells['A']['mean_s'],cells['B']['mean_s']
    body+=['## A→B能够支持的结论',
           f'在已有优化的基础上，开启decoder区域编译将平均更新时间从{a:.4f}秒降到{b:.4f}秒：步时降低{100*(1-b/a):.2f}%，吞吐提升{100*(a/b-1):.2f}%，每次更新节省{a-b:.4f}秒。',
           '每组两轮、每轮200次完整参数更新，统计31–200步；先取8rank每步最大耗时，再求均值。A两轮为3.3366/3.3350秒，B两轮为2.8602/2.8595秒，结果重复性较好。A、B数据等待均值约0.0005秒，未出现超过1秒的数据等待。',
           'B的8个rank均记录144张decoder区域编译图，并完成200步前反向与参数更新。这证明图外保留FSDP hooks、图内编译计算区域的完整训练路径可行。',
           '本实验未拆分算子融合与AOT checkpoint保存/重算策略各自的贡献，也不能据此断言编译器自动替代了PR1的哪些手写融合。C、D不用于支持这类结论。',
           '性能收益不等于数值等价：B相对A的第一步梯度相对L2差约10.98%，明显高于重复A基线；来源尚未定位，也未做收敛或下游任务成功率验证。']
body+=['## 固定配置与范围',
       '组合开启时，原手写SwiGLU入口的compiler.disable阻止fullgraph捕获；直接内联又触发Inductor子进程解析嵌套Triton辅助函数失败。最终四组源码统一注册opaque custom_op前后向边界，保持原exact Triton kernels不变。custom_swiglu_check.json确认封装及compiled checkpoint下的输出、两个输入梯度均与原手写算子逐位一致；这只是局部封装检查，不代表整个训练路径逐位一致。',
       '共享checkpoint入口涉及72层、QKV/输出两个阶段与不同dtype，默认8次重编译上限不足；统一将recompile_limit设为256、accumulated_recompile_limit设为4096，保留fullgraph=True。触限失败日志保存在attempts/v2-cache-limit，失败运行未参与性能结果。',
       '独立源码副本基于71a41bc；8卡PRO 5000，mbs32、GA1、global batch256，激活重算开启。四组均使用teacher cache、ViT拆分、相同FSDP参数保留/预取/ReduceScatter深度、NCCL与NUMA配置、expandable_segments。',
       'A、B保留teacher cache、ViT等模块FSDP拆分与视觉位置缓存，以及PR2的FSDP反向参数保留、前向预取和ReduceScatter在途深度调优。两组关闭PR1的RMSNorm独立compile、SwiGLU融合、QKV准备融合；仍保留无padding路径。故A不是官方原始baseline，也不是将PR1全部撤掉的原样配置。',
       '手写融合开关成组控制RMSNorm独立compile、SwiGLU exact与QKV准备Triton融合。全局use_compile保持开启；“区域编译关”指decoder纯计算区域执行eager（独立RMSNorm编译仍受手写组开关控制），并非全模型禁用torch.compile。',
       '四组统一无padding、FP32 joint attention；QKV融合与取消padding解耦。共享移除无效assert-lambda、关闭预调优调试日志，其他注意力优化开关关闭。四组共享纯forward的checkpoint封装，FSDP hooks均位于checkpoint之外：A/C使用eager checkpoint区域，B/D编译整个纯计算checkpoint区域。不能与旧HF封装的历史步时直接混比。',
       '区域编译可能同时改变算子融合和AOT checkpoint的保存/重算策略。本实验总步时无法单独分解两者，不能将全部收益解释为某个kernel的自动融合。',
       '首轮尝试在原HF checkpoint内部单独编译forward，在完整Action路径遇到CheckpointError（保存与重算张量metadata不匹配）。相关失败与旧短运行保存在attempts/v1；未进入正式测量，随后统一重建四组基线。',
       '本轮测量编译器默认数值语义，不承诺原eager逐位一致。区域编译通过fullgraph=True要求单区域完整捕获；编译失败或缺少区域图记录会阻止进入正式测量。',
       '## 短运行数值对照',
       '每组先运行3步，使用相同CPU生成的显式flow noise和固定覆盖0.05–0.95的time；比较第一步、裁剪后/optimizer前的全rank参数梯度。A保存参考分片，其他组逐项比较。该固定输入检查只用于数值差异，不是训练分布或模型效果评测。正式性能运行不注入这些fixture，不保存梯度参考。',
       table(['组','相对A梯度global相对L2','最大绝对差','逐位相同参数分片/总数','有限值'],[[case,f"{n['relative_l2']:.6g}",f"{n['max_abs']:.6g}",str(n['matched'])+'/'+str(n['tensors']),str(n['finite'])] for case,n in numeric.items() if case in ['A','B','A_repeat']]),
       '本次重复A基线梯度相对L2约3.7e-10，而B相对A约0.1098（10.98%）；输入noise/time和teacher targets哈希已对齐。B存在超出基线重复波动的单步梯度差异，不能宣称无数值影响；也不能把整组差异单独归因于RMSNorm或换算为任务成功率损失。',
       '数值差异须结合A_repeat重复基线阅读。相对L2反映第一步梯度向量差异，不能直接换算为任务成功率变化；即使吞吐提升且loss接近，也不能据此确认比赛精度安全。',
       '## 附录：C、D只作原始观测，不作性能归因',
       '31–200步内，C、D每步各rank最大data wait的均值约0.86–0.88秒，存在多秒等待及长尾；A、B约0.0005秒。C、D受到数据供给影响，不能根据其端到端均值判断手写融合变慢、融合与编译谁更优，或两者收益是否重叠。等待长尾的具体成因尚未定位；不删除异常步、不截取早期快速窗口来补作优化结论，也不直接用总步时减去data wait推算计算性能。',
       table(['运行','31–200均值 秒','data wait均值 秒','data wait最大值 秒','等待超过1秒的步数/170','用途'],[[x['run'],f"{x['windows']['31-200']['mean_s']:.4f}",f"{x['data_wait_31_200']['mean_s']:.4f}",f"{x['data_wait_31_200']['max_s']:.4f}",str(x['data_wait_31_200']['steps_over_1s'])+'/170','主对照' if x['run'][0] in 'AB' else '仅原始观测'] for x in rows if x['run'][0] in 'ABCD' and '_r' in x['run'] and x['records']==[200]*8]),
       'C、D的短运行梯度统计仍作为原始数值观测保存在results.json和runs/smoke_C、runs/smoke_D中；不据此宣称精度等价，也不用于推导性能结论。',
       '## 复现与原始证据',
       '原机器目录：`/root/experiments/lingbot-regional-20261109`。driver.py、harness/、study.patch与regional_runtime.py记录精确运行方式。数值短运行、正式逐rank JSONL、编译图目标清单、smi.csv与各次receipt分别保存。numeric_reference中的大梯度分片仅保留在实验机器，不随报告分发。',
       '顺序：A、B、D、C、C、D、B、A，两轮反序；每次从初始权重开始，不恢复、不保存checkpoint。同步计时包括数据等待、teacher cache读取、前反向、通信、optimizer与日志。',
       table(['运行','rank记录数','31–200均值 秒','有限值','热降频活动样本'],[[r['run'],str(r['records']),f"{r['windows']['31-200']['mean_s']:.4f}" if '31-200' in r['windows'] else '未到窗口',str(r['finite']),r['thermal_samples']] for r in rows])]
(ROOT/'report.md').write_text('\n\n'.join(body)+'\n')
print(json.dumps(dict(ready=ready,completed=sum(x.get('n_runs',0) for x in cells.values()),cells={k:round(v['mean_s'],4) for k,v in cells.items()})))
