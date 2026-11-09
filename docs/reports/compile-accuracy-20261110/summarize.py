"""Qihan-style run-level window-mean permutation tests; finite/bitwise checks are separate evidence."""
import itertools,json,math,statistics as st
from pathlib import Path
R=Path(__file__).resolve().parent
runs={}
for p in (R/'runs').glob('*'):
 f=p/'rank0.jsonl'
 if not f.exists():continue
 try:rows=[json.loads(s) for s in f.read_text().splitlines()]
 except json.JSONDecodeError:continue
 runs[p.name]=dict(steps=len(rows),first_loss=rows[0]['loss'] if rows else None,losses=[x['loss'] for x in rows],max_grad_norm=max((x['grad_norm'] for x in rows),default=None),finite=all(math.isfinite(x[k]) for x in rows for k in ['loss','vla_loss','grad_norm']))
num={}
for name in ['num_A','num_A_repeat','num_B','num_B_cast','num_PR1']:
 cell={}
 for kind in ['grad','parameter']:
  files=list((R/'runs'/name).glob(kind+'_rank*.json'))
  if len(files)!=8:continue
  data=[json.loads(f.read_text()) for f in files];logical={}
  for item in data:
   for row in item['rows']:logical[row['name']]=logical.get(row['name'],True) and row.get('bit_equal',True)
  d=sum(x['sum_diff2'] for x in data);ref=sum(x['sum_ref2'] for x in data)
  cell[kind]=dict(relative_l2=(d/max(ref,1e-300))**.5,max_abs=max((row.get('max_abs',0) for x in data for row in x['rows']),default=0),matched=sum(row.get('bit_equal',True) for x in data for row in x['rows']),tensors=sum(x['parameters'] for x in data),logical_matched=sum(logical.values()),logical_tensors=len(logical),finite=all(x['finite'] for x in data))
 b=R/'runs'/name/'boundaries.jsonl'
 if b.exists():
  rows=[json.loads(s) for s in b.read_text().splitlines()]
  cell['first_boundary_differences']=[{k:x[k] for k in ['name','relative_l2','max_abs'] if k in x} for x in rows if x.get('bit_equal') is False][:12]
 num[name]=cell
arms={c:[n for n,v in runs.items() if n.startswith('train_'+c) and v['steps']==300] for c in ['A','B']}
stats=[]
for start,end in [(1,300),(151,300),(251,300)]:
 a=[st.mean(runs[n]['losses'][start-1:end]) for n in arms['A']];b=[st.mean(runs[n]['losses'][start-1:end]) for n in arms['B']]
 if not a or not b:continue
 vals=a+b;observed=abs(st.mean(b)-st.mean(a));hits=total=0
 for subset in itertools.combinations(range(len(vals)),len(a)):
  chosen=set(subset);x=[v for i,v in enumerate(vals) if i in chosen];y=[v for i,v in enumerate(vals) if i not in chosen]
  hits+=abs(st.mean(y)-st.mean(x))>=observed-1e-15;total+=1
 stats.append(dict(window=f'{start}-{end}',a_runs=len(a),b_runs=len(b),a_means=a,b_means=b,mean_A=st.mean(a),mean_B=st.mean(b),relative_percent=100*(st.mean(b)/st.mean(a)-1),p_value=hits/total,permutations=total))
ready=len(arms['A'])==len(arms['B'])==6
profiles={}
for p in (R/'runs').glob('profile_*'):
 f=p/'operator_profile.json'
 if f.exists():
  rows=json.loads(f.read_text());profiles[p.name]=sorted(rows,key=lambda x:x['self_gpu_us'],reverse=True)[:40]
results=dict(complete=ready,method='Run-level mean, unpaired exact two-sided permutation; six repeated restarts per arm, identical initial weights/data/aligned original-distribution flow samples',numeric=num,loss_statistics=stats,profiles=profiles,runs={k:{a:b for a,b in v.items() if a!='losses'} for k,v in runs.items()})
(R/'results.json').write_text(json.dumps(results,indent=2))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
body=['# Decoder区域编译：数值影响与编译收益来源',
 '**状态：'+('300步loss对照完成' if ready else '进行中；不作最终精度判断')+'**。本轮只比较原A/B改法；此前C/D有数据等待，不用于性能结论。',
 '## 比较方法',
 '参考Qihan报告第4.1节：首步对比全rank梯度与optimizer更新后参数；同初始权重/数据/随机抽样，A/B各6次重启，每次300步，按运行分别计算1–300、151–300、251–300平均loss，再作双侧精确置换检验（6对6共924种分法）。p值大不证明等价，小p也不等于任务效果明显受损；没有下游成功率评测。',
 '用独立CPU随机流显式生成noise和time，保持原训练分布：标准正态noise，Beta(1.5,1)*0.999+0.001的time。不同于前轮均匀铺开的固定time fixture；也不同于原CUDA采样实现。因此绝对首步loss/梯度差不与上轮直接混比。所有运行逐步记录输入张量哈希，核对A/B同一步输入；模型其余RNG状态不被fixture改写。',
 'B_cast只在decoder区域的Inductor编译中启用emulate_precision_casts；用于检验保留中间低精度舍入的作用，不保证规约次序或全部反向语义严格一致，也不影响外层视觉/attention的编译配置。',
 '## 首步数值检查',
 table(['运行','对象','相对A的L2差','最大绝对差','逐位相同分片/总数','逐位相同逻辑参数/总数'],[[name,kind,f"{v['relative_l2']:.8g}",f"{v['max_abs']:.8g}",str(v['matched'])+'/'+str(v['tensors']),str(v['logical_matched'])+'/'+str(v['logical_tensors'])] for name,c in num.items() for kind,v in c.items() if kind in ['grad','parameter']]),
 '完整参数梯度、更新后权重的逐项统计，以及rank0选定层0/1/17/35的输入、输出及输出反向梯度差异见runs/num_*。大张量参考仅保留在/raid/experiments/lingbot-compile-accuracy-20261110/references，不随报告分发。',
 '## 300步loss比较',
 table(['窗口','完成运行 A/B','A平均loss','B平均loss','相对差','置换p值'],[[x['window'],str(x['a_runs'])+'/'+str(x['b_runs']),f"{x['mean_A']:.8f}",f"{x['mean_B']:.8f}",f"{x['relative_percent']:+.5f}%",f"{x['p_value']:.6g}"] for x in stats]),
 '## 编译收益证据',
 '单独profile A、B、PR1仅RMSNorm、仅SwiGLU、仅QKV准备，以及PR1全开。全都保持相同无padding FP32 attention与通信配置；只用第11–13步的rank0算子/设备活动与生成代码分析，不用这些短运行的端到端均值作性能排名。',
 'runs/profile_*/fx包含区域FX图；codegen包含rank0的Inductor生成代码和IR；operator_profile.json及trace.json.gz记录CPU/CUDA事件。decoder边界以decoder.vlm/expert.*标注。PR1 QKV准备的跨流cast/cat/RoPE位于decoder层调用之外，需区分外层原有compile与这次新增区域compile；不能把时间接近当成三项优化完全相同的证明。',
 '生成代码融合了哪些ATen操作、对应CUDA kernel数量/耗时、checkpoint保存/重算差别，须联合解释。profile是有开销的rank0观测；算子GPU时间与端到端墙钟不可直接相加归因。',
 '## 原始记录',
 table(['运行','完成步数','loss/grad有限值','最大梯度范数','第一步总loss'],[[n,v['steps'],v['finite'],v['max_grad_norm'],v['first_loss']] for n,v in runs.items()]),
 '源代码与运行方式见study.patch、source_freeze.json、study_runner.py和harness/。状态在status.json；发生错误自动停止后续队列，不占用他人GPU任务。']
if ready:
 significant=[x for x in stats if x['p_value']<.05]
 body+=['## 300步对照能够支持的判断']
 if significant:
  body.append('本设计下，以下窗口的平均loss存在统计上可分辨的变化（未作多窗口校正）：'+ '; '.join(f"{x['window']}：{x['relative_percent']:+.5f}%，p={x['p_value']:.6g}" for x in significant)+'。差异大小与是否显著需同时阅读，不能直接换算为任务成功率变化。')
 else:
  body.append('六次对六次的300步对照中，三个预设窗口均未分辨出p<0.05的平均loss差异。这不证明数值等价，也不证明完整训练或下游任务效果相同。')
 body.append('Qihan报告中kernel fusion的1–300、151–300、251–300窗口loss相对差分别约+0.02%、+0.12%、+0.23%，可作为量级参考，不能作为本轮预先约定的验收阈值；两次实验的配置、抽样实现与重复次数不同，不跨实验合并p值或作显著性排名。')
if num.get('num_B',{}).get('grad') and num.get('num_B_cast',{}).get('grad'):
 normal=num['num_B']['grad']['relative_l2'];casts=num['num_B_cast']['grad']['relative_l2']
 body+=['## 中间舍入对照的初步定位',
        f'默认B的首步梯度相对L2差为{normal*100:.3f}%，仅区域编译保留中间低精度舍入后为{casts*100:.3f}%；整体差异没有被消除。',
        '对齐的第一个VLM decoder输入逐位一致。其Q输出相对L2差从默认B约3.96e-3降到保留舍入后的约3.22e-5，说明中间舍入对早期前向差异有贡献；但不能解释全部模型梯度差。其余舍入、规约、编译反向或后续层放大效应仍需定位，不能宣称已证明全部差异都是正常浮点误差。']
if (R/'mechanisms.json').exists():
 evidence=json.loads((R/'mechanisms.json').read_text())
 body+=['## 已提取的kernel与生成代码观测',
        '下表为实际CUDA kernel计数和前向CPU launch作用域的对应关系，按rank0的3个profile更新平均；不以profile墙钟排名。完整名称、FX和生成代码映射见mechanisms.json。该文件另保存关键词检索分组，但关键词不能覆盖A的拆散pow/mean/mul与B的融合kernel的相同工作量，不用它比较Norm或激活的总开销。',
        table(['配置','全部CUDA kernel/更新','VLM QKV前向kernel/更新','VLM O/MLP前向kernel/更新','Expert QKV前向kernel/更新','Expert O/MLP前向kernel/更新','保存代码文件数','区域FX图数'],[[name,f"{v['kernel_count']/3:.1f}",*[f"{v['forward_launch_scopes'].get(phase,{}).get('kernel_count',0)/3:.1f}" for phase in ['vlm.qkv','vlm.omlp','expert.qkv','expert.omlp']],len(v['code']),len(v['fx'])] for name,v in evidence['cases'].items()])]
if (R/'mechanisms.json').exists():
 evidence=json.loads((R/'mechanisms.json').read_text())['cases']
 if 'A' in evidence and 'B' in evidence:
  a,b=evidence['A'],evidence['B']
  norm_graphs=sum(x['norm'] for x in b['fx']);silu_graphs=sum(x['silu'] for x in b['fx'])
  body+=['## A/B编译范围与kernel变化的解读',
         f'新增decoder区域FX记录中，{norm_graphs}张包含rsqrt规范化计算，{silu_graphs}张包含SiLU计算。它们对应RMSNorm与SwiGLU路径，说明区域编译覆盖了PR1前两项所针对的计算；图覆盖不代表舍入语义或融合kernel与手写实现完全相同。',
         table(['前向CPU launch作用域','A kernel/更新','B kernel/更新'],[[phase,f"{a['forward_launch_scopes'].get(phase,{}).get('kernel_count',0)/3:.1f}",f"{b['forward_launch_scopes'].get(phase,{}).get('kernel_count',0)/3:.1f}"] for phase in ['vlm.qkv','vlm.omlp','expert.qkv','expert.omlp']]),
         '上表按CPU launch correlation关联decoder前向作用域，不包含图外FSDP hooks，也不把无作用域的反向/重算kernel强行分配到某一层。作用域内具体kernel名称见mechanisms.json；需结合生成代码确认实际融合。',
         '区域还覆盖Q/K/V和O投影、残差连接、action expert的调制及MLP等张量计算；因此并非只打开PR1三个开关。PR1的跨流cast/cat/RoPE准备在这些层调用之外，不能因为最终步时接近就视作三项优化等效。',
         '这些证据用于解释计算覆盖与kernel变化；未把原16.64%吞吐收益精确分配到各项，也未分离AOT checkpoint保存/重算策略的贡献。']
if (R/'mechanisms.json').exists():
 cases_evidence=json.loads((R/'mechanisms.json').read_text())['cases']
 if all(x in cases_evidence for x in ['A','B','RMS','SWI','QKV','PR1']):
  def count(case,phase):return cases_evidence[case]['forward_launch_scopes'][phase]['kernel_count']/3
  body+=['## 与PR1的重叠及不同之处',
         f"VLM QKV前向：A为{count('A','vlm.qkv'):.0f}次kernel/更新，仅RMSNorm编译与区域B均为{count('B','vlm.qkv'):.0f}次。结合RMSNorm融合生成代码，这支持此阶段与PR1 RMSNorm优化存在明显重叠；调用数相同不等于kernel实现或数值语义逐位相同。",
         f"VLM O/MLP前向：PR1全开为{count('PR1','vlm.omlp'):.0f}次，区域B为{count('B','vlm.omlp'):.0f}次；B生成代码还融合了残差add等周边操作，并非只有独立SwiGLU。",
         f"Expert前向：A与PR1的QKV/O-MLP分别为{count('A','expert.qkv'):.0f}/{count('A','expert.omlp'):.0f}次，区域B降到{count('B','expert.qkv'):.0f}/{count('B','expert.omlp'):.0f}次。区域编译也覆盖action expert的AdaRMSNorm调制与MLP周边计算，这些收益范围超出PR1的VLM RMSNorm/SwiGLU单项。",
         '仅QKV准备融合不改变上述四类decoder前向作用域的kernel数，变化发生在跨流准备等作用域外；它与本轮新增decoder区域编译的边界不同。',
         '因此，区域编译确实自动覆盖了PR1前两项针对的部分计算，并进一步融合更大区域的周边/反向操作，但不能视为自动启用PR1三项同一实现。最终步时接近可以来自不同优化范围的组合，不能据此断言数学等价或收益完全重合。']
(R/'report.md').write_text('\n\n'.join(body)+'\n')
print(json.dumps(dict(complete=ready,training_counts={k:len(v) for k,v in arms.items()},numeric=list(num),profiles=list(profiles))))
