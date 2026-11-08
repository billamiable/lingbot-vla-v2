"""Export the completed study as a self-contained Chinese HTML report and standard plots."""
import datetime
import html
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/root/experiments/lingbot-ac-mbs-20261108')
data=json.loads((ROOT/'aggregate.json').read_text())
assert data['ready'], 'All eight uniform-allocator runs must finish before rendering the final report'
cells=data['cells']
out=ROOT/'delivery'
out.mkdir(exist_ok=True)
labels={'main_ac32':'main 重算32 GA1','main_noac16':'main 不重算16 GA2',
        'opt_ac32':'优化版 重算32 GA1','opt_noac16':'优化版 不重算16 GA2'}
order=list(labels)
fig,axes=plt.subplots(1,2,figsize=(10.4,4.0),layout='constrained')
for ax in axes:
    ax.spines[['top','right']].set_visible(False)
    ax.set_xticks([0,1],['Official main','Optimized'])
    ax.set_axisbelow(True)
    ax.grid(axis='y',alpha=.2)
for j,(suffix,color,title) in enumerate([('ac32','#3169a8','AC on, micro 32, GA 1'),('noac16','#d17a27','AC off, micro 16, GA 2')]):
    x=[i+(-.19 if j==0 else .19) for i in range(2)]
    keys=[code+'_'+suffix for code in ['main','opt']]
    times=[cells[k]['windows']['31-200']['mean_s'] for k in keys]
    mem=[cells[k]['peak_mib']/1024 for k in keys]
    bars=axes[0].bar(x,times,.34,color=color,label=title)
    axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    for xi,k in zip(x,keys):
        vals=cells[k]['windows']['31-200']['per_run_mean_s']
        axes[0].scatter([xi]*len(vals),vals,color='black',s=12,zorder=3)
    bars=axes[1].bar(x,mem,.34,color=color)
    axes[1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
axes[0].set_ylabel('Seconds per optimizer update (lower is better)')
axes[1].set_ylabel('Peak nvidia-smi memory, GiB (8-rank maximum)')
axes[0].set_ylim(0,max(cells[k]['windows']['31-200']['mean_s'] for k in order)*1.2)
axes[1].set_ylim(0,max(cells[k]['peak_mib']/1024 for k in order)*1.15)
fig.legend(loc='outside lower center',ncol=2,frameon=False)
fig.savefig(out/'comparison.png',dpi=180)
buf=io.StringIO();fig.savefig(buf,format='svg');svg=buf.getvalue();svg=svg[svg.index('<svg'):];fig.savefig(out/'comparison.svg');plt.close(fig)

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

run_summaries={r['run']:r for r in json.loads((ROOT/'summary_all.json').read_text())}
primary=[];compat=[];per_run=[];health=[]
for k in order:
    c=cells[k];w=c['windows']['31-200']
    primary.append([labels[k],f"{w['mean_s']:.4f}",f"{w['mean_run_median_s']:.4f}",f"{w['samples_per_s']:.2f}",f"{c['peak_mib']:,}",f"{c['peak_allocated_gib']:.2f}"])
    compat.append([labels[k],f"{c['windows']['11-30']['mean_s']:.4f}",f"{c['windows']['31-80']['mean_run_median_s']:.4f}",f"{w['mean_s']:.4f}"])
    health.append([labels[k],f"{w['loss_mean']:.5f}",f"{w['vla_loss_mean']:.5f}",f"{c['max_grad_norm']:.5f}",c['clipped_updates'],'全部有限' if c['finite'] else '异常'])
    for run,a,b in zip(c['runs'],w['per_run_mean_s'],w['per_run_median_s']):
        rw=run_summaries[run]['windows']['31-200']
        per_run.append([run,f'{a:.4f}',f'{b:.4f}',f"{rw['p95_s']:.4f}",f"{rw['mean_data_wait_s']*1000:.3f}",', '.join(map(str,w['slow_steps'][run])) or '无'])

choices=[]
for code in ['main','opt']:
    c=data['comparisons'][code];gain=c['ac32_throughput_gain_pct']
    choices.append(f"{'main' if code=='main' else '优化版'}：重算32与不重算16的完整更新均值分别为 {c['ac32_s']:.4f} 秒和 {c['noac16_s']:.4f} 秒。重算32的吞吐{'提高' if gain>=0 else '降低'} {abs(gain):.2f}%。")

repro=json.loads((ROOT/'reproduction_gate.json').read_text())
repro_rows=[[name,f"{r['target_median_s']:.3f}",f"{r['median_s']:.4f}",f"{r['deviation_pct']:+.3f}%"] for name,r in repro.items()]
audit=json.loads((ROOT/'consistency_audit.json').read_text())
pairs=[p for p in audit['input_pairs'] if p['a'].startswith('alloc_')]
assert len(pairs)==4 and all(not p['differences'] for p in pairs), 'Input audit must pass for every repeated pair'
assert all(c['finite'] for c in cells.values()), 'Nonfinite training values must be investigated'
thermal=sum(c['thermal_slowdown_samples'] for c in cells.values())
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LingBot VLA 激活重算与微批量四组对照实验报告</title>
<style>body{{font:15px/1.7 system-ui,"Noto Sans CJK SC",sans-serif;color:#202630;max-width:1040px;margin:36px auto;padding:0 24px}}h1{{font-size:27px}}h2{{font-size:21px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:18px 0}}th,td{{text-align:left;padding:8px;border-bottom:1px solid #d9dee5;vertical-align:top}}th{{background:#f2f5f8}}code{{font-size:13px;word-break:break-all}}svg{{width:100%;height:auto}}.meta{{color:#596575}}@media print{{body{{margin:0;font-size:11px}}h2{{break-after:avoid}}tr{{break-inside:avoid}}svg{{max-height:300px}}}}</style>
<h1>LingBot VLA 激活重算与微批量四组对照实验报告</h1>
<p class="meta">{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')} · 单机8卡 RTX PRO 5000 · global batch 256 · 两轮镜像重复</p>
<h2>结论</h2>{''.join('<p>'+x+'</p>' for x in choices)}
<p>以上比较使用同一分配器设置、相同global batch及完整参数更新的端到端时间。短程loss仅检查训练健康，未进行下游成功率或完整收敛评测。</p>
<h2>实验配置与方法</h2>
<p>方案A：激活重算开启、micro batch 32、梯度累积1。方案B：激活重算关闭、micro batch 16、梯度累积2。两者均为8卡，每次更新处理256个样本，使用相同初始权重、学习率1e-4与50000步学习率调度。每次实际训练200次更新，不保存checkpoint。</p>
<p>速度差是两种完整执行配方的净效果，包含重算、micro batch、梯度累积、编译和FSDP调度的交互。</p>
<p>main为be969b8；优化版为Qihan报告使用的71a41bc，开启ViT拆分与位置缓存、离线teacher cache、RMSNorm编译、QKV与SwiGLU融合、FSDP反向参数保留、预取深度4及ReduceScatter在途深度4。额外bf16 attention与fsdp_align_root关闭。main使用在线teacher；同一代码内的两个方案保持teacher模式一致。</p>
<p>四组统一设置 <code>PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True</code>。运行顺序为main B、main A、优化B、优化A、优化A、优化B、main A、main B。数据为RoboTwin clean50 PNG，8个worker，prefetch factor 2，NCCL_MIN_CTAS=8、NCCL_NTHREADS=128、NCCL_P2P_READ_ENABLE=1，每个rank按GPU绑定NUMA。</p>
<p>计时从取数据前开始，覆盖所有累积micro-batch、teacher、前向、反向、通信、optimizer和日志，取8个rank的最大耗时。主窗口为第31–200次；主均值包含所有卡顿。每组两次运行等权合并；表中的中位数是两次运行中位数的平均。吞吐=256÷主均值。</p>
<h2>Qihan报告复现</h2><p>使用原样benchmark各运行80步，先验证第31–80步的中位数。两组均完整完成，复现偏差小于1%。</p>{table(['版本','报告中位数 秒','本次中位数 秒','偏差'],repro_rows)}
<h2>四组性能与显存</h2>{svg}{table(['配置','完整更新均值 秒','中位数 秒','samples/s','nvidia-smi峰值 MiB','torch分配峰值 GiB'],primary)}
<p>黑点为每次运行的均值。nvidia-smi显存按1秒采样，torch分配峰值来自运行时累积峰值，均取两次运行与8个rank的最大值。热降频活动样本数：{thermal}。</p>
<h2>与既有测试方法统一的三套口径</h2><p>逐文件核对确认Qihan的5个核心benchmark脚本与公开bench-repro@9dc0677逐字节一致，区别在取样区间与统计量。本次增加观测字段，不改变训练计算或计时边界。</p>{table(['配置','11–30均值 秒','31–80中位数 秒','31–200均值 秒'],compat)}
<h2>显存可运行性与数值检查</h2><p>默认分配器下main A完成200步，主均值4.5312秒；main B曾通过5步短运行，但正式运行在第5次更新时rank6 backward发生CUDA显存不足，峰值72829 MiB。失败记录完整保留。随后四组统一启用expandable_segments重新测量，主表不混用默认分配器结果。</p>
<p>统一设置后的8次正式运行均完成200次更新，所有记录的loss和梯度范数均为有限值。前3次更新、8个rank的CPU输入张量按完整更新拼接后，两种micro batch方案在每轮均逐项匹配。改变micro batch划分仍会改变模型内部flow-matching随机数的消耗顺序和浮点累加，因此不以短程loss高低判定模型效果。</p>
{table(['配置','31–200总loss均值','31–200 VLA loss均值','最大梯度范数','触发裁剪更新数','数值'],health)}
<h2>逐次运行与复现</h2>{table(['运行','31–200均值 秒','31–200中位数 秒','P95 秒','平均数据等待 毫秒','超过中位数1.3倍的更新'],per_run)}
<p>完整证据目录：<code>{ROOT}</code>。保留manifest.json、method_audit.json、effective_args.json、每rank逐步JSONL、输入指纹、smi.csv、source.txt、trainer.executed.py、每次启动日志及receipt。源码补丁为optimized-vs-main.patch。重新汇总依次运行summarize_study.py、audit_study.py、aggregate_study.py。</p>
</html>'''
(out/'report.html').write_text(body)
print(out/'report.html')
