"""Reconstruct the retained combined figure from saved values; not the original renderer."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


def render(root, out):
    if out.exists() and any(out.iterdir()):raise RuntimeError('Choose an empty output directory.')
    out.mkdir(parents=True,exist_ok=True)
    banks={k:[json.loads(s) for s in (root/'evidence/historical'/f'{k}.jsonl').read_text().splitlines()]
           for k in ('task2_curated','task2_large','task3_curated','task3_large')}
    t1=json.loads((root/'evidence/historical/task1_recorded_summary.json').read_text())['values']
    task1_values=[[1-t1['abstain_rate'],t1['avg_grounded_token_ratio']]]
    task1_labels=[f"T1 all ({t1['samples']:,})"]
    for mode,label in [('coordinates','Coord.'),('function','Function'),('qtl','QTL')]:
        r=t1['per_mode'][mode]
        task1_labels.append(f"{label} ({r['samples']:,})")
        task1_values.append([r['route_accuracy'],r['avg_grounded_token_ratio']])
    metrics=('faithfulness','ar','csr','cvr')
    values=[[statistics.mean(r[k] for r in rows) for k in metrics] for rows in banks.values()]
    labels=[f'{label} ({len(rows)})' for label,rows in zip(('T2 curated','T2 large','T3 curated','T3 large'),banks.values())]
    plt.rcParams.update({'font.family':['Arial','DejaVu Sans'],'font.size':7.4,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(7.55,3.08),facecolor='white')
    grid=fig.add_gridspec(2,3,width_ratios=[1.25,1,1],left=.16,right=.99,bottom=.16,top=.89,wspace=.42,hspace=.52)
    cmap=LinearSegmentedColormap.from_list('scorecard',['#EAF0F7','#AADCA9','#0F4D92'])
    norm=Normalize(.70,1)

    def scorecard(ax,rows,cols,data):
        ax.set_xlim(0,len(cols));ax.set_ylim(len(rows),0)
        ax.set_xticks([i+.5 for i in range(len(cols))],cols,fontsize=6.5)
        ax.set_yticks([i+.5 for i in range(len(rows))],rows,fontsize=6.3)
        ax.tick_params(axis='x',top=True,labeltop=True,bottom=False,labelbottom=False,length=0,pad=2)
        ax.tick_params(axis='y',length=0,pad=2)
        for spine in ax.spines.values():spine.set_visible(False)
        for i,row in enumerate(data):
            for j,v in enumerate(row):
                ax.add_patch(Rectangle((j+.055,i+.055),.89,.89,facecolor=cmap(norm(v)),edgecolor='white',linewidth=.35))
                ax.text(j+.5,i+.5,'100%' if v>=.995 else f'{100*v:.1f}%',ha='center',va='center',
                        fontsize=5.7,color='white' if v>=.94 else '#272727')
    scorecard(fig.add_subplot(grid[0,0]),task1_labels,['Route','Ground.'],task1_values)
    scorecard(fig.add_subplot(grid[1,0]),labels,['Faith.','AR','CSR','CVR'],values)
    scatter={}
    for col,key,color,title in [(1,'task2_large','#0F4D92','Task 2 large audit'),(2,'task3_large','#9A4D8E','Task 3 large audit')]:
        rows=banks[key];ax=fig.add_subplot(grid[:,col])
        x=[r['faithfulness'] for r in rows];y=[r['ar'] for r in rows]
        ax.scatter(x,y,s=[20+42*r['cvr'] for r in rows],color=color,alpha=.74,edgecolors='white',linewidths=.5)
        ax.axvline(statistics.mean(x),color=color,lw=1.1,alpha=.9)
        ax.axhline(statistics.mean(y),color=color,lw=1.1,alpha=.38)
        ax.set_xlim(.79,1);ax.set_ylim(.91,1.005)
        ax.set_xticks([.80,.86,.92,.98]);ax.set_yticks([.92,.96,1])
        ax.set_xlabel('Faithfulness');ax.set_ylabel('AR')
        ax.set_title(title,fontsize=7.3)
        ax.text(.03,.025,f'mean {statistics.mean(x):.3f}, {statistics.mean(y):.3f}',transform=ax.transAxes,fontsize=6)
        ax.grid(color='#CFCECE',lw=.5,alpha=.6)
        scatter[key]=[dict(example_id=r['example_id'],faithfulness=r['faithfulness'],ar=r['ar'],cvr=r['cvr']) for r in rows]
    fig.text(.04,.965,'(a) Task-aligned scorecard',fontsize=8)
    fig.text(.46,.965,'(b) Query-level metric distributions',fontsize=8)
    for ext in ('pdf','svg','png'):
        fig.savefig(out/f'figure2_overall_results_combined.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    payload=dict(renderer_provenance='New reconstruction of retained numeric panels, not recovered original combined-layout code.',
                 task1_scope='Recorded historical aggregate only; original answer logs unavailable.',
                 task1_labels=task1_labels,task1_values=task1_values,synthesis_labels=labels,
                 synthesis_metrics=list(metrics),synthesis_values=values,scatter=scatter)
    (out/'figure_data.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    return dict(rendered='figure2_overall_results_combined.pdf',scatter_points=sum(map(len,scatter.values())),
                source='retained numeric rows and Task 1 recorded summary',pixel_identity_claimed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    print(json.dumps(render(Path(__file__).resolve().parents[1],p.parse_args().out),indent=2))
