"""Render poster figures from the completed experiment; no API calls."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
font_manager.fontManager.addfont(str(ASSETS / 'NotoSans-Regular.ttf'))
font_manager.fontManager.addfont(str(ASSETS / 'NotoSans-Bold.ttf'))
plt.rcParams.update({'font.family':'Noto Sans','font.size':24,'svg.fonttype':'path',
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#9aafbf','text.color':'#173248',
                     'axes.labelcolor':'#173248','xtick.color':'#173248','ytick.color':'#516578'})
BLUE, SKY, INK = '#0069aa', '#438ba7', '#173248'
with (ROOT.parent / 'results/metrics.csv').open() as f:
    all_rows = list(csv.DictReader(f))
rows = [r for r in all_rows if r['corpus']=='mixed']
assert [r['variant'] for r in rows]==[f'R{i}' for i in range(7)]
values = {k:[float(r[k])*100 for r in rows] for k in ['Q','precision','recall']}
fig, axes = plt.subplots(2,1,figsize=(21.97,6.40),sharex=True)
fig.subplots_adjust(left=.052,right=.973,top=.91,bottom=.205,hspace=.62)
x = list(range(7))
for ax in axes:
    ax.set_ylim(0,105); ax.set_xlim(-.18,6.18)
    ax.set_yticks([0,50,100]); ax.set_yticklabels(['0','50','100'])
    ax.grid(axis='y',color='#dce7ed',linewidth=1.1)
    ax.set_axisbelow(True); ax.tick_params(length=0,pad=10)
    ax.spines['left'].set_visible(False)
axes[0].set_title('Answer quality Q, %  ·  100 questions',loc='left',fontweight='bold',fontsize=27,pad=16)
axes[0].plot(x,values['Q'],color=BLUE,lw=3.6,marker='o',ms=10,zorder=3)
for xx,y in zip(x,values['Q']):
    axes[0].annotate(f'{y:.1f}',(xx,y),xytext=(0,13),textcoords='offset points',ha='center',fontsize=25,fontweight='bold',color=BLUE)
axes[1].set_title('Citation quality, %  ·  95 questions',loc='left',fontweight='bold',fontsize=27,pad=18)
axes[1].plot(x,values['precision'],color=BLUE,lw=3.6,marker='o',ms=9,label='Precision')
axes[1].plot(x,values['recall'],color=SKY,lw=3.6,ls=(0,(5,3)),marker='s',ms=9,label='Recall')
axes[1].legend(loc='upper left',bbox_to_anchor=(.31,1.48),ncol=2,frameon=False,fontsize=24,handlelength=1.8,columnspacing=1.2)
for metric,idx,offset,color in [('precision',0,(0,14),BLUE),('precision',5,(0,-30),BLUE),('precision',6,(0,-31),BLUE),('recall',0,(0,13),SKY),('recall',5,(0,14),SKY),('recall',6,(0,14),SKY)]:
    y=values[metric][idx]
    axes[1].annotate(f'{y:.1f}',(idx,y),xytext=offset,textcoords='offset points',ha='center',fontsize=24,color=color,fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(['R0\nBaseline','R1\n+ OCR','R2\n+ Multiscale','R3\n+ Lexical','R4\n+ Reranker','R5\n+ Typed','R6\n+ Citations'],fontsize=24)
fig.savefig(ASSETS/'sequence.svg',facecolor='white')
plt.close(fig)

orig=next(r for r in all_rows if r['corpus']=='original' and r['variant']=='R0')
vals=[float(orig['Q'])*100,values['Q'][0],values['Q'][1]]
fig,ax=plt.subplots(figsize=(10.4,2.15))
fig.subplots_adjust(left=.39,right=.92,top=.99,bottom=.23)
ax.barh([2,1,0],vals,height=.53,color=['#9aaebc','#c1d3de',BLUE])
ax.set_yticks([2,1,0]);ax.set_yticklabels(['Original · R0','Mixed · R0','Mixed + OCR · R1'],fontsize=24)
ax.set_xlim(0,100);ax.set_xticks([0,50,100]);ax.set_xticklabels(['0','50','100%'],fontsize=24)
ax.tick_params(length=0,pad=9);ax.spines['left'].set_visible(False)
for yy,v in zip([2,1,0],vals):ax.text(v+2,yy,f'{v:.1f}',va='center',fontsize=24,fontweight='bold',color=INK)
fig.savefig(ASSETS/'ocr.svg',facecolor='white')
plt.close(fig)
print('Figures generated from results/metrics.csv')
