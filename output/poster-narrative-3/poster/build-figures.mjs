// Rebuild HTML chart assets from the measured CSV. No PDF/PPTX export or model calls.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const dir=path.dirname(fileURLToPath(import.meta.url));
const csv=(await fs.readFile(path.join(dir,'../results/metrics.csv'),'utf8')).trim().split(/\r?\n/);
const keys=csv.shift().split(',');
const rows=csv.map(line=>Object.fromEntries(line.split(',').map((v,i)=>[keys[i],v])));
const mixed=rows.filter(row=>row.corpus==='mixed');
if(mixed.map(row=>row.variant).join(',')!=='R0,R1,R2,R3,R4,R5,R6')throw new Error('Unexpected conditions');
const fmt=v=>(Number(v)*100).toFixed(1);
const escape=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;');
const text=(x,y,label,cls='',anchor='start')=>`<text x="${x}" y="${y}" class="${cls}" text-anchor="${anchor}">${escape(label)}</text>`;
const width=2100,height=390,panelWidth=672,gap=42,plotTop=142,plotBottom=323;
const metrics=[['Q','Answer quality Q','100 questions','#0069aa'],['precision','Citation precision','95 questions','#0069aa'],['recall','Citation recall','95 questions','#438ba7']];
let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="2100" height="390" viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="chart-title chart-desc"><title id="chart-title">Answer quality, citation precision and citation recall across R0 to R6</title><desc id="chart-desc">Three panels use the same 0 to 100 percent scale and the same cumulative variants. Values come from results/metrics.csv.</desc><style>text{font-family:Noto,'Noto Sans',sans-serif;font-size:32px;fill:#173248}.title{font-size:35px;font-weight:700}.scope,.axis{fill:#516578}.value{font-size:32px;font-weight:700;paint-order:stroke;stroke:#fff;stroke-width:8px;stroke-linejoin:round}.stage{font-size:32px}.grid{stroke:#dce7ed;stroke-width:1.6}.line{fill:none;stroke-width:4.5;stroke-linejoin:round}</style>`;
for(const [panel,[key,title,n,color]] of metrics.entries()){
 const left=panel*(panelWidth+gap),x0=left+104,x1=left+panelWidth-39;
 svg+=`<g data-metric="${key}">`+text(left,44,title+', %','title')+text(left,92,n,'scope');
 for(const tick of [0,50,100]){
  const y=plotBottom-(plotBottom-plotTop)*tick/100;
  svg+=`<line x1="${x0-8}" y1="${y}" x2="${x1+12}" y2="${y}" class="grid"/>`+text(x0-45,y+11,String(tick),'axis','end');
 }
 const pts=mixed.map((row,i)=>({x:x0+i*(x1-x0)/6,y:plotBottom-(plotBottom-plotTop)*Number(row[key]),v:fmt(row[key]),id:row.variant}));
 svg+=`<polyline class="line" stroke="${color}" ${key==='recall'?'stroke-dasharray="11 7"':''} points="${pts.map(p=>`${p.x},${p.y}`).join(' ')}"/>`;
 for(const [i,p] of pts.entries()){
  svg+=key==='recall'?`<rect x="${p.x-6}" y="${p.y-6}" width="12" height="12" fill="${color}"/>`:`<circle cx="${p.x}" cy="${p.y}" r="6" fill="${color}"/>`;
  // Place labels over each point, except the R2 dips, whose labels sit below the incoming line.
  svg+=`<g class="point-label" data-variant="${p.id}" data-value="${p.v}">${text(p.x,p.y+((key==='Q'||key==='recall')&&i===2?44:(key==='precision'&&i===5)||(key==='recall'&&i===0)?38:-22),p.v,'value','middle')}</g>`;
  svg+=text(p.x,365,p.id,'stage','middle');
 }
 svg+='</g>';
}

svg+='</svg>';
await fs.writeFile(path.join(dir,'../results/pipeline-metrics.svg'),svg+'\n');
const original=rows.find(r=>r.corpus==='original'&&r.variant==='R0');
const controls=[
 ['Original corpus · R0',fmt(original.Q),'#91aabd'],
 ['Mixed corpus (original + synthetic scans) · R0',fmt(mixed[0].Q),'#b0c7d7'],
 ['Mixed corpus + OCR · R1',fmt(mixed[1].Q),'#0069aa']
];
let ocr=`<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="284" viewBox="0 0 1000 284" role="img" aria-labelledby="ocr-title ocr-desc"><title id="ocr-title">OCR recovers answer quality on scanned documents</title><desc id="ocr-desc">Original corpus R0: 65.4 percent. Mixed corpus R0: 46.2 percent. Mixed corpus with OCR R1: 63.2 percent. Bars share a zero to 100 percent scale.</desc><style>text{font-family:Noto,'Noto Sans',sans-serif;font-size:32px;fill:#173248}.value{font-weight:700}.axis{fill:#516578}</style>`;
for(const [i,[label,value,color]] of controls.entries()){
 const y=40+i*77;
 ocr+=text(0,y,label)+`<rect x="0" y="${y+15}" width="1000" height="18" fill="#f0f5f8"/><rect x="0" y="${y+15}" width="${Number(value)*10}" height="18" fill="${color}"/>`+text(1000,y,value,'value','end');
}
for(const value of [0,50,100])ocr+=text(value*10,267,value===100?'100%':String(value),'axis',value===0?'start':value===100?'end':'middle');
ocr+='</svg>';
await fs.writeFile(path.join(dir,'../results/ocr-control.svg'),ocr+'\n');
console.log('Updated canonical SVG figures from metrics.csv.');
