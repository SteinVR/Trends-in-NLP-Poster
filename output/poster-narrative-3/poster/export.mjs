import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import fs from 'node:fs/promises';
const require=createRequire(import.meta.url),{chromium}=require('playwright');
const dir=path.dirname(fileURLToPath(import.meta.url));
const tmp=path.resolve(dir,'../../../tmp/poster-narrative-3');
await fs.mkdir(tmp,{recursive:true});
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',args:['--no-sandbox']});
try{
 const page=await browser.newPage({viewport:{width:3180,height:2246},deviceScaleFactor:1});
 await page.goto('file://'+path.join(dir,'index.html'));await page.evaluate(()=>document.fonts.ready);await page.emulateMedia({media:'print'});
 const check=await page.evaluate(()=>{
  const rect=e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height,bottom:r.bottom,right:r.right}};
  const stageRows=[...document.querySelectorAll('.stage')].map(e=>['h3','p','.icon','.tech'].map(sel=>rect(e.querySelector(sel)).y));
  const stageRowSpreads=stageRows[0].map((_,i)=>Math.max(...stageRows.map(r=>r[i]))-Math.min(...stageRows.map(r=>r[i])));
  const baselineTops=[...document.querySelectorAll('.flow strong')].map(e=>rect(e).y);
  const baselineSpread=Math.max(...baselineTops)-Math.min(...baselineTops);
  return {stageRowSpreads,baselineSpread,sectionOrder:[...document.querySelectorAll('section')].map(e=>e.id),embeddingParent:document.querySelector('.flow .model').parentElement.textContent,rightColumnOrder:[...document.querySelectorAll('.right>section')].map(e=>e.id),poster:rect(document.querySelector('.poster')),cols:[...document.querySelectorAll('.col')].map(e=>({class:e.className,...rect(e),scrollHeight:e.scrollHeight})),sections:[...document.querySelectorAll('section')].map(e=>({id:e.id,...rect(e)})),paragraphFonts:[...new Set([...document.querySelectorAll('p')].map(e=>getComputedStyle(e).fontSize))],missingImages:[...document.images].filter(e=>!e.complete||!e.naturalWidth).map(e=>e.src)};
 });
 const badColumns=check.cols.filter(c=>c.scrollHeight>Math.ceil(c.height)+1);
 const badSections=check.sections.filter(s=>s.bottom>check.poster.bottom-30+1);
 const smallParagraphs=check.paragraphFonts.filter(s=>parseFloat(s)<32);
 await fs.writeFile(path.join(tmp,'layout-check.json'),JSON.stringify(check,null,2));
 if(badColumns.length||badSections.length||smallParagraphs.length||check.missingImages.length||check.stageRowSpreads.some(v=>v>0.5)||check.rightColumnOrder.join(',')!=='s9,s10,s11,s12,s13'||check.sectionOrder.join(',')!==Array.from({length:13},(_,i)=>'s'+(i+1)).join(',')||check.embeddingParent!=='Semantic retrievalQwen3-Embedding-0.6B')
   throw new Error('Poster validation failed: '+JSON.stringify({badColumns,badSections,smallParagraphs,missingImages:check.missingImages,stageRowSpreads:check.stageRowSpreads,baselineSpread:check.baselineSpread,rightColumnOrder:check.rightColumnOrder}));
 console.log('A1 landscape layout verified; all paragraphs at least 24 pt.');
 await page.pdf({path:path.join(dir,'poster.pdf'),preferCSSPageSize:true,printBackground:true,displayHeaderFooter:false});
 await page.locator('.poster').screenshot({path:path.join(dir,'poster.png')});
}finally{await browser.close()}
