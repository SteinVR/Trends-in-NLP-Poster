// Usage: NODE_PATH=<runtime node_modules> node export.mjs
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs/promises';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');
const dir = path.dirname(fileURLToPath(import.meta.url));
const browser = await chromium.launch({headless:true,executablePath:process.env.CHROMIUM_PATH || '/usr/bin/chromium',args:['--no-sandbox']});
try {
 const page = await browser.newPage({viewport:{width:2246,height:3179},deviceScaleFactor:1});
 await page.goto('file://'+path.join(dir,'index.html'));
 await page.evaluate(()=>document.fonts.ready);
 await page.emulateMedia({media:'print'});
 const check = await page.evaluate(()=>{
  const poster=document.querySelector('.poster'), pr=poster.getBoundingClientRect();
  const footer=document.querySelector('footer').getBoundingClientRect();
  return {width:pr.width,height:pr.height,contentBottom:footer.bottom,
   overflowing:footer.bottom>pr.bottom-parseFloat(getComputedStyle(poster).paddingBottom),
   missingImages:[...document.images].filter(x=>!x.complete||!x.naturalWidth).map(x=>x.src),
   sectionPositions:[...document.querySelectorAll('section,footer')].map(x=>({tag:x.tagName,top:x.getBoundingClientRect().top,bottom:x.getBoundingClientRect().bottom})),
   bodyFont:getComputedStyle(document.querySelector('p')).fontSize};
 });
 console.log(JSON.stringify(check,null,2));
 if(check.overflowing||check.missingImages.length)throw new Error('Layout overflow or missing images');
 await page.pdf({path:path.join(dir,'poster.pdf'),preferCSSPageSize:true,printBackground:true,displayHeaderFooter:false});
 await page.locator('.poster').screenshot({path:path.join(dir,'poster.png')});
 await fs.writeFile(path.join(dir,'layout-check.json'),JSON.stringify(check,null,2)+'\n');
}finally{await browser.close()}
