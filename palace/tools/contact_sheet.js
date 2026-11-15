/* Contact sheets: every room's threshold view, and the lookout with each zone lit, so rooms that could be
   confused are seen side by side. Run from course/:
     node palace/tools/contact_sheet.js <out-dir> [--size 390x844] [--rooms 04,08] [--no-lookout]
   Writes <out-dir>/threshold-<room>.png, <out-dir>/lookout-<k>.png, and the two sheets
   <out-dir>/sheet-thresholds.png and <out-dir>/sheet-lookout.png. The walk runs on the page's test clock
   (__palace.clock('manual')), so every shot is taken at the same moment of the walk on any machine. */
const path=require('path'),fs=require('fs');
const { chromium } = require(process.env.PLAYWRIGHT||'/Users/vincentbarthelemy/Perso/learnkit/node_modules/playwright');
const URL='file://'+path.resolve(__dirname,'..','index.html');
const args=process.argv.slice(2);const out=path.resolve(args[0]&&!args[0].startsWith('--')?args[0]:'palace-sheets');
const opt=k=>{const i=args.indexOf('--'+k);return i>=0?args[i+1]:null;};
const [W,H]=(opt('size')||'390x844').split('x').map(Number);const only=opt('rooms')?opt('rooms').split(','):null;
fs.mkdirSync(out,{recursive:true});
const sheet=async(b,files,cols,name,cw)=>{const p=await b.newPage({viewport:{width:cols*cw,height:400},deviceScaleFactor:1});
  const html='<body style="margin:0;background:#000;display:grid;grid-template-columns:repeat('+cols+','+cw+'px)">'+files.map(f=>'<div style="position:relative"><img style="width:'+cw+'px;display:block" src="file://'+f+'"><span style="position:absolute;left:4px;top:2px;color:#ff0;font:bold 15px monospace;background:#000a;padding:1px 4px">'+path.basename(f,'.png').replace(/^(threshold|lookout)-/,'')+'</span></div>').join('')+'</body>';
  const hp=path.join(out,name+'.html');fs.writeFileSync(hp,html);await p.goto('file://'+hp);await p.waitForFunction(()=>[...document.images].every(i=>i.complete));
  await p.screenshot({path:path.join(out,name+'.png'),fullPage:true});await p.close();fs.unlinkSync(hp);};
(async()=>{const b=await chromium.launch();const ctx=await b.newContext({viewport:{width:W,height:H},deviceScaleFactor:2});const p=await ctx.newPage();
  const errs=[];p.on('pageerror',e=>errs.push(e.message));p.on('console',m=>{if(m.type()==='error')errs.push(m.text());});
  await p.goto(URL+'#gate');await p.waitForFunction(()=>window.__palace);await p.evaluate(()=>{try{localStorage.clear();}catch(e){}window.__palace.clock('manual');});
  const rooms=await p.evaluate(()=>window.__palace.STOPS().filter(s=>s.kind==='room').map(s=>({room:s.room,i:s.i})));
  const shots=[];
  for(const r of rooms){if(only&&!only.includes(r.room))continue;
    await p.evaluate(i=>{const P=window.__palace;P.goTo(i,{instant:true});P.advance(2500);document.getAnimations().forEach(a=>a.finish());},r.i);
    const f=path.join(out,'threshold-'+r.room+'.png');await p.screenshot({path:f});shots.push(f);}
  if(shots.length)await sheet(b,shots,8,'sheet-thresholds',W/2|0);
  if(!args.includes('--no-lookout')){const L=[];await p.evaluate(()=>{const P=window.__palace,n=P.STOPS().length-1;P.goTo(n,{instant:true});P.advance(200);P.S.lookoutT0=P.S.vnow;});
    for(let k=0;k<=8;k++){await p.evaluate(k=>{const P=window.__palace;const want=P.S.lookoutT0+(k*6.5+5.6)*1000;P.advance(want-P.S.vnow);document.getAnimations().forEach(a=>a.finish());},k);
      const f=path.join(out,'lookout-'+k+'.png');await p.screenshot({path:f});L.push(f);}
    await sheet(b,L,5,'sheet-lookout',W/2|0);}
  const st=await p.evaluate(()=>window.__palace.stats());console.log('stats at the lookout',JSON.stringify(st));
  console.log(errs.length?'errors: '+errs.slice(0,3).join(' | '):'no page errors');console.log('wrote',shots.length,'threshold shots and the sheets to',out);await b.close();})();
