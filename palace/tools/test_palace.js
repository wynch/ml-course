/* Browser tests for the palace page, with the Playwright the course's learnkit installs.
   node course/palace/tools/test_palace.js [shots-dir]
   Opens index.html from disk; headless Chromium gives WebGL through SwiftShader.
   The walk runs on its own clock. Most sections switch the page to the test clock (__palace.clock('manual')) and
   step it with __palace.advance(ms) or __palace.runUntil(pred, ms): a check never sleeps on the wall clock, and a
   slow machine takes longer to run the same steps but sees the same walk. Only the sections about animation
   frames themselves (a lost context, a hidden page, frame times) use real time, and they wait on events. */
const path=require('path');
const { chromium } = require(process.env.PLAYWRIGHT||'/Users/vincentbarthelemy/Perso/learnkit/node_modules/playwright');
const URL='file://'+path.resolve(__dirname,'..','index.html');
const T0=Date.now();let fails=0;const ok=(c,msg)=>{if(!c){fails++;console.log('FAIL',msg);}else console.log('ok  ',msg);};
async function page(b,vp,opts={}){const ctx=await b.newContext({viewport:vp,deviceScaleFactor:opts.dpr||2,reducedMotion:opts.reduce?'reduce':'no-preference',hasTouch:!!opts.touch});
  const p=await ctx.newPage();p.errs=[];p.remote=[];p.on('request',r=>{if(!r.url().startsWith('file://')&&!r.url().startsWith('data:'))p.remote.push(r.url());});p.on('pageerror',e=>p.errs.push(e.message));p.on('console',m=>{if(m.type()==='error')p.errs.push(m.text());});return p;}
/* the page is up, and on the test clock unless asked otherwise */
const ready=async(p,manual=true)=>{await p.waitForFunction(()=>window.__palace&&window.__palace.STOPS().length>0);if(manual)await p.evaluate(()=>window.__palace.clock('manual'));};
/* step the walk's clock; CSS transitions on the card are finished too, so a click never waits for one */
const adv=(p,ms)=>p.evaluate(ms=>{window.__palace.advance(ms);document.getAnimations().forEach(a=>a.finish());},ms);
const arrived=(p,i,ms=30000)=>p.evaluate(([i,ms])=>{const r=window.__palace.runUntil(()=>window.__palace.S.arrived===i,ms);document.getAnimations().forEach(a=>a.finish());return r;},[i,ms]);
(async()=>{const b=await chromium.launch({args:['--enable-precise-memory-info','--js-flags=--expose-gc']});
  /* 1. the whole route by Next, phone portrait */
  {const p=await page(b,{width:390,height:844});await p.goto(URL);await ready(p);
    const n=await p.evaluate(()=>{const P=window.__palace;try{localStorage.clear();}catch(e){}P.goTo(0,{instant:true});return P.STOPS().length;});
    const want=await p.evaluate(()=>{const P=window.__palace;return 2+Object.keys(P.ROOMS).length+P.OBJS.length;});
    ok(n===177&&want===177,'177 stops: gate, 39 thresholds, 136 loci, lookout ('+n+')');
    /* the shape of the walk: 3 or 4 loci a room, at least 2.8 m apart; the rail is continuous and never runs into a signature piece; no leg takes much over 8 s */
    const shape=await p.evaluate(()=>{const P=window.__palace,out={few:[],near:[],gaps:0,sig:P.sigCrossings().length,legs:P.legTimes()};
      Object.keys(P.ROOMS).forEach(id=>{const o=P.OBJS.filter(o=>o.room===id);if(o.length<3||o.length>4)out.few.push(id+':'+o.length);
        for(let i=0;i<o.length;i++)for(let j=i+1;j<o.length;j++){const d=Math.hypot(o[i].pos[0]-o[j].pos[0],o[i].pos[1]-o[j].pos[1]);if(d<2.8)out.near.push(o[i].id+'~'+o[j].id+' '+d.toFixed(2));}});
      const R=P.PATH();for(let k=1;k<R.n;k++){if(Math.hypot(R.X[k]-R.X[k-1],R.Y[k]-R.Y[k-1],R.H[k]-R.H[k-1])>.35)out.gaps++;}
      out.maxLeg=Math.max(...out.legs.map(l=>l.t));out.faded=out.legs.filter(l=>l.fade).length;delete out.legs;return out;});
    ok(shape.few.length===0,'every room has 3 or 4 loci '+shape.few.join(' '));ok(shape.near.length===0,'loci in a room stand at least 2.8 m apart '+shape.near.join(' '));
    ok(shape.gaps===0,'the rail is continuous: no step longer than 0.35 m ('+shape.gaps+')');ok(shape.sig===0,'the rail never passes through a signature piece ('+shape.sig+' samples)');
    ok(shape.maxLeg<=8.5,'no leg takes over 8.5 s, long ones fading first (longest '+shape.maxLeg.toFixed(1)+' s; '+shape.faded+' legs fade)');
    const cards=await p.evaluate(()=>{const P=window.__palace,bad=[];Object.values(P.ROOMS).forEach(r=>r.objects.concat(r.also).forEach(o=>{const c=o.card||{};if(!(c.title&&c.text&&o.checked&&o.checked.length>10&&o.idea&&o.image))bad.push(o.id);}));return bad;});
    ok(cards.length===0,'every locus and every demoted idea has a card with a title, a text and a note of where it was checked '+cards.join(' '));
    /* every room has a signature piece, standing: its box rises at least 0.3 m above the floor, or it is the sunken slope */
    const sig=await p.evaluate(()=>{const P=window.__palace,out=[];Object.values(P.ROOMS).forEach(r=>{const bx=P.sigboxes().filter(q=>q.id===r.id);const F=r.level*4;if(!bx.length)out.push(r.id+' none');else if(r.id!=='14'&&!bx.some(q=>q.b[5]-F>=.3))out.push(r.id+' flat');});return out;});
    ok(sig.length===0,'every room has a signature piece that stands off its floor '+sig.join(' '));
    /* a planted crossing of a signature piece is caught */
    ok(await p.evaluate(()=>{const P=window.__palace,b=P.sigboxes()[0].b;const R=P.PATH();const k=5,sx=R.X[k],sy=R.Y[k],sh=R.H[k];R.X[k]=(b[0]+b[3])/2;R.Y[k]=(b[1]+b[4])/2;R.H[k]=(b[2]+b[5])/2;const n=P.sigCrossings().length;R.X[k]=sx;R.Y[k]=sy;R.H[k]=sh;return n>0&&P.sigCrossings().length===0;}),'a planted signature crossing is caught');
    /* the walk goes from the gate through the stair core and down the residual corridor into the atrium */
    ok(await p.evaluate(()=>{const P=window.__palace,R=P.PATH(),st=P.STOPS();const a=R.stopArc[st.findIndex(s=>s.kind==='room'&&s.room==='03')],b=R.stopArc[st.findIndex(s=>s.kind==='room'&&s.room==='04')];const seen=new Set();for(let k=Math.round(a/.2);k<Math.round(b/.2);k++)seen.add(R.ND[k]);return seen.has('core')&&seen.has('residual_corridor');}),'from the gate to the atrium the walk crosses the core and the residual corridor');
    const walls=await p.evaluate(()=>window.__palace.wallCrossings().length);ok(walls===0,'the rail never crosses a wall ('+walls+' crossings)');
    const floors=await p.evaluate(()=>window.__palace.floorCrossings().length);ok(floors===0,'the rail never passes through a floor or a ceiling ('+floors+' crossings)');
    /* both tests must be able to fail: plant a crossing of each kind, see it reported, take it out */
    const planted=await p.evaluate(()=>{const P=window.__palace,R=P.PATH();let k=-1;for(let i=5;i<R.n-5;i++)if(R.ND[i]==='04'&&R.ND[i+1]==='04'&&R.Y[i]>3&&R.Y[i]<7){k=i;break;}
      const sy=R.Y[k+1],sh=R.H[k+1];R.Y[k+1]=-2;const w=P.wallCrossings().length;R.Y[k+1]=sy;R.H[k+1]=-1;const f=P.floorCrossings().length;R.H[k+1]=sh;return{w,f,w0:P.wallCrossings().length,f0:P.floorCrossings().length};});
    ok(planted.w>0&&planted.w0===0,'a planted wall crossing is caught ('+planted.w+'), and gone when removed');ok(planted.f>0&&planted.f0===0,'a planted floor crossing is caught ('+planted.f+'), and gone when removed');
    const hist0=await p.evaluate(()=>history.length);
    const peak={points:0,lineVerts:0,calls:0,tris:0};let heap0=0;const at={};
    let noCard=[],badObj=[],late=[];
    for(let i=0;i<n;i++){if(i>0)await p.click('#next');if(!await arrived(p,i))late.push(i);
      const r=await p.evaluate(i=>{const P=window.__palace,st=P.STOPS()[i];const idea=document.getElementById('c-idea').textContent.trim();
        const o=st.kind==='object'&&st.ob.furnished?P.objOnScreen(st.ob):null;const dy=Math.abs(((P.CAM.yaw-st.yaw)%(2*Math.PI)+3*Math.PI)%(2*Math.PI)-Math.PI);return{idea,kind:st.kind,id:st.ob?st.ob.id:st.kind==='room'?st.room:st.kind,o,dy};},i);
      if(!r.idea)noCard.push(i);
      const stt=await p.evaluate(()=>window.__palace.stats());for(const k in peak)if(stt[k]>peak[k]){peak[k]=stt[k];peak[k+'At']=r.id;}
      if(['04','04-1'].includes(r.id))at[r.id]=stt;
      if(i===20)heap0=await p.evaluate(()=>{if(window.gc)window.gc();return performance.memory?performance.memory.usedJSHeapSize:0;});
      if(r.dy>.03)badObj.push((r.id||i)+' faces '+r.dy.toFixed(2)+' rad off');
      if(r.o){const [x0,y0,x1,y1]=r.o.box;if(r.o.n<60||x0<-1||y0<-1||x1>r.o.W+1||y1>r.o.H+1)badObj.push(r.id+' n='+r.o.n+' box='+r.o.box.map(v=>v|0).join(','));}}
    ok(late.length===0,'Next reaches every stop within 30 s of the walk\'s clock '+late.join(','));
    ok(noCard.length===0,'every stop has a card '+(noCard.length?noCard.join(','):''));
    ok(badObj.length===0,'every furnished object has more than 60 points on screen, all inside the view'+(badObj.length?': '+badObj.join(' | '):''));
    ok(p.errs.length===0,'no console errors on the walk '+p.errs.slice(0,3).join(' | '));
    const heap1=await p.evaluate(()=>{if(window.gc)window.gc();return performance.memory?performance.memory.usedJSHeapSize:0;});
    const short=s=>s&&JSON.stringify({points:s.points,lineVerts:s.lineVerts,calls:s.calls,tris:s.tris});
    console.log('peak per frame over the walk',JSON.stringify(peak));console.log('at #04',short(at['04']),'· at #04/1',short(at['04-1']));console.log('heap after 20 stops',(heap0/1e6).toFixed(1),'MB; after all 177',(heap1/1e6).toFixed(1),'MB');
    ok(peak.points<=9000,'never more than 9,000 points in a frame ('+peak.points+' at '+peak.pointsAt+')');ok(peak.lineVerts<=20000,'never more than 20,000 line vertices in a frame, both passes ('+peak.lineVerts+' at '+peak.lineVertsAt+')');
    ok(peak.calls<=42,'never more than 42 draw calls ('+peak.calls+' at '+peak.callsAt+')');
    ok(heap1-heap0<8e6,'the heap grows by under 8 MB between stop 20 and the end of the walk');
    ok(await p.evaluate(h=>history.length===h,hist0),'walking adds no history entries: Back leaves the page');
    /* the lookout: each zone lights in turn and the view closes in on it; the tower's storeys lift apart while it is lit, and come back down after */
    const look=await p.evaluate(()=>{const P=window.__palace,out=[];P.S.lookoutT0=P.S.vnow;
      for(let k=0;k<=8;k++){P.advance(P.S.lookoutT0+(k*6.5+5.6)*1000-P.S.vnow);const s=P.stats();out.push({k,lit:P.S.rec&&P.S.rec.k,lift:+(P.S.lift||0).toFixed(2),points:s.points,calls:s.calls,lineVerts:s.lineVerts,tris:s.tris});}
      P.goTo(P.STOPS().length-2,{instant:true});P.advance(3000);out.push({after:true,lift:+(P.S.lift||0).toFixed(2)});return out;});
    console.log('at #lookout, zone by zone',JSON.stringify(look.slice(0,9).map(l=>[l.k,l.points,l.calls,l.lineVerts,l.tris])));
    ok(look.slice(0,9).every(l=>l.lit===l.k&&l.points<=9000&&l.calls<=42&&l.lineVerts<=20000),'the lookout lights each zone in turn, within the budgets');
    ok(look[7].lift===1&&look[6].lift===0&&look[9].lift===0,'the tower lifts apart while it is lit, and only then ('+look.map(l=>l.lift).join(' ')+')');
    ok(p.remote.length===0,'no request leaves the folder'+(p.remote.length?': '+p.remote.slice(0,3).join(' '):''));
    await p.close();}
  /* 1b. every object stop at the other two sizes: the object is in view and faced squarely */
  for(const vp of [{width:844,height:390},{width:1280,height:800}]){const p=await page(b,vp,{dpr:1});await p.goto(URL);await ready(p);
    const bad=await p.evaluate(()=>{const P=window.__palace,st=P.STOPS(),out=[];
      for(const s of st){if(s.kind!=='object'||!s.ob.furnished)continue;P.goTo(s.i,{instant:true});P.advance(34);const o=P.objOnScreen(s.ob);const [x0,y0,x1,y1]=o.box;
        const dy=Math.abs(((P.CAM.yaw-s.yaw)%(2*Math.PI)+3*Math.PI)%(2*Math.PI)-Math.PI);if(o.n<60||x0<-1||y0<-1||x1>o.W+1||y1>o.H+1||dy>.03)out.push(s.ob.id+' n='+o.n+' dy='+dy.toFixed(2));}return out;});
    ok(bad.length===0,'at '+vp.width+'x'+vp.height+' every furnished object is in view and faced'+(bad.length?': '+bad.join(' | '):''));await p.close();}
  /* 2. scroll, drag-look, tap-to-go, mini-map, floor switch, rehearsal, list */
  {const p=await page(b,{width:1280,height:800},{dpr:1});await p.goto(URL+'#04');await ready(p);await adv(p,800);
    const i0=await p.evaluate(()=>window.__palace.S.target);
    await p.mouse.move(900,400);for(let k=0;k<4;k++){await p.mouse.wheel(0,200);await adv(p,40);}await adv(p,1500);
    const i1=await p.evaluate(()=>window.__palace.S.target);ok(i1>i0,'wheel walks forward ('+i0+' → '+i1+')');
    await arrived(p,i1);
    const y0=await p.evaluate(()=>window.__palace.CAM.yaw);await p.mouse.move(900,400);await p.mouse.down();await p.mouse.move(700,380,{steps:8});await adv(p,17);
    const y1=await p.evaluate(()=>window.__palace.CAM.yaw);await p.mouse.up();await adv(p,1500);const y2=await p.evaluate(()=>window.__palace.CAM.yaw);
    ok(Math.abs(y1-y0)>.2&&Math.abs(y2-y0)<.03,'drag turns the head and springs back ('+[y0,y1,y2].map(v=>v.toFixed(2))+')');
    /* tap-to-go: from the atrium threshold, tap object 3 */
    await p.evaluate(()=>{const P=window.__palace;const i=P.STOPS().findIndex(s=>s.kind==='room'&&s.room==='04');P.goTo(i,{instant:true});});await adv(p,400);
    const tp=await p.evaluate(()=>{const P=window.__palace;const cr=document.getElementById('card').getBoundingClientRect();const ob=P.OBJS.find(o=>{if(o.room!=='04')return false;const q=P.project(o.pos[0],o.pos[1],o.pos[2]);return q&&q.x>cr.right+30&&q.x<P.CAM.W-30&&q.y>80&&q.y<P.CAM.H-80;});if(!ob)return null;const q=P.project(ob.pos[0],ob.pos[1],ob.pos[2]);return q&&{x:q.x,y:q.y,want:P.STOPS().findIndex(s=>s.ob===ob)};});
    if(tp){await p.mouse.click(tp.x,tp.y);await arrived(p,tp.want);const at=await p.evaluate(()=>window.__palace.S.arrived);ok(at===tp.want,'tap on an object walks to it');}else ok(false,'an object of the atrium visible from the threshold');
    await p.click('#mini');await adv(p,400);ok(await p.isVisible('#map .panel'),'the mini-map opens the map');
    await p.click('#map-tabs [data-plan="cellar"]');ok(await p.$('#map-svg [data-room="05a"]')!==null,'floor switch shows the cellar plan');
    await p.screenshot({path:process.argv[2]?path.join(process.argv[2],'desktop-map-cellar.png'):'/dev/null'}).catch(()=>{});
    await p.click('#map-tabs [data-plan="ground"]');await p.click('#map-svg [data-room="09"]');
    const want=await p.evaluate(()=>window.__palace.STOPS().findIndex(s=>s.kind==='room'&&s.room==='09'));await arrived(p,want);
    ok(await p.evaluate(w=>window.__palace.S.arrived===w,want),'tapping room 09 on the map walks there along the route');
    await p.click('#btn-menu');await p.click('[data-m="rehearse"]');await adv(p,2500);
    ok((await p.textContent('#c-acts')).includes('Reveal'),'rehearsal hides the card behind Reveal');
    await p.click('[data-act="reveal"]');await adv(p,200);ok((await p.textContent('#c-idea')).length>30,'Reveal shows the idea');
    /* rehearsal trains recall: captions hidden until Reveal, each object marked, the missed ones offered again */
    {const cap=await p.evaluate(()=>window.__palace.S.rehearse&&window.__palace.S.rehearse.list.length);let k=0;
      for(;k<cap;k++){if(k>0){await adv(p,1200);await p.click('[data-act="reveal"]');await adv(p,150);}await p.click(k===1?'[data-act="grade"][data-g="0"]':'[data-act="grade"][data-g="1"]');}
      await adv(p,300);const t=await p.textContent('#c-title'),a2=await p.textContent('#c-acts');ok(t.includes('You had '+(cap-1)+' of '+cap)&&a2.includes('The 1 I missed'),'rehearsal keeps score and offers the missed ones again ('+t+')');
      await p.click('[data-act="missed"]');await adv(p,300);ok(await p.evaluate(()=>window.__palace.S.rehearse.list.length===1),'"the ones I missed" rehearses only those');await p.click('[data-act="stoprehearse"]');await adv(p,300);}
    await p.click('#btn-list');ok(await p.isVisible('#list'),'list view opens');
    const li=await p.$$eval('#list ol > li',x=>x.length);ok(li===136,'the list holds 136 loci ('+li+')');
    const also=await p.evaluate(()=>{const P=window.__palace;const want=Object.values(P.ROOMS).flatMap(r=>r.also.map(a=>a.id));const got=[...document.querySelectorAll('#list [data-also]')].map(e=>e.dataset.also);return{want:want.length,missing:want.filter(w=>!got.includes(w))};});
    ok(also.want===54&&also.missing.length===0,'every one of the 54 demoted ideas appears in the list '+also.missing.join(' '));
    const small=await p.evaluate(()=>[...document.querySelectorAll('#list *')].filter(e=>e.childNodes.length&&[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())&&parseFloat(getComputedStyle(e).fontSize)<12).length);
    ok(small===0,'no list text under 12 px');
    ok(p.errs.length===0,'no console errors in the controls test '+p.errs.slice(0,3).join(' | '));await p.close();}
  /* 3. deep links and resume */
  {const p=await page(b,{width:390,height:844});await p.goto(URL+'#27/3');await ready(p);
    ok(await p.evaluate(()=>{const P=window.__palace;const st=P.STOPS()[P.S.target];return st.ob&&st.ob.id==='27-3';}),'#27/3 opens object 3 of the slate room');
    await adv(p,500);await p.goto(URL.replace(/#.*/,'')+'?again');await ready(p);
    ok(await p.evaluate(()=>{const P=window.__palace;const st=P.STOPS()[P.S.target];return st.ob&&st.ob.id==='27-3';}),'a return visit resumes at the last stop');await p.close();}
  /* 4. reduced motion starts in the list */
  {const p=await page(b,{width:390,height:844},{reduce:true});await p.goto(URL);await ready(p);
    ok(await p.isVisible('#list')&&!(await p.isVisible('#app')),'prefers-reduced-motion starts in the list view');
    await p.click('#btn-3d');await p.click('#next');await adv(p,50);
    ok(await p.evaluate(()=>{const P=window.__palace;return P.S.arrived===P.S.target;}),'reduced motion cuts between stops');await p.close();}
  /* 6. deep links: every room id and every object id; nonsense falls back to the start */
  {const p=await page(b,{width:390,height:844});await p.goto(URL);await ready(p);
    const bad=await p.evaluate(()=>{const P=window.__palace,out=[];const st=P.STOPS();const set=h=>history.replaceState(null,'',h);
      for(const id of Object.keys(P.ROOMS)){set('#'+id);const i=P.stopFromHash();if(i<0||st[i].kind!=='room'||st[i].room!==id)out.push(id);}
      for(const ob of P.OBJS){const [r,k]=ob.id.split('-');for(const h of ['#'+r+'/'+k,'#'+r+'-'+k]){set(h);const i=P.stopFromHash();if(i<0||st[i].ob!==ob)out.push(h);}}
      set('#gate');if(P.stopFromHash()!==0)out.push('gate');set('#lookout');if(P.stopFromHash()!==st.length-1)out.push('lookout');
      for(const h of ['#99','#04/9','#<script>','#%E0%A4%A','#']){set(h);let i;try{i=P.stopFromHash();}catch(e){i='throws '+e.message;}if(i!==-1)out.push(h+' → '+i);}set('#gate');return out;});
    ok(bad.length===0,'deep links: 39 rooms, 136 loci in both spellings, gate, lookout; nonsense is refused'+(bad.length?': '+bad.join(' '):''));
    /* a typed hash walks there; a far one fades and still arrives */
    const want=await p.evaluate(()=>{location.hash='#33/2';return window.__palace.STOPS().findIndex(s=>s.ob&&s.ob.id==='33-2');});
    await p.waitForFunction(()=>document.getElementById('fade').classList.contains('on')||window.__palace.S.target!==0);await arrived(p,want);
    ok(await p.evaluate(w=>window.__palace.S.arrived===w,want),'changing the hash by hand goes to that stop');ok(p.errs.length===0,'no errors on deep links '+p.errs.slice(0,2).join(' | '));await p.close();}
  /* 7. context loss and restore; hidden page. These are about animation frames, so they run on real frames and wait on them. */
  {const p=await page(b,{width:390,height:844});await p.goto(URL+'#04/1');await ready(p,false);
    const frames=()=>window.__palace.S.stats.frames.length&&window.__palace.S.stats.frames.slice(-1)[0];
    await p.waitForFunction(()=>window.__palace.S.stats.frames.length>3);
    const ext=await p.evaluate(()=>{const gl=document.getElementById('stage').getContext('webgl');const e=gl.getExtension('WEBGL_lose_context');if(!e)return false;window.__lc=e;e.loseContext();return true;});
    if(!ext)console.log('skip context loss: no WEBGL_lose_context');else{
      await p.waitForFunction(()=>!window.__palace.glOK());const lost=true;await p.evaluate(()=>window.__lc.restoreContext());await p.waitForFunction(()=>window.__palace.glOK(),null,{timeout:15000}).catch(()=>{});
      const f0=await p.evaluate(frames);const drawing=await p.waitForFunction(f0=>window.__palace.S.stats.frames.slice(-1)[0]>f0,f0,{timeout:15000}).then(()=>true).catch(()=>false);
      const r={lost,back:await p.evaluate(()=>window.__palace.glOK()),drawing,pts:await p.evaluate(()=>window.__palace.stats().points)};
      ok(r.lost&&r.back&&r.drawing&&r.pts>500,'a lost WebGL context stops drawing, and drawing resumes when it is restored '+JSON.stringify(r));}
    const hid=await p.evaluate(async()=>{const P=window.__palace;const frame=()=>new Promise(r=>requestAnimationFrame(()=>r()));Object.defineProperty(document,'hidden',{configurable:true,get:()=>true});await frame();await frame();
      const a=P.S.stats.frames.slice(-1)[0],v=P.S.vnow;for(let k=0;k<12;k++)await frame();const b=P.S.stats.frames.slice(-1)[0],v2=P.S.vnow;
      delete document.hidden;let c=b;for(let k=0;k<200&&c===b;k++){await frame();c=P.S.stats.frames.slice(-1)[0];}return{paused:a===b&&v===v2,resumed:c>b,jump:P.S.vnow-v2};});
    ok(hid.paused&&hid.resumed&&hid.jump<700,'a hidden page draws nothing and its clock stands still; it resumes without a jump '+JSON.stringify(hid));
    ok(p.errs.length===0,'no errors on context loss '+p.errs.slice(0,2).join(' | '));await p.close();}
  /* 8. resize and turning the phone: same stop, object still in view */
  {const p=await page(b,{width:390,height:844});await p.goto(URL+'#05/3');await ready(p);await adv(p,300);
    const chk=()=>p.evaluate(()=>{const P=window.__palace,st=P.STOPS()[P.S.target];const o=P.objOnScreen(st.ob);return{id:st.ob&&st.ob.id,n:o.n,in:o.box[0]>=-1&&o.box[1]>=-1&&o.box[2]<=o.W+1&&o.box[3]<=o.H+1,cw:document.getElementById('stage').width,W:o.W,dpr:P.CAM.dpr};});
    const outs=[];for(const vp of [{width:844,height:390},{width:1280,height:800},{width:320,height:568},{width:390,height:844}]){await p.setViewportSize(vp);
      /* the page redraws its rail 120 ms after the last resize event: wait for the canvas to take the new size, then step the walk */
      await p.waitForFunction(w=>window.__palace.CAM.W===w&&document.getElementById('stage').width===Math.round(w*window.__palace.CAM.dpr),vp.width);await adv(p,300);const c=await chk();outs.push(vp.width+'x'+vp.height+':'+c.id+' n='+c.n+(c.in?'':' OUT')+(Math.abs(c.cw-c.W*c.dpr)>1?' canvas '+c.cw:''));}
    ok(outs.every(o=>o.includes('05-3')&&!o.includes('OUT')&&!o.includes('canvas')&&+o.split('n=')[1].split(' ')[0]>60),'turning and resizing keeps the stop and the object in view, canvas resized: '+outs.join(' | '));
    ok(await p.evaluate(()=>{const c=document.getElementById('card').getBoundingClientRect();return c.bottom<=innerHeight+1&&c.left>=0&&c.right<=innerWidth+1;}),'the card stays on screen after turning back');await p.close();}
  /* 9. touch: a swipe up walks on, a sideways drag looks and does not walk, the canvas leaves pinch-zoom to the browser */
  {const p=await page(b,{width:390,height:844},{touch:true});await p.goto(URL+'#04');await ready(p);await adv(p,500);
    const cdp=await p.context().newCDPSession(p);const touch=async(pts)=>{for(let i=0;i<pts.length;i++){await cdp.send('Input.dispatchTouchEvent',{type:i===0?'touchStart':'touchMove',touchPoints:[{x:pts[i][0],y:pts[i][1]}]});await adv(p,16);}await cdp.send('Input.dispatchTouchEvent',{type:'touchEnd',touchPoints:[]});};
    const t0=await p.evaluate(()=>window.__palace.S.target);const side=[];for(let k=0;k<=10;k++)side.push([300-k*18,300+k]);await touch(side);await adv(p,900);
    const t1=await p.evaluate(()=>window.__palace.S.target);ok(t1===t0,'a sideways drag looks around and does not walk ('+t0+' → '+t1+')');
    const up=[];for(let k=0;k<=10;k++)up.push([200,420-k*14]);await touch(up);await adv(p,1800);
    const t2=await p.evaluate(()=>Math.round(window.__palace.S.target));ok(t2===t0+1,'one swipe up walks one stop on ('+t0+' → '+t2+')');
    ok(await p.evaluate(()=>getComputedStyle(document.getElementById('stage')).touchAction==='pinch-zoom'&&!/user-scalable\s*=\s*no|maximum-scale/.test(document.querySelector('meta[name=viewport]').content)),'pinch-zoom is left to the browser');
    ok(await p.evaluate(()=>{const d=document.documentElement;return d.scrollHeight<=innerHeight+1&&getComputedStyle(d).overflow==='hidden';}),'the 3-D view has nothing to scroll, so it cannot trap a scroll');
    const small=await p.evaluate(()=>[...document.querySelectorAll('#app button, #app a[href]')].filter(e=>e.offsetParent!==null).map(e=>{const r=e.getBoundingClientRect();return{id:e.id||e.textContent.trim().slice(0,12),w:r.width,h:r.height};}).filter(r=>r.w<43.5||r.h<43.5));
    ok(small.length===0,'every control in the 3-D view is at least 44 px each way at 390 px '+JSON.stringify(small));
    await p.click('#btn-list');ok(await p.evaluate(()=>{const d=document.documentElement;window.scrollTo(0,4000);return d.scrollHeight>innerHeight*3&&window.scrollY>1000;}),'the list scrolls as a normal page');
    ok(await p.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'nothing scrolls sideways at 390 px');await p.close();}
  /* 10. keyboard: arrows walk; the map keeps focus on a room while it redraws; Enter goes; Escape closes and gives focus back */
  {const p=await page(b,{width:1280,height:800},{dpr:1});await p.goto(URL+'#03');await ready(p);await adv(p,300);
    const i0=await p.evaluate(()=>window.__palace.S.target);await p.keyboard.press('ArrowRight');ok(await p.evaluate(i=>window.__palace.S.target===i+1,i0),'ArrowRight walks one stop');
    await p.keyboard.press('m');await adv(p,400);await p.focus('#map-svg [data-room="08"]');await adv(p,1300);
    ok(await p.evaluate(()=>document.activeElement&&document.activeElement.getAttribute('data-room')==='08'),'a room on the map keeps keyboard focus across redraws');
    await p.keyboard.press('Enter');const want=await p.evaluate(()=>window.__palace.STOPS().findIndex(s=>s.kind==='room'&&s.room==='08'));await arrived(p,want);ok(await p.evaluate(w=>window.__palace.S.arrived===w,want),'Enter on a room walks there');
    await p.click('#btn-menu');await adv(p,350);await p.keyboard.press('Escape');await adv(p,400);ok(await p.evaluate(()=>document.activeElement.id==='btn-menu'&&document.getElementById('menu').hidden),'Escape closes a dialog and returns focus to its button');
    /* tab order in the 3-D view reaches every control and nothing hidden */
    const order=await p.evaluate(()=>[...document.querySelectorAll('#app button, #app a[href]')].filter(e=>e.offsetParent!==null&&!e.disabled).map(e=>e.id||e.textContent.trim()).join(', '));console.log('controls in tab order:',order);
    /* contrast of every piece of text on the card, the HUD and the list, against the ground colour */
    const lowC=await p.evaluate(()=>{const lum=c=>{const v=c.map(x=>{x/=255;return x<=.03928?x/12.92:Math.pow((x+.055)/1.055,2.4);});return .2126*v[0]+.7152*v[1]+.0722*v[2];};const rgb=s=>(s.match(/[\d.]+/g)||[]).slice(0,4).map(Number);
      const bgOf=e=>{for(;e;e=e.parentElement){const b=rgb(getComputedStyle(e).backgroundColor);if(b.length>=3&&(b.length<4||b[3]>.5))return b.slice(0,3);}return[10,17,28];};
      const scan=root=>[...document.querySelectorAll(root+' *')].filter(e=>e.offsetParent!==null&&[...e.childNodes].some(n=>n.nodeType===3&&n.textContent.trim())).map(e=>{const cs=getComputedStyle(e);const f=rgb(cs.color).slice(0,3),b=bgOf(e);const L1=lum(f),L2=lum(b);const cr=(Math.max(L1,L2)+.05)/(Math.min(L1,L2)+.05);return{t:e.textContent.trim().slice(0,24),cr:+cr.toFixed(2),px:parseFloat(cs.fontSize)};});
      let all=scan('#app');document.getElementById('btn-list').click();all=all.concat(scan('#list'));const min=Math.min(...all.map(a=>a.cr));return{n:all.length,min,bad:all.filter(a=>a.cr<4.5||a.px<12).slice(0,5)};});
    ok(lowC.bad.length===0,'contrast: '+lowC.n+' text elements, lowest ratio '+lowC.min+':1, none under 4.5:1, none under 12 px '+JSON.stringify(lowC.bad));
    const sem=await p.evaluate(()=>{const L=document.getElementById('list');return{h2:L.querySelectorAll('h2').length,h3:L.querySelectorAll('h3').length,li:L.querySelectorAll('ol > li').length,empty:[...L.querySelectorAll('ol > li')].filter(li=>!li.querySelector('.l-idea').textContent.trim()||!li.querySelector('a[href]')).length,why:[...L.querySelectorAll('.l-why')].filter(e=>e.textContent.trim().length>20).length,
      hiddenCanvas:document.getElementById('stage').getAttribute('aria-hidden')==='true',live:!!document.querySelector('#live[aria-live]'),focusable:[...L.querySelectorAll('a,button')].every(e=>e.tabIndex>=0)};});
    ok(sem.h2===8&&sem.h3===39&&sem.li===136&&sem.empty===0&&sem.why===39&&sem.hiddenCanvas&&sem.live&&sem.focusable,'the list is whole: 8 zones, 39 rooms each with its sentence, 136 loci each with an idea and a lesson link; all links reachable by Tab '+JSON.stringify(sem));
    ok(p.errs.length===0,'no errors in the keyboard test '+p.errs.slice(0,2).join(' | '));await p.close();}
  /* 11. storage that throws: the page still walks */
  {const ctx=await b.newContext({viewport:{width:390,height:844}});await ctx.addInitScript(()=>{const boom=()=>{throw new DOMException('denied','SecurityError');};Storage.prototype.getItem=boom;Storage.prototype.setItem=boom;Storage.prototype.removeItem=boom;});
    const p=await ctx.newPage();const errs=[];p.on('pageerror',e=>errs.push(e.message));await p.goto(URL);await ready(p);await adv(p,300);await p.click('#next');await arrived(p,1);
    ok(errs.length===0&&await p.evaluate(()=>window.__palace.S.arrived===1),'with localStorage throwing, the page loads and walks '+errs.slice(0,2).join(' | '));await p.click('#btn-menu');await p.click('[data-m="forget"]');await adv(p,300);ok(errs.length===0,'and Forget my progress does not throw');await ctx.close();}
  /* 12. no WebGL at all: the list, with a plain sentence saying why */
  {const b2=await chromium.launch({args:['--disable-3d-apis']});const ctx=await b2.newContext({viewport:{width:390,height:844}});const p=await ctx.newPage();const errs=[];p.on('pageerror',e=>errs.push(e.message));p.on('console',m=>{if(m.type()==='error')errs.push(m.text());});
    await p.goto(URL+'#04/2');await p.waitForFunction(()=>window.__palace&&!document.getElementById('list').hidden);
    const r=await p.evaluate(()=>({gl:!!document.createElement('canvas').getContext('webgl'),list:!document.getElementById('list').hidden,app:document.getElementById('app').hidden,note:!document.getElementById('nogl').hidden,btn:document.getElementById('btn-3d').hidden,li:document.querySelectorAll('#list ol > li').length,walk:document.querySelectorAll('#list [data-go]').length,
      atRoom:(()=>{const e=document.getElementById('l-04');const r=e.getBoundingClientRect();return r.top<innerHeight&&r.bottom>0;})()}));
    ok(!r.gl&&r.list&&r.app&&r.note&&r.btn&&r.li===136&&r.walk===0&&r.atRoom&&errs.length===0,'without WebGL: list shown at the linked room, the reason given, no 3-D button, no "Walk here", no errors '+JSON.stringify(r)+' '+errs.slice(0,2).join(' | '));await b2.close();}
  /* 13. the view never turns fast: heading rate and speed on four long legs, stepped at 60 frames a second on the walk's clock */
  {const p=await page(b,{width:390,height:844},{dpr:1});for(const from of ['00c/4','33/4','10/4','03/4']){await p.goto(URL+'#'+from);await p.reload();await ready(p);await adv(p,300);
    const r=await p.evaluate(()=>{const P=window.__palace,dt=1000/60;let last=null,max=0,maxV=0;const v0=P.S.vnow;P.step(1);
      for(let t=0;t<60000;t+=dt){P.advance(dt,{dt,render:false});const c=P.CAM;if(last&&Math.hypot(c.x-last.x,c.y-last.y,c.h-last.h)>1.5)last=null;/* the fade of a long leg moves the walker in one frame, behind a dark screen */
        if(last){let d=Math.abs(c.yaw-last.yaw);d=Math.min(d,2*Math.PI-d);max=Math.max(max,d/(dt/1000));maxV=Math.max(maxV,Math.hypot(c.x-last.x,c.y-last.y,c.h-last.h)/(dt/1000));}last={yaw:c.yaw,x:c.x,y:c.y,h:c.h};
        if(P.S.arrived===P.S.target&&t>500)return{rate:+max.toFixed(2),speed:+maxV.toFixed(1),secs:+((P.S.vnow-v0)/1000).toFixed(1)};}
      return{rate:max,speed:maxV,secs:60,timeout:true};});
    ok(r.rate<=2.2&&r.speed<=7.5&&r.secs<=9.5&&!r.timeout,'from '+from+' to the next room: heading turns at most '+r.rate+' rad/s, speed at most '+r.speed+' m/s, '+r.secs+' s');}await p.close();}
  /* 14. leg times measured, not estimated: the five longest legs by the estimate are walked by Next and timed on the walk's own clock */
  {const p=await page(b,{width:390,height:844},{dpr:1});await p.goto(URL);await ready(p);
    const legs=await p.evaluate(()=>{try{localStorage.clear();}catch(e){}return window.__palace.legTimes().map((l,k)=>({to:k+1,t:l.t,fade:l.fade})).sort((x,y)=>y.t-x.t);});
    const pick=legs.filter(l=>!l.fade).slice(0,3).concat(legs.filter(l=>l.fade).slice(0,2));const out=[];let bad=0;
    for(const l of pick){await p.evaluate(i=>window.__palace.goTo(i-1,{instant:true}),l.to);await arrived(p,l.to-1);await adv(p,150);
      const v0=await p.evaluate(()=>window.__palace.S.vnow);await p.click('#next');await arrived(p,l.to,90000);const m=(await p.evaluate(()=>window.__palace.S.vnow)-v0)/1000;
      out.push(l.to+': '+m.toFixed(1)+' s (estimate '+l.t.toFixed(1)+(l.fade?', fades':'')+')');if(m>8.6||Math.abs(m-l.t)>1)bad++;}
    ok(bad===0,'the longest legs, measured: none over 8.6 s, each within 1 s of its estimate: '+out.join('; '));await p.close();}
  /* 15. walls are solid to a tap: an object in the next room answers only where the line of sight passes through the door */
  {const p=await page(b,{width:1280,height:800},{dpr:1});await p.goto(URL);await ready(p);
    const r=await p.evaluate(()=>{const P=window.__palace,out={hidden:0,moved:[]},st=P.STOPS();for(let i=0;i<st.length;i++){if(st[i].kind!=='room')continue;P.goTo(i,{instant:true});P.advance(50,{render:false});const here=st[i].room,nb=P.NODES[here].doors;
        for(const ob of P.OBJS){if(!nb.includes(ob.room)||P.seenThroughDoor(here,ob))continue;const q=P.project(ob.pos[0],ob.pos[1],ob.pos[2]);if(!q||q.x<0||q.x>P.CAM.W||q.y<0||q.y>P.CAM.H)continue;out.hidden++;P.tapAt(q.x,q.y);
          if(P.S.target!==i){const t=st[P.S.target];if(t.ob===ob)out.moved.push(here+'>'+ob.id);P.goTo(i,{instant:true});}}}return out;});
    ok(r.hidden>20&&r.moved.length===0,'a tap on a wall never reaches the object behind it ('+r.hidden+' hidden objects tapped) '+r.moved.join(' '));await p.close();}
  /* 16. a damaged saved position (a fraction, a huge number, a string) does not stop the page */
  {const p=await page(b,{width:390,height:844},{dpr:1});await p.goto(URL);await ready(p,false);const got=[];
    for(const v of ['{"stop":1.5,"visited":["04","zz"]}','{"stop":1e9}','{"stop":"7"}','{"stop":-3,"visited":7}','not json']){await p.evaluate(v=>localStorage.setItem('palace.v1',v),v);await p.reload();
      got.push(await p.waitForFunction(()=>window.__palace,null,{timeout:10000}).then(()=>p.evaluate(()=>{const P=window.__palace;return P&&Number.isInteger(P.S.target)&&P.S.target>=0&&P.S.target<P.STOPS().length;})).catch(()=>false));}
    ok(got.every(Boolean)&&p.errs.length===0,'a damaged saved position is survived '+JSON.stringify(got)+' '+p.errs.slice(0,2).join(' | '));await p.close();}
  /* 5. frame times while walking, at each size, on real animation frames (a report, not a check) */
  for(const vp of [{width:390,height:844},{width:844,height:390},{width:1280,height:800}]){const p=await page(b,vp);await p.goto(URL+'#03');await ready(p,false);
    for(let k=0;k<3;k++){await p.click('#next');await p.waitForTimeout(600);}
    const st=await p.evaluate(()=>window.__palace.stats());console.log('frames',vp.width+'x'+vp.height,JSON.stringify(st));await p.close();}
  await b.close();console.log('run time',((Date.now()-T0)/1000).toFixed(1)+' s');console.log(fails?fails+' failed':'all passed');process.exit(fails?1:0);})();
