/* The model house: a walkable memory palace of the course, drawn with plain WebGL 1.
   Plan coordinates: x east, y north, metres; h height in metres (level × 4).
   World coordinates for GL: [x, h, -y].
   Drawn per frame, measured over the whole walk by tools/test_palace.js: at most 12,000
   points, 26,000 line vertices (lines are drawn twice on dense screens, one device pixel
   apart, to give them weight), about 500 filled triangles and 60 draw calls. Walls,
   floors and ceilings are opaque fills written to the depth buffer, so a wall hides what
   is behind it; lines and points are depth-tested against them. One formation is
   evaluated per frame (two during a cross-fade). Device pixel ratio is capped at 2 and
   drops to 1.5, then 1, when frames come more than 26 ms apart for about two seconds. */
(()=>{
'use strict';
const DATA=JSON.parse(document.getElementById('palace-data').textContent);
const PAL=DATA.palace, MOVES=DATA.moves, CARDS=DATA.cards;
const MF=window.MovesF, PO=window.PalaceObjects;
const {F,N,H,G,clamp,lerp,ease,sstep}=MF;
const TAU=Math.PI*2, STOREY=4, EYE=1.6;
const $=id=>document.getElementById(id);
const esc=t=>String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const escB=t=>esc(t).replace(/&lt;(\/?)b&gt;/g,'<$1b>');
const lerpA=(a,b,t)=>{let d=((b-a)%TAU+TAU*1.5)%TAU-Math.PI;return a+d*t;};

/* LINK BASE: lesson routes live in the reader; explorables and quizzes sit beside palace/ on the site and in the reader. */
const INREADER=/\/reader\/palace\/(index\.html)?$/.test(location.pathname);
const LINKS={lesson:m=>(INREADER?'../index.html':'../reader/index.html')+'#/'+m,explorable:f=>'../explorables/'+f,quiz:q=>'../quizzes/'+q+'.html'};
/* /LINK BASE */

/* ---------- palette: the Moves dark palette; one accent per zone ---------- */
const BG=[10,17,28];
const RGB=[[228,234,242],[126,162,255],[242,178,62],[74,90,118]];
MF.setRGB(RGB,'#98A6B8');
const ZONES=['garden','cellar','ground','foundations','first','second','yard','tower'];
const ZI=Object.fromEntries(ZONES.map((z,i)=>[z,i]));
const ACC={garden:[242,178,62],cellar:[214,140,52],ground:[126,162,255],foundations:[196,152,92],first:[150,182,255],second:[186,196,216],yard:[214,224,236],tower:[160,200,236]};
const DIM={garden:1,cellar:.85,ground:1,foundations:.7,first:.95,second:.9,yard:.95,tower:.95};
const FURNISHED=new Set(['garden','ground']);
const ZNAME=Object.fromEntries(PAL.floors.map(f=>[f.id,f.name]));
const ZREASON=Object.fromEntries(PAL.floors.map(f=>[f.id,f.reason]));

/* ---------- nodes: rooms and halls ---------- */
const NODES={};
PAL.rooms.forEach(r=>NODES[r.id]={...r,kind:'room'});
PAL.halls.forEach(h=>NODES[h.id]={...h,kind:'hall'});
const ROOMS=Object.fromEntries(PAL.rooms.map(r=>[r.id,r]));
const levelsOf=n=>n.id==='core'?[-2,-1,0,1,2]:n.id==='tower_stair'?[0,1,2,3,4]:[n.level];
const OUTDOOR=new Set(['garden_path','yard_path','outside_stair']);
function wallSpec(n){/* what walls a node shows: ceiling height (0 = none), wall height, dim */
  if(OUTDOOR.has(n.id))return{ceil:0,wall:0};
  if(n.floor==='garden')return{ceil:0,wall:.9};
  if(n.id==='32')return{ceil:0,wall:1.1,fill:true};
  if(n.id==='balcony')return{ceil:3.6,wall:3.6,open:'east',fill:true};
  const c={cellar:3.0,foundations:3.2,yard:3.0}[n.floor]||3.6;return{ceil:c,wall:c,fill:true};}
const lh=L=>L*STOREY;
const centre=r=>[(r[0]+r[2])/2,(r[1]+r[3])/2];

function shared(a,b){const [ax0,ay0,ax1,ay1]=a,[bx0,by0,bx1,by1]=b;
  if(ax1===bx0||bx1===ax0){const x=ax1===bx0?ax1:ax0,lo=Math.max(ay0,by0),hi=Math.min(ay1,by1);if(hi>lo)return{axis:'x',c:x,lo,hi};}
  if(ay1===by0||by1===ay0){const y=ay1===by0?ay1:ay0,lo=Math.max(ax0,bx0),hi=Math.min(ax1,bx1);if(hi>lo)return{axis:'y',c:y,lo,hi};}
  return null;}
const DOORW=.75; /* half width of a door */
/* every ground-floor room opens on the residual corridor by a wide bay, so the stream is in sight from inside the room */
const BAYW=2;const halfW=k=>k.split('|').includes('residual_corridor')&&!k.split('|').includes('core')&&!k.split('|').includes('balcony')?BAYW:DOORW;
const doorCache={};
function door(a,b){const k=a<b?a+'|'+b:b+'|'+a;let d=doorCache[k];
  if(!d){const A=NODES[a<b?a:b],B=NODES[a<b?b:a];const w=shared(A.rect,B.rect);if(!w)throw new Error('no wall '+k);
    const LA=levelsOf(A),LB=levelsOf(B);let L=LA.filter(l=>LB.includes(l));const level=L.length?L[0]:Math.min(...LA,...LB);
    const m=(w.lo+w.hi)/2;d={k,w,level,hw:halfW(k),pt:w.axis==='x'?[w.c,m]:[m,w.c],ids:[A.id,B.id]};doorCache[k]=d;}
  const A=NODES[a],ca=centre(A.rect);const n=d.w.axis==='x'?[ca[0]<d.w.c?1:-1,0]:[0,ca[1]<d.w.c?1:-1];
  return{...d,n};}
const ALLDOORS=[];{const seen=new Set();Object.values(NODES).forEach(n=>n.doors.forEach(b=>{const k=n.id<b?n.id+'|'+b:b+'|'+n.id;if(!seen.has(k)){seen.add(k);ALLDOORS.push(door(n.id,b));}}));}

/* ---------- objects ---------- */
function makeFormation(expr){return expr.split(/\s+then\s+/).map(e=>new Function('F','return F.'+e.trim())(F));}
const OBJS=[];const OBJ_BY_ID={};
PAL.rooms.forEach(r=>{r.objects.forEach((o,k)=>{const spec=PO.OBJ[o.id]||{};const furnished=FURNISHED.has(r.floor);
  let seq;if(!furnished)seq=[PO.P.marker()];else if(spec.make){const m=spec.make();seq=Array.isArray(m)?m:[m];}
  else if(o.formation.moves_formation)seq=makeFormation(o.formation.moves_formation);else seq=[PO.P.marker()];
  const ob={id:o.id,data:o,room:r.id,zone:r.floor,k,count:r.objects.length,seq,spec,furnished,kind:furnished?(spec.kind||'panel'):'marker',
    base:spec.base||(furnished?1.45:1.2),scale:furnished?.85:.6,assembled:0,assembling:false,seen:false,cache:null};
  OBJS.push(ob);OBJ_BY_ID[o.id]=ob;});});
const PROLOGUE={id:'prologue',k:0,count:1,data:{idea:PAL.prologue.what},room:'garden_path',zone:'garden',seq:PO.P.prologue(),spec:{},furnished:true,kind:'panel',base:1.45,scale:.85,assembled:0,seen:false};
const EPILOGUE={id:'epilogue',k:0,count:1,room:'32',zone:'tower',seq:[F.motifs()],spec:{},furnished:true,kind:'panel',base:3.2,scale:2.2,assembled:1,seen:false};

/* ---------- loci: objects stand clockwise from the entry door ---------- */
const INSET=1.25;
function perim(r){const x0=r[0]+INSET,y0=r[1]+INSET,x1=r[2]-INSET,y1=r[3]-INSET;
  /* clockwise from the north-west corner, north up: north edge east, east edge south, south edge west, west edge north */
  const E=[[[x0,y1],[x1,y1],[0,-1]],[[x1,y1],[x1,y0],[-1,0]],[[x1,y0],[x0,y0],[0,1]],[[x0,y0],[x0,y1],[1,0]]];
  const L=E.map(e=>Math.hypot(e[1][0]-e[0][0],e[1][1]-e[0][1]));const tot=L.reduce((a,b)=>a+b,0);
  const at=t=>{t=((t%tot)+tot)%tot;for(let j=0;j<4;j++){if(t<=L[j]){const f=t/L[j],e=E[j];return{p:[lerp(e[0][0],e[1][0],f),lerp(e[0][1],e[1][1],f)],n:e[2],edge:j,t};}t-=L[j];}return{p:E[0][0],n:E[0][2],edge:0,t:0};};
  const proj=p=>{let best=1e9,bt=0,acc=0;for(let j=0;j<4;j++){const e=E[j];const dx=e[1][0]-e[0][0],dy=e[1][1]-e[0][1];const f=clamp(((p[0]-e[0][0])*dx+(p[1]-e[0][1])*dy)/(L[j]*L[j]));
      const q=[e[0][0]+dx*f,e[0][1]+dy*f];const d=Math.hypot(q[0]-p[0],q[1]-p[1]);if(d<best){best=d;bt=acc+f*L[j];}acc+=L[j];}return bt;};
  return{at,proj,tot,L};}
function placeLoci(room,entryPt,level){const P=perim(room.rect),n=room.objects.length;const t0=P.proj(entryPt);
  const blocked=[];room.doors.forEach(b=>{const d=door(room.id,b);if(d.level===level||levelsOf(NODES[b]).includes(level))blocked.push([P.proj(d.pt),d.hw+1.3]);});
  let cum=0;for(let j=0;j<4;j++){blocked.push([cum,1.35]);cum+=P.L[j];}
  const step=.1,free=[];for(let t=0;t<P.tot;t+=step){const tt=t0+t;const ok=blocked.every(b=>{const d=Math.abs(((tt-b[0])%P.tot+P.tot*1.5)%P.tot-P.tot/2);return d>b[1];});if(ok)free.push(tt);}
  /* spread the loci as far apart as the free wall allows (farthest-point picks), then number them clockwise from the entry */
  const C=free.map(t=>P.at(t));const pick=[0];const dist=C.map(c=>Math.hypot(c.p[0]-C[0].p[0],c.p[1]-C[0].p[1]));
  while(pick.length<n&&pick.length<C.length){let b=0;for(let j=1;j<C.length;j++)if(dist[j]>dist[b])b=j;pick.push(b);for(let j=0;j<C.length;j++)dist[j]=Math.min(dist[j],Math.hypot(C[j].p[0]-C[b].p[0],C[j].p[1]-C[b].p[1]));}
  /* a few rounds of relaxation: move each pick to the free spot that is farthest from the others */
  for(let it=0;it<6;it++)for(let q=0;q<pick.length;q++){let best=pick[q],bd=-1;for(let j=0;j<C.length;j++){let m=1e9;for(let r=0;r<pick.length;r++)if(r!==q)m=Math.min(m,Math.hypot(C[j].p[0]-C[pick[r]].p[0],C[j].p[1]-C[pick[r]].p[1]));if(m>bd){bd=m;best=j;}}pick[q]=best;}
  pick.sort((a,b)=>a-b);const out=pick.map(j=>C[j]);while(out.length<n)out.push(C[C.length-1]);return out;}

/* ---------- the rail: waypoints, stops, arc length ---------- */
let VIEW={fw:1,fh:.5,ox:0,oy:.42,vfov:70,aspect:.46};
function stopDistance(ob){const e=(ob.kind==='marker'?.7:1.22)*ob.scale;const tv=Math.tan(VIEW.vfov*Math.PI/360),th=tv*VIEW.aspect;
  return Math.max(2.1,e/(tv*VIEW.fh),e/(th*VIEW.fw));}
let STOPS=[],WP=[],PATH=null;
const lookFov=()=>VIEW.aspect<1?36:30;/* a long lens aloft, so the far garden is not dwarfed by the near tower */
function buildRail(){STOPS=[];WP=[];
  const wp=(x,y,h,node,stop)=>{WP.push({x,y,h,node,stop:stop===undefined?-1:stop});};
  const addStop=(s,x,y,h,node)=>{s.i=STOPS.length;s.pos=[x,y,h];STOPS.push(s);wp(x,y,h,node,s.i);return s;};
  const inRoomClamp=(r,x,y,m=.6)=>[clamp(x,r[0]+m,r[2]-m),clamp(y,r[1]+m,r[3]-m)];
  const objStop=(ob,locus,node,rect,level,extra={})=>{const o=locus.p,n=locus.n;let D=stopDistance(ob);
      const lim=n[0]?(n[0]>0?rect[2]-o[0]:o[0]-rect[0]):(n[1]>0?rect[3]-o[1]:o[1]-rect[1]);let sc=1;if(D>lim-.6){sc=Math.max(.55,(lim-.6)/D);D=lim-.6;}
      ob.fit=sc;const cx=o[0]+n[0]*D,cy=o[1]+n[1]*D;const eye=ob.kind==='3d'?2.3:EYE;
      ob.pos=[o[0],o[1],lh(level)+ob.base];ob.n=n;
      return addStop({kind:'object',ob,room:extra.room||ob.room,yaw:Math.atan2(-n[1],-n[0]),look:[o[0],o[1],lh(level)+ob.base],...extra},cx,cy,lh(level)+eye,node);};
  /* prologue at the garden gate */
  {const gp=NODES.garden_path.rect;const locus={p:[101,-3.1],n:[0,1]};objStop(PROLOGUE,locus,'garden_path',gp,0,{kind:'prologue',room:'garden_path',zone:'garden'});}
  let prevNode='garden_path';
  PAL.route.forEach((st,k)=>{const room=ROOMS[st.room];const chain=k===0?['garden_path',st.room]:[PAL.route[k-1].room,...st.via,st.room];
    let cur=null;/* post-point of the last door crossed */
    for(let j=0;j<chain.length-1;j++){const A=chain[j],B=chain[j+1];const d=door(A,B);const L=d.level;
      const pre=[d.pt[0]-d.n[0]*.9,d.pt[1]-d.n[1]*.9],post=[d.pt[0]+d.n[0]*.9,d.pt[1]+d.n[1]*.9];
      const nodeA=NODES[A];
      if(j>0)hallPath(nodeA,cur,{p:pre,L},wp);else wp(pre[0],pre[1],lh(L)+EYE,A);
      wp(d.pt[0],d.pt[1],lh(L)+EYE,B);wp(post[0],post[1],lh(L)+EYE,B);cur={p:post,L,n:d.n};}
    const L=room.level,r=room.rect,c=centre(r);
    const ent=inRoomClamp(r,cur.p[0]+cur.n[0]*1.5,cur.p[1]+cur.n[1]*1.5,.4);/* 2.4 m inside the door: the rise to the overview is gentle */
    addStop({kind:'room',room:room.id,step:st,yaw:Math.atan2(c[1]-ent[1],c[0]-ent[0]),look:[c[0],c[1],lh(L)+.9]},ent[0],ent[1],lh(L)+(room.floor==='garden'?2.9:2.55),room.id);
    const loci=placeLoci(room,cur.p,L);
    room.objects.forEach((o,q)=>{const ob=OBJ_BY_ID[o.id];objStop(ob,loci[q],room.id,r,L);});
    prevNode=room.id;});
  /* the lookout: rise above the tower and look back over the whole house */
  {const r=ROOMS['32'].rect,c=centre(r);wp(c[0],c[1],lh(4)+EYE+.8,'32');wp(c[0],c[1],lh(4)+8,'sky');
    /* stand off to the south-west, high, and back away until the whole site fits the free part of the screen */
    const port=VIEW.aspect<1;const az=(port?20:34)*Math.PI/180,el=(port?36:31)*Math.PI/180;const look=port?[40,5,2]:[44,4,2];
    const corners=[];const box=(r,h0,h1)=>[[r[0],r[1]],[r[2],r[1]],[r[0],r[3]],[r[2],r[3]]].forEach(([x,y])=>{corners.push([x,y,h0]);corners.push([x,y,h1]);});
    box([-12,-16,0,-4],0,17.2);box([0,0,68,24],0,11.6);box([70,-4,106,20],0,1);box([0,-16,56,0],0,3);
    const posAt=D=>[look[0]-D*Math.cos(el)*Math.cos(az),look[1]-D*Math.cos(el)*Math.sin(az),look[2]+D*Math.sin(el)];
    const sv=[CAM.fovAdd,CAM.oyK];CAM.fovAdd=lookFov()-VIEW.vfov;CAM.oyK=1;let D=60,at=posAt(D);
    for(;D<420;D+=3){at=posAt(D);setCamera(at[0],at[1],at[2],az,-el);if(corners.every(q=>{const pr=project(q[0],q[1],q[2]);if(!pr)return false;const nx=pr.x/CAM.W*2-1,ny=1-pr.y/CAM.H*2;return Math.abs(nx-VIEW.ox)<=VIEW.fw&&Math.abs(ny-VIEW.oy)<=VIEW.fh;}))break;}
    CAM.fovAdd=sv[0];CAM.oyK=sv[1];
    addStop({kind:'lookout',room:'32',zone:'tower',yaw:az,look},at[0],at[1],at[2],'sky');
    EPILOGUE.pos=[c[0],c[1],lh(4)+EPILOGUE.base];const dx=at[0]-c[0],dy=at[1]-c[1],dl=Math.hypot(dx,dy);EPILOGUE.n=[dx/dl,dy/dl];}
  PATH=sampleRail();}

/* Paths inside halls. a: {p,L} entry post-point (already pushed), b: {p,L} exit pre-point (pushed here). */
function hallPath(n,a,b,wp){const r=n.rect,id=n.id;
  const H=(L)=>lh(L)+EYE;
  if(id==='core'){const C=[6,17],R=3.2;const ta=Math.atan2(a.p[1]-C[1],a.p[0]-C[0]),tb=Math.atan2(b.p[1]-C[1],b.p[0]-C[0]);const dL=b.L-a.L;
    const dir=dL>=0?1:-1;/* one spiral stair: counter-clockwise going up */
    let base=((dir*(tb-ta))%TAU+TAU)%TAU;let tot=base+TAU*(Math.abs(dL)-1);while(tot<Math.max(1,Math.abs(dL))*1.4*Math.PI)tot+=TAU;
    const n2=Math.ceil(tot/(TAU/28));for(let k=0;k<=n2;k++){const f=k/n2,th=ta+dir*tot*f;wp(C[0]+R*Math.cos(th),C[1]+R*Math.sin(th),lerp(H(a.L),H(b.L),sstep(0,1,f)*.0+f),id);}
    wp(b.p[0],b.p[1],H(b.L),id);return;}
  if(id==='garden_path'&&b.L<a.L){/* the garden steps: down under the balcony */
    wp(75.6,1,H(a.L),id);for(let k=1;k<=10;k++){const f=k/10;wp(lerp(75.6,69.2,f),1,lerp(H(a.L),H(b.L),f),id);}wp(b.p[0],b.p[1],H(b.L),id);return;}
  if(id==='outside_stair'){/* two flights down the south face */
    const y1=-1,y2=-3;wp(58.4,-.95,H(a.L),id);
    for(let k=0;k<=8;k++){const f=k/8;wp(lerp(58.8,66.4,f),y1,lerp(H(2),H(1),f),id);}
    wp(67.2,-1.5,H(1),id);wp(67.2,-2.5,H(1),id);
    for(let k=0;k<=8;k++){const f=k/8;wp(lerp(66.4,58.8,f),y2,lerp(H(1),H(0),f),id);}
    wp(57.8,-2.6,H(0),id);wp(b.p[0],b.p[1],H(b.L),id);return;}
  if(id==='tower_stair'){if(b.L===a.L){wp(b.p[0],b.p[1],H(b.L),id);return;}
    wp(-10.6,-9,H(a.L),id);wp(-10.6,-5.4,H(a.L),id);wp(-11.4,-5.2,H(a.L),id);
    for(let k=0;k<=8;k++){const f=k/8;wp(-11.4,lerp(-5.8,-14.2,f),lerp(H(a.L),H(b.L),f),id);}
    wp(-10.6,-14.8,H(b.L),id);wp(-10.6,-11,H(b.L),id);wp(b.p[0],b.p[1],H(b.L),id);return;}
  const w=r[2]-r[0],h=r[3]-r[1];
  if(w>h*2){const cy=(r[1]+r[3])/2;wp(a.p[0],cy,H(a.L),id);wp(b.p[0],cy,H(b.L),id);}
  else if(h>w*2){const cx=(r[0]+r[2])/2;wp(cx,a.p[1],H(a.L),id);wp(cx,b.p[1],H(b.L),id);}
  wp(b.p[0],b.p[1],H(b.L),id);}

/* Corner rounding (never at stops), then even samples every 0.2 m with node and heading. */
const DS=.2,TURN=1.25,VCAP=6;
function sampleRail(){const P=[];const W=WP;
  for(let i=0;i<W.length;i++){const w=W[i];if(i===0||i===W.length-1||w.stop>=0){P.push([w.x,w.y,w.h,w.node,w.stop]);continue;}
    const a=W[i-1],c=W[i+1];const la=Math.hypot(w.x-a.x,w.y-a.y,w.h-a.h),lc=Math.hypot(c.x-w.x,c.y-w.y,c.h-w.h);const r=Math.min(.7,la*.45,lc*.45);
    if(r<.05){P.push([w.x,w.y,w.h,w.node,-1]);continue;}
    const p0=[w.x+(a.x-w.x)*r/la,w.y+(a.y-w.y)*r/la,w.h+(a.h-w.h)*r/la],p2=[w.x+(c.x-w.x)*r/lc,w.y+(c.y-w.y)*r/lc,w.h+(c.h-w.h)*r/lc];
    for(let k=0;k<=4;k++){const t=k/4,u=1-t;P.push([u*u*p0[0]+2*u*t*w.x+t*t*p2[0],u*u*p0[1]+2*u*t*w.y+t*t*p2[1],u*u*p0[2]+2*u*t*w.h+t*t*p2[2],w.node,-1]);}}
  const X=[],Y=[],Hh=[],ND=[],S=[];const stopArc=new Float64Array(STOPS.length);let acc=0;
  const push=(p)=>{X.push(p[0]);Y.push(p[1]);Hh.push(p[2]);ND.push(p[3]);};
  let carry=0;push(P[0]);if(P[0][4]>=0)stopArc[P[0][4]]=0;
  for(let i=1;i<P.length;i++){const a=P[i-1],b=P[i];const L=Math.hypot(b[0]-a[0],b[1]-a[1],b[2]-a[2]);if(L<1e-6){if(b[4]>=0)stopArc[b[4]]=acc;continue;}
    let t=DS-carry;while(t<L){const f=t/L;push([lerp(a[0],b[0],f),lerp(a[1],b[1],f),lerp(a[2],b[2],f),f<.5?a[3]:b[3]]);t+=DS;}
    carry=L-(t-DS);acc+=L;if(b[4]>=0)stopArc[b[4]]=acc;}
  push(P[P.length-1]);const n=X.length;
  /* arc of sample k is k*DS except the tail; store arcs explicitly */
  const A=new Float64Array(n);for(let k=0;k<n;k++)A[k]=Math.min(acc,k*DS);A[n-1]=acc;
  /* heading: facing at stops, the way ahead between them */
  const YAW=new Float32Array(n),PIT=new Float32Array(n);
  const pos=k=>[X[k],Y[k],Hh[k]];
  let si=0;for(let k=0;k<n;k++){const s=A[k];while(si<STOPS.length-2&&stopArc[si+1]<=s)si++;
    const s0=STOPS[si],s1=STOPS[Math.min(STOPS.length-1,si+1)];const a0=stopArc[si],a1=stopArc[Math.min(STOPS.length-1,si+1)];const seg=Math.max(1e-6,a1-a0),d1=s-a0,d2=a1-s;
    const k2=Math.min(n-1,k+Math.round(2.6/DS));const dx=X[k2]-X[k],dy=Y[k2]-Y[k],dh=Hh[k2]-Hh[k];const hd=Math.hypot(dx,dy);
    let ay=hd>.05?Math.atan2(dy,dx):(k?YAW[k-1]:s0.yaw),ap=Math.atan2(dh,Math.max(hd,.4))*.8;
    const fy=st=>st.yaw,fp=(st,p)=>{const L=st.look;return Math.atan2(L[2]-p[2],Math.hypot(L[0]-p[0],L[1]-p[1])+1e-6);};
    let yaw,pit;const p=pos(k);
    if(seg<6){const t=sstep(0,1,d1/seg);yaw=lerpA(fy(s0),fy(s1),t);pit=lerp(fp(s0,p),fp(s1,p),t);}
    else{const wa=1-sstep(0,2.6,d1),wb=1-sstep(0,2.6,d2);yaw=lerpA(ay,fy(s0),wa);pit=lerp(ap,fp(s0,p),wa);yaw=lerpA(yaw,fy(s1),wb);pit=lerp(pit,fp(s1,p),wb);}
    YAW[k]=yaw;PIT[k]=clamp(pit,-.75,.6);}
  /* smooth the heading between stops (vector average over ±1 m), keeping stops exact */
  const R=Math.round(1/DS);const Y2=new Float32Array(n),P2=new Float32Array(n);
  for(let k=0;k<n;k++){let cx=0,cy=0,pp=0,w=0;for(let j=-R;j<=R;j++){const q=clamp(k+j,0,n-1);const ww=1-Math.abs(j)/(R+1);cx+=Math.cos(YAW[q])*ww;cy+=Math.sin(YAW[q])*ww;pp+=PIT[q]*ww;w+=ww;}Y2[k]=Math.atan2(cy,cx);P2[k]=pp/w;}
  for(let i=0;i<STOPS.length;i++){const k=Math.round(stopArc[i]/DS);for(let j=-1;j<=1;j++){const q=clamp(k+j,0,n-1);Y2[q]=STOPS[i].yaw;P2[q]=fpAt(STOPS[i],[X[q],Y[q],Hh[q]]);}}
  /* a speed limit along the rail, so the view never turns faster than TURN rad/s; eased both ways so the walker brakes before a corner */
  const VL=new Float32Array(n);for(let k=0;k<n;k++){const q=Math.min(n-1,k+1),q0=Math.max(0,k-1);let d=Math.abs(Y2[q]-Y2[q0]);d=Math.min(d,TAU-d);const dp=Math.abs(P2[q]-P2[q0]);const K=Math.max(d,dp)/((q-q0)*DS);VL[k]=Math.min(VCAP,TURN/Math.max(K,1e-3));}
  for(let k=n-2;k>=0;k--)VL[k]=Math.min(VL[k],Math.sqrt(VL[k+1]*VL[k+1]+2*2.4*DS));for(let k=1;k<n;k++)VL[k]=Math.min(VL[k],Math.sqrt(VL[k-1]*VL[k-1]+2*2.4*DS));
  return{X,Y,H:Hh,ND,A,YAW:Y2,PIT:P2,VL,n,len:acc,stopArc};}
function fpAt(st,p){const L=st.look;return clamp(Math.atan2(L[2]-p[2],Math.hypot(L[0]-p[0],L[1]-p[1])+1e-6),-.75,.6);}
function railAt(a){const P=PATH;const f=clamp(a,0,P.len)/DS;let k=Math.floor(f);if(k>=P.n-1)k=P.n-2;const t=clamp(f-k,0,1);
  const r={x:lerp(P.X[k],P.X[k+1],t),y:lerp(P.Y[k],P.Y[k+1],t),h:lerp(P.H[k],P.H[k+1],t),yaw:lerpA(P.YAW[k],P.YAW[k+1],t),pit:lerp(P.PIT[k],P.PIT[k+1],t),node:t<.5?P.ND[k]:P.ND[k+1]};
  /* at and near a stop, face exactly what the stop faces */
  const si=stopIndexAt(a),d=Math.abs(P.stopArc[si]-a),w=1-sstep(0,.8,d);if(w>0){const st=STOPS[si];r.yaw=lerpA(r.yaw,st.yaw,w);r.pit=lerp(r.pit,fpAt(st,[r.x,r.y,r.h]),w);}
  return r;}

/* ---------- static geometry: walls, floors, stairs, streams; grouped by cell for culling ---------- */
const CELLS=[];const CELL_BY={};
function cellFor(id,zone){if(!CELL_BY[id]){const c={id,zone,pts:[],lns:[],fl:[],fc:[],box:[1e9,1e9,1e9,-1e9,-1e9,-1e9]};CELLS.push(c);CELL_BY[id]=c;}return CELL_BY[id];}
const ZONE_OF_HALL={garden_path:'garden',loss_pit:'cellar',gradient_corridor:'cellar',core:'ground',residual_corridor:'ground',balcony:'ground',gallery1:'first',gallery2:'second',outside_stair:'second',yard_path:'yard',tower_stair:'tower'};
const zoneOf=id=>NODES[id]?(NODES[id].kind==='room'?NODES[id].floor:ZONE_OF_HALL[id]):'tower';
function grow(c,x,y,h){const b=c.box;b[0]=Math.min(b[0],x);b[1]=Math.min(b[1],y);b[2]=Math.min(b[2],h);b[3]=Math.max(b[3],x);b[4]=Math.max(b[4],y);b[5]=Math.max(b[5],h);}
function L3(c,x0,y0,h0,x1,y1,h1,col,a){const z=ZI[c.zone];c.lns.push(x0,h0,-y0,col[0]/255,col[1]/255,col[2]/255,a,z, x1,h1,-y1,col[0]/255,col[1]/255,col[2]/255,a,z);grow(c,x0,y0,h0);grow(c,x1,y1,h1);}
function P3(c,x,y,h,col,a,size){c.pts.push(x,h,-y,col[0]/255,col[1]/255,col[2]/255,a,size,ZI[c.zone]);grow(c,x,y,h);}
/* Fills: opaque surfaces a shade off the ground colour, written to the depth buffer, so a wall hides what is behind it.
   tint k: how far from the ground colour toward the zone's accent. Vertex: x,h,-y, r,g,b, 1, zone. */
const TINT={floor:.1,wall:.06,ceil:.035,earth:.012};
function tint(zone,k){const a=ACC[zone];return[BG[0]+(a[0]-BG[0])*k,BG[1]+(a[1]-BG[1])*k,BG[2]+(a[2]-BG[2])*k];}
function Q3(arr,zone,col,a,b,c2,d){const z=ZI[zone],r=col[0]/255,g=col[1]/255,bl=col[2]/255;[a,b,c2,a,c2,d].forEach(p=>arr.push(p[0],p[2],-p[1],r,g,bl,1,z));}
/* a rectangle minus holes, as rectangles */
function minus(rect,holes){let rs=[rect];(holes||[]).forEach(h=>{rs=rs.flatMap(r=>{if(h[0]>=r[2]||h[2]<=r[0]||h[1]>=r[3]||h[3]<=r[1])return[r];const o=[];const x0=Math.max(r[0],h[0]),x1=Math.min(r[2],h[2]);
    if(h[0]>r[0])o.push([r[0],r[1],h[0],r[3]]);if(h[2]<r[2])o.push([h[2],r[1],r[2],r[3]]);if(h[1]>r[1])o.push([x0,r[1],x1,h[1]]);if(h[3]<r[3])o.push([x0,h[3],x1,r[3]]);return o;});});return rs;}
const SLABS=[];/* every horizontal fill: {rect,h,holes}, for the test that the rail never passes through one */
function slab(arr,zone,col,rect,h,holes,note){minus(rect,holes).forEach(r=>Q3(arr,zone,col,[r[0],r[1],h],[r[2],r[1],h],[r[2],r[3],h],[r[0],r[3],h]));SLABS.push({rect,h,holes:holes||[],note});}
/* openings: the garden steps' well, and four grates in the ground-floor corridor over the gradient channel */
const WELL=[68,.2,75.6,1.8];
const GRATES=[18,30,42,54].map(x=>[x,11.35,x+1.8,12.65]);
const drawn={},WALLSEGS=[],JAMBS=new Set();
function wallLine(c,axis,cc,lo,hi,level,spec,col,dim){/* one wall edge minus door gaps and lines already drawn */
  const key=axis+':'+cc+':'+level;const done=drawn[key]||(drawn[key]=[]);
  let segs=[[lo,hi]];const cut=(a,b)=>{segs=segs.flatMap(([s,e])=>b<=s||a>=e?[[s,e]]:[...(a>s?[[s,a]]:[]),...(b<e?[[b,e]]:[])]);};
  done.forEach(([a,b])=>cut(a,b));
  const gaps=ALLDOORS.filter(d=>d.w.axis===axis&&d.w.c===cc&&(d.level===level||(levelsOf(NODES[d.ids[0]]).includes(level)&&levelsOf(NODES[d.ids[1]]).includes(level))));
  const top=segs.slice();gaps.forEach(d=>{const m=axis==='x'?d.pt[1]:d.pt[0];cut(m-d.hw,m+d.hw);});
  const H0=lh(level),wall=spec.wall,ceil=spec.ceil,lint=Math.min(2.3,wall);
  const P=(u,h)=>axis==='x'?[cc,u,h]:[u,cc,h];
  const fillH=spec.fill?Math.max(wall,ceil):0,wcol=fillH?tint(c.zone,TINT.wall):null;
  if(fillH){segs.forEach(([s,e])=>Q3(c.fl,c.zone,wcol,P(s,H0),P(e,H0),P(e,H0+fillH),P(s,H0+fillH)));
    if(fillH>lint)gaps.forEach(d=>{const m=axis==='x'?d.pt[1]:d.pt[0];const w=d.hw;if(m-w<lo-1e-6||m+w>hi+1e-6)return;if(done.some(([a,b])=>m>a&&m<b))return;Q3(c.fl,c.zone,wcol,P(m-w,H0+lint),P(m+w,H0+lint),P(m+w,H0+fillH),P(m-w,H0+fillH));});}
  segs.forEach(([s,e])=>{WALLSEGS.push({axis,c:cc,level,lo:s,hi:e,h:Math.max(wall,ceil)});const a=P(s,H0),b=P(e,H0);L3(c,a[0],a[1],a[2],b[0],b[1],b[2],col,.62*dim);
    if(wall&&!ceil){const a2=P(s,H0+wall),b2=P(e,H0+wall);L3(c,a2[0],a2[1],a2[2],b2[0],b2[1],b2[2],col,.5*dim);const a3=P(s,H0+wall*.5),b3=P(e,H0+wall*.5);L3(c,a3[0],a3[1],a3[2],b3[0],b3[1],b3[2],col,.28*dim);}
    /* faint uprights so a wall reads as a surface: every 1.5 m indoors, a fence post every 1.2 m in the garden */
    const step=ceil?1.5:1.2,vh=ceil||wall;if(vh)for(let u=Math.ceil((s+.3)/step)*step;u<e-.3;u+=step){const a4=P(u,H0),b4=P(u,H0+vh);L3(c,a4[0],a4[1],a4[2],b4[0],b4[1],b4[2],col,(ceil?.13:.4)*dim);}});
  if(ceil)top.forEach(([s,e])=>{const a=P(s,H0+ceil),b=P(e,H0+ceil);L3(c,a[0],a[1],a[2],b[0],b[1],b[2],col,.38*dim);});
  gaps.forEach(d=>{const m=axis==='x'?d.pt[1]:d.pt[0];const w=d.hw;if(m-w<lo-1e-6||m+w>hi+1e-6)return;const jk=d.k+':'+level;if(JAMBS.has(jk))return;JAMBS.add(jk);[m-w,m+w].forEach(u=>{const a=P(u,H0),b=P(u,H0+lint);L3(c,a[0],a[1],a[2],b[0],b[1],b[2],col,.6*dim);});
    if(wall>=2.3){const a=P(m-w,H0+lint),b=P(m+w,H0+lint);L3(c,a[0],a[1],a[2],b[0],b[1],b[2],col,.8*dim);}});
  done.push([lo,hi]);}
function buildStatic(){Object.keys(drawn).forEach(k=>delete drawn[k]);WALLSEGS.length=0;SLABS.length=0;JAMBS.clear();CELLS.length=0;Object.keys(CELL_BY).forEach(k=>delete CELL_BY[k]);
  const order=[...PAL.route.map(s=>s.room)];const nodesSorted=[...Object.values(NODES)].sort((a,b)=>ZI[zoneOf(a.id)]-ZI[zoneOf(b.id)]||order.indexOf(a.id)-order.indexOf(b.id));
  nodesSorted.forEach(n=>{const zone=zoneOf(n.id);const c=cellFor(n.id,zone);const col=ACC[zone],dim=DIM[zone];const spec=wallSpec(n);const r=n.rect;
    const LV=levelsOf(n),shaft=LV.length>1;
    LV.forEach((L,li)=>{const H0=lh(L);
      /* floor and ceiling fills; a stair shaft has a floor at its foot only, and the core a roof */
      if(!OUTDOOR.has(n.id)&&(!shaft||li===0)){const holes=n.id==='residual_corridor'?GRATES:null;slab(c.fl,zone,tint(zone,n.floor==='garden'||zone==='garden'?TINT.floor*.4:TINT.floor),r,H0,holes,n.id+' floor');}
      if(n.id==='garden_path')slab(c.fl,zone,tint(zone,TINT.floor*.4),r,H0,[WELL],'garden path');
      if(spec.ceil&&(!shaft||(n.id==='core'&&li===LV.length-1))){const holes=n.id==='gradient_corridor'?GRATES:null;slab(c.fc,zone,tint(zone,TINT.ceil),r,H0+spec.ceil,holes,n.id+' ceiling');}
      if(spec.wall||spec.ceil){
        const edges=[['y',r[1],r[0],r[2]],['y',r[3],r[0],r[2]],['x',r[0],r[1],r[3]],['x',r[2],r[1],r[3]]];
        edges.forEach(([ax,cc,lo,hi])=>{if(spec.open==='east'&&ax==='x'&&cc===r[2]){L3(c,cc,lo,H0+1.1,cc,hi,H0+1.1,col,.4*dim);for(let u=lo;u<=hi;u+=2)L3(c,cc,u,H0,cc,u,H0+1.1,col,.25*dim);return;}wallLine(c,ax,cc,lo,hi,L,spec,col,dim);});
        const vh=spec.ceil||spec.wall;[[r[0],r[1]],[r[2],r[1]],[r[0],r[3]],[r[2],r[3]]].forEach(([x,y])=>L3(c,x,y,H0,x,y,H0+vh,col,.45*dim));}
      /* floor: joints every 2 m indoors, so the floor shows its depth */
      if(!OUTDOOR.has(n.id)&&n.floor!=='garden'&&zone!=='garden'&&(!shaft||li===0)){for(let x=Math.ceil((r[0]+.5)/2)*2;x<r[2]-.4;x+=2)L3(c,x,r[1],H0+.005,x,r[3],H0+.005,col,.1*dim);for(let y=Math.ceil((r[1]+.5)/2)*2;y<r[3]-.4;y+=2)L3(c,r[0],y,H0+.005,r[2],y,H0+.005,col,.1*dim);}
      /* floor: a sparse grid of points */
      const gs=n.floor==='garden'?1.0:OUTDOOR.has(n.id)?1.8:1.5;const pa=n.floor==='garden'?.34:(OUTDOOR.has(n.id)?.22:.26);
      if(n.id!=='core'&&!(n.id==='tower_stair'))for(let x=r[0]+gs/2;x<r[2];x+=gs)for(let y=r[1]+gs/2;y<r[3];y+=gs){
        const jx=n.floor==='garden'?(H(x*31+y,3)-.5)*.6:0,jy=n.floor==='garden'?(H(x+y*17,5)-.5)*.6:0;P3(c,x+jx,y+jy,H0+.01,col,pa*dim,n.floor==='garden'?.05:.04);}
    });
    if(n.id==='core'){/* the spiral stair: treads every 12°, rails, from the footings to the second floor */
      const C=[6,17];for(let th=0,k=0;th<=TAU*4.05;th+=TAU/30,k++){const h=lh(-2)+th/TAU*STOREY;L3(c,C[0]+1.1*Math.cos(th),C[1]+1.1*Math.sin(th),h,C[0]+4.3*Math.cos(th),C[1]+4.3*Math.sin(th),h,col,.3);
        const th2=th+TAU/30,h2=lh(-2)+th2/TAU*STOREY;if(th2<=TAU*4.05)L3(c,C[0]+4.3*Math.cos(th),C[1]+4.3*Math.sin(th),h+.9,C[0]+4.3*Math.cos(th2),C[1]+4.3*Math.sin(th2),h2+.9,col,.35);}
      L3(c,C[0],C[1],lh(-2),C[0],C[1],lh(2)+3.6,col,.3);}
    if(n.id==='garden_path'){for(let k=0;k<=10;k++){const x=lerp(75.6,69.2,k/10),h=lerp(0,-4,k/10);L3(c,x,.2,h,x,1.8,h,col,.4);}L3(c,75.6,.2,0,69.2,.2,-4,col,.3);L3(c,75.6,1.8,0,69.2,1.8,-4,col,.3);}
    if(n.id==='outside_stair'){const fl=(x0,x1,y,h0,h1)=>{for(let k=0;k<=12;k++){const f=k/12,x=lerp(x0,x1,f),h=lerp(h0,h1,f);L3(c,x,y-.8,h,x,y+.8,h,col,.38);}L3(c,x0,y-.8,h0+1,x1,y-.8,h1+1,col,.3);};
      fl(58.8,66.4,-1,8,4);fl(66.4,58.8,-3,4,0);L3(c,66.6,-.2,4,67.8,-.2,4,col,.3);L3(c,67.8,-.2,4,67.8,-3.8,4,col,.3);L3(c,66.6,-3.8,4,67.8,-3.8,4,col,.3);}
    if(n.id==='tower_stair'){for(let L=0;L<4;L++){for(let k=0;k<=12;k++){const f=k/12,y=lerp(-5.8,-14.2,f),h=lh(L)+f*STOREY;L3(c,-11.9,y,h,-10.9,y,h,col,.35);}}}
  });
  /* a plinth under every object: a ring on the floor and a post */
  OBJS.forEach(ob=>{if(!ob.pos)return;const c=CELL_BY[ob.room];const col=ACC[ob.zone],dim=DIM[ob.zone];const f=ob.pos[2]-ob.base;
    for(let k=0;k<28;k++){const th=k/28*TAU;P3(c,ob.pos[0]+.3*Math.cos(th),ob.pos[1]+.3*Math.sin(th),f+.02,col,.4*dim,.03);}
    if(ob.kind!=='3d')L3(c,ob.pos[0],ob.pos[1],f,ob.pos[0],ob.pos[1],f+Math.max(.2,ob.base-.95*ob.scale),col,.35*dim);});
  CELLS.forEach(c=>{c.grid=c.pts.length/9;});/* floor grids and plinth rings come first in each cell: far cells and the view from aloft leave them out */
  furnish();
  /* the earth: one dark sheet under the garden and the yard, so the cellar cannot be seen from outside; the house and the steps' well are cut out */
  {const e=cellFor('earth','garden');const holes=[[0,0,68,24],NODES.garden_path.rect];Object.values(NODES).forEach(n=>{if(n.kind==='room'&&n.level===0&&n.floor!=='ground')holes.push(n.rect);});holes.push(NODES.tower_stair.rect);
    slab(e.fl,'garden',tint('garden',TINT.earth),[-70,-80,150,90],-.04,holes,'earth');
    const gp=CELL_BY.garden_path,wc=tint('cellar',TINT.wall);/* the well's two side walls and its foot */
    [WELL[1],WELL[3]].forEach(y=>Q3(gp.fl,'cellar',wc,[WELL[0],y,-4],[WELL[2],y,-4],[WELL[2],y,0],[WELL[0],y,0]));Q3(gp.fl,'cellar',wc,[WELL[2],WELL[1],-4],[WELL[2],WELL[3],-4],[WELL[2],WELL[3],0],[WELL[2],WELL[1],0]);
    slab(gp.fl,'cellar',tint('cellar',TINT.floor),[WELL[0],WELL[1],69.2,WELL[3]],-4,null,'well foot');
    /* grates: bars across each opening */
    const rc=CELL_BY.residual_corridor;GRATES.forEach(g=>{L3(rc,g[0],g[1],.01,g[2],g[1],.01,ACC.cellar,.7);L3(rc,g[0],g[3],.01,g[2],g[3],.01,ACC.cellar,.7);for(let x=g[0];x<=g[2]+1e-6;x+=.3)L3(rc,x,g[1],.01,x,g[3],.01,ACC.cellar,.5);});}
  /* the residual and gradient channels' banks, and a tower silhouette */
  const rc=CELL_BY.residual_corridor,gc=CELL_BY.gradient_corridor;
  [11.1,12.9].forEach(y=>{L3(rc,12,y,.02,60,y,.02,ACC.ground,.5);L3(gc,12,y,-3.98,60,y,-3.98,ACC.cellar,.45);});
  const tc=CELL_BY['15'];[[-12,-16],[0,-16],[0,-4],[-12,-4]].forEach(([x,y])=>L3(tc,x,y,0,x,y,lh(4)+1.1,ACC.tower,.22));
  /* stars over the open sky */
  const st=cellFor('stars','garden');for(let i=0;i<420;i++)P3(st,-60+H(i,901)*200,-70+H(i,903)*150,40+H(i,905)*70,RGB[0],.25+.5*H(i,907),.12+.2*H(i,909));
}

/* Room furniture that makes a room a place: stakes and string, trees, a pond, a portal, shelves, haze. */
function furnish(){const C=id=>CELL_BY[id];
  {const c=C('00a'),col=ACC.garden;const y=12,x0=96.5,x1=103.5;L3(c,x0,y,0,x0,y,1.1,col,.9);L3(c,x1,y,0,x1,y,1.1,col,.9);L3(c,x0,y,.95,x1,y,.95,RGB[0],.7);
    for(let k=0;k<70;k++){const x=lerp(x0+.4,x1-.4,H(k,931)),side=k%2?1:-1;P3(c,x,y+side*(.5+H(k,933)*1.3),.05+H(k,935)*.3,k%2?RGB[1]:RGB[2],.75,.06);}}
  {const c=C('00b'),col=ACC.garden;[[83,3],[93,3],[83,19],[93,19]].forEach(([x,y],j)=>{L3(c,x,y,0,x,y,1.9,col,.6);
    for(let k=0;k<70;k++){const u=H(k+j*97,941),v=H(k+j*97,943),r=.35+.55*Math.cbrt(H(k+j*97,945)),th=u*TAU,ph=Math.acos(2*v-1);P3(c,x+r*Math.sin(ph)*Math.cos(th),y+r*Math.sin(ph)*Math.sin(th),2.5+r*Math.cos(ph)*.8,col,.5,.05);}});}
  {const c=C('00c'),col=RGB[1];const cx=76,cy=11;for(let k=0;k<90;k++){const th=k/90*TAU;P3(c,cx+2.2*Math.cos(th),cy+3.6*Math.sin(th),.02,col,.55,.045);}
    for(let k=0;k<70;k++){const r=Math.sqrt(H(k,951)),th=H(k,953)*TAU;P3(c,cx+2*r*Math.cos(th),cy+3.3*r*Math.sin(th),.01,col,.25,.04);}}
  {const c=C('03'),col=ACC.ground;const x=.05,y=5;L3(c,x,y-1.6,0,x,y-1.6,2.6,col,.7);L3(c,x,y+1.6,0,x,y+1.6,2.6,col,.7);
    for(let k=0;k<16;k++){const a0=k/16*Math.PI,a1=(k+1)/16*Math.PI;L3(c,x,y-1.6*Math.cos(a0),2.6+1*Math.sin(a0),x,y-1.6*Math.cos(a1),2.6+1*Math.sin(a1),col,.7);}}
  {const c=C('07'),col=ACC.ground;[.7,1.3,1.9,2.5].forEach(h=>L3(c,50.6,23.55,h,59.4,23.55,h,col,.45));for(let x=51;x<=59;x+=2)L3(c,x,23.55,.2,x,23.55,2.7,col,.25);}
  /* a mark inlaid in the floor of each remaining ground-floor room, so the room is known from its threshold and from the lookout */
  const inlay=(id,P,a=.55,col=ACC.ground)=>{const c=C(id);for(let k=1;k<P.length;k++)L3(c,P[k-1][0],P[k-1][1],.015,P[k][0],P[k][1],.015,col,a);};
  const ring=(cx,cy,r,n=40)=>{const P=[];for(let k=0;k<=n;k++){const t=k/n*TAU;P.push([cx+r*Math.cos(t),cy+r*Math.sin(t)]);}return P;};
  /* 04 atrium: every point looks at every other: a ring of eight, all chords drawn */
  {const V=[];for(let k=0;k<8;k++){const t=k/8*TAU+.2;V.push([21+2.4*Math.cos(t),5+2.4*Math.sin(t)]);}inlay('04',ring(21,5,2.4),.6);for(let a=0;a<8;a++)for(let b=a+1;b<8;b++)inlay('04',[V[a],V[b]],.22);}
  /* 08 picture door: a picture cut into sixteen patches */
  {for(let k=0;k<=4;k++){inlay('08',[[33.6+k*1.2,2.6],[33.6+k*1.2,7.4]],k%4?.3:.6);inlay('08',[[33.6,2.6+k*1.2],[38.4,2.6+k*1.2]],k%4?.3:.6);}}
  /* 05 balcony room: thirty-two layers in a row, toward the balcony */
  {for(let k=0;k<32;k++){const x=44.5+k*.42;inlay('05',[[x,3.6],[x,6.4]],k%8===0?.6:.3);}inlay('05',[[44.5,3.6],[57.52,3.6]],.45);inlay('05',[[44.5,6.4],[57.52,6.4]],.45);}
  /* 18 satchel room: one closed bag, the same size however long the text */
  {inlay('18',[[44,17.6],[47,17.6],[47,20],[44,20],[44,17.6]],.6);const P=[];for(let k=0;k<=16;k++){const t=k/16*Math.PI;P.push([45.5-.9*Math.cos(t),20+.9*Math.sin(t)]);}inlay('18',P,.6);inlay('18',[[44,19.2],[47,19.2]],.3);}
  /* 27 slate room: a ruled slate */
  {for(let k=0;k<=8;k++){inlay('27',[[34.5+k*.5,17],[34.5+k*.5,21]],k%8?.25:.6);inlay('27',[[34.5,17+k*.5],[38.5,17+k*.5]],k%8?.25:.6);}}
  /* 33 blank-line room: a line of blanks to fill */
  {for(let k=0;k<8;k++)inlay('33',[[13.6+k*.88,19],[14.2+k*.88,19]],.7);}
  {const c=C('09'),col=RGB[0];for(let k=0;k<160;k++)P3(c,lerp(22.5,31.5,H(k,961)),lerp(14.5,23.5,H(k,963)),.1+.7*H(k,965),col,.12,.09);}
}

/* ---------- GL ---------- */
const cv=$('stage'),ov=$('overlay'),octx=ov.getContext('2d');
let gl=null,PROG={},BUF={},glOK=false;
const VS_P=`attribute vec3 p;attribute vec4 c;attribute float s;attribute float z;uniform mat4 vp;uniform float ppm,fn,ff,cap;uniform float lit[8];varying vec4 vc;
void main(){vec4 q=vp*vec4(p,1.);gl_Position=q;float w=max(q.w,.05);float sz=s*ppm/(3.*pow(w/3.,.8));gl_PointSize=clamp(sz,1.,cap);
float fog=smoothstep(fn,ff,w);int zi=int(z+.5);float L=1.;for(int k=0;k<8;k++){if(k==zi)L=lit[k];}vc=vec4(c.rgb,c.a*(1.-fog)*L*min(1.,sz*sz/2.));}`;
const FS_P=`precision mediump float;varying vec4 vc;void main(){vec2 d=gl_PointCoord-.5;float r=length(d)*2.;float a=smoothstep(1.,.55,r);if(a<.01)discard;gl_FragColor=vec4(vc.rgb*vc.a*a,vc.a*a);}`;
const VS_L=`attribute vec3 p;attribute vec4 c;attribute float z;uniform mat4 vp;uniform float fn,ff;uniform float lit[8];varying vec4 vc;
void main(){vec4 q=vp*vec4(p,1.);gl_Position=q;float fog=smoothstep(fn,ff,q.w);int zi=int(z+.5);float L=1.;for(int k=0;k<8;k++){if(k==zi)L=lit[k];}vc=vec4(c.rgb,c.a*(1.-fog)*L);}`;
const FS_L=`precision mediump float;varying vec4 vc;void main(){gl_FragColor=vec4(vc.rgb*vc.a,vc.a);}`;
/* fills: lit by the walker's own lamp (brightest near the camera), fading to the ground colour with distance */
const VS_F=`attribute vec3 p;attribute vec4 c;attribute float z;uniform mat4 vp;uniform float lit[8];varying vec3 vc;varying vec3 wp;varying float vl;
void main(){gl_Position=vp*vec4(p,1.);wp=p;vc=c.rgb;int zi=int(z+.5);float L=1.;for(int k=0;k<8;k++){if(k==zi)L=lit[k];}vl=L;}`;
const FS_F=`precision mediump float;varying vec3 vc;varying vec3 wp;varying float vl;uniform vec3 eye,bg;uniform float fn,ff,lamp;
void main(){float d=distance(wp,eye);float fog=smoothstep(fn,ff,d);float k=mix(1.,.55+.85/(1.+d*d*.012),lamp);gl_FragColor=vec4(bg+(vc-bg)*k*vl*(1.-fog),1.);}`;
function mkProg(vs,fs){const sh=(t,s)=>{const o=gl.createShader(t);gl.shaderSource(o,s);gl.compileShader(o);if(!gl.getShaderParameter(o,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(o));return o;};
  const p=gl.createProgram();gl.attachShader(p,sh(gl.VERTEX_SHADER,vs));gl.attachShader(p,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));
  const loc={};['p','c','s','z'].forEach(a=>loc[a]=gl.getAttribLocation(p,a));['vp','ppm','fn','ff','cap','lit','eye','bg','lamp'].forEach(u=>loc['u'+u]=gl.getUniformLocation(p,u));p.loc=loc;return p;}
const DYNP=12000,DYNL=24000;
const dynP=new Float32Array(DYNP*9),dynL=new Float32Array(DYNL*8);
let staticP=null,staticL=null;
function initGL(){try{gl=cv.getContext('webgl',{antialias:true,alpha:false,premultipliedAlpha:true,powerPreference:'default'})||cv.getContext('experimental-webgl');}catch(e){gl=null;}
  if(!gl)return false;
  try{PROG.p=mkProg(VS_P,FS_P);PROG.l=mkProg(VS_L,FS_L);PROG.f=mkProg(VS_F,FS_F);}catch(e){console.warn(e);return false;}
  uploadStatic();
  BUF.dp=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,BUF.dp);gl.bufferData(gl.ARRAY_BUFFER,dynP.byteLength,gl.DYNAMIC_DRAW);
  BUF.dl=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,BUF.dl);gl.bufferData(gl.ARRAY_BUFFER,dynL.byteLength,gl.DYNAMIC_DRAW);
  gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.polygonOffset(1,2);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
  return true;}
function uploadStatic(){let np=0,nl=0,nf=0;CELLS.forEach(c=>{c.p0=np;np+=c.pts.length/9;c.pn=c.pts.length/9;c.q0=c.p0+(c.grid||0);c.qn=c.pn-(c.grid||0);c.l0=nl;nl+=c.lns.length/8;c.ln=c.lns.length/8;c.f0=nf;c.fn=c.fl.length/8;nf+=c.fn;});CELLS.forEach(c=>{c.c0=nf;c.cn=c.fc.length/8;nf+=c.cn;});
  const staticF=new Float32Array(nf*8);CELLS.forEach(c=>{staticF.set(c.fl,c.f0*8);staticF.set(c.fc,c.c0*8);});BUF.sf=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,BUF.sf);gl.bufferData(gl.ARRAY_BUFFER,staticF,gl.STATIC_DRAW);
  staticP=new Float32Array(np*9);staticL=new Float32Array(nl*8);CELLS.forEach(c=>{staticP.set(c.pts,c.p0*9);staticL.set(c.lns,c.l0*8);});
  BUF.sp=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,BUF.sp);gl.bufferData(gl.ARRAY_BUFFER,staticP,gl.STATIC_DRAW);
  BUF.sl=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,BUF.sl);gl.bufferData(gl.ARRAY_BUFFER,staticL,gl.STATIC_DRAW);}
function bindP(buf){const L=PROG.p.loc;gl.bindBuffer(gl.ARRAY_BUFFER,buf);const st=36;
  gl.enableVertexAttribArray(L.p);gl.vertexAttribPointer(L.p,3,gl.FLOAT,false,st,0);gl.enableVertexAttribArray(L.c);gl.vertexAttribPointer(L.c,4,gl.FLOAT,false,st,12);
  gl.enableVertexAttribArray(L.s);gl.vertexAttribPointer(L.s,1,gl.FLOAT,false,st,28);gl.enableVertexAttribArray(L.z);gl.vertexAttribPointer(L.z,1,gl.FLOAT,false,st,32);}
function bindL(buf,prog){const L=(prog||PROG.l).loc;gl.bindBuffer(gl.ARRAY_BUFFER,buf);const st=32;
  gl.enableVertexAttribArray(L.p);gl.vertexAttribPointer(L.p,3,gl.FLOAT,false,st,0);gl.enableVertexAttribArray(L.c);gl.vertexAttribPointer(L.c,4,gl.FLOAT,false,st,12);
  gl.enableVertexAttribArray(L.z);gl.vertexAttribPointer(L.z,1,gl.FLOAT,false,st,28);}

/* ---------- camera ---------- */
const CAM={x:0,y:0,h:0,yaw:0,pit:0,vp:new Float32Array(16),ppm:1,W:1,H:1,dpr:1,fovAdd:0,oyK:1};
function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;}return o;}
function setCamera(x,y,h,yaw,pit){CAM.x=x;CAM.y=y;CAM.h=h;CAM.yaw=yaw;CAM.pit=pit;
  const e=[x,h,-y],f=[Math.cos(pit)*Math.cos(yaw),Math.sin(pit),-Math.cos(pit)*Math.sin(yaw)];
  let s=[f[1]*0-f[2]*1,f[2]*0-f[0]*0,f[0]*1-f[1]*0];/* f × up */const sl=Math.hypot(...s)||1;s=s.map(v=>v/sl);
  const u=[s[1]*f[2]-s[2]*f[1],s[2]*f[0]-s[0]*f[2],s[0]*f[1]-s[1]*f[0]];
  const V=new Float32Array([s[0],u[0],-f[0],0, s[1],u[1],-f[1],0, s[2],u[2],-f[2],0, -(s[0]*e[0]+s[1]*e[1]+s[2]*e[2]),-(u[0]*e[0]+u[1]*e[1]+u[2]*e[2]),(f[0]*e[0]+f[1]*e[1]+f[2]*e[2]),1]);
  const t=1/Math.tan((VIEW.vfov+CAM.fovAdd)*Math.PI/360),nr=.12,fr=700;const P=new Float32Array([t/VIEW.aspect,0,0,0, 0,t,0,0, 0,0,(fr+nr)/(nr-fr),-1, 0,0,2*fr*nr/(nr-fr),0]);
  /* shift the optical centre to the free part of the screen (away from the card) */
  P[8]-=VIEW.ox;P[9]-=VIEW.oy*CAM.oyK;
  CAM.vp=mul(P,V);CAM.ppm=CAM.H*CAM.dpr*t/2;}
function project(x,y,h){const m=CAM.vp,X=x,Y=h,Z=-y;const cx=m[0]*X+m[4]*Y+m[8]*Z+m[12],cy=m[1]*X+m[5]*Y+m[9]*Z+m[13],cw=m[3]*X+m[7]*Y+m[11]*Z+m[15];
  if(cw<.05)return null;return{x:(cx/cw*.5+.5)*CAM.W,y:(1-(cy/cw*.5+.5))*CAM.H,w:cw};}

/* ---------- formation recording: Moves' 2-D draw calls become 3-D segments ---------- */
const SINK={seg:null,pt:null,text:null};
const colCache=new Map();
function parseCol(s){let v=colCache.get(s);if(v)return v;const m=String(s).match(/rgba?\(([^)]+)\)/);if(m){const p=m[1].split(',').map(Number);v=[p[0],p[1],p[2],p.length>3?p[3]:1];}else if(s&&s[0]==='#'){const h=s.slice(1);v=[parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16),1];}else v=[200,200,200,1];if(colCache.size>4096)colCache.clear();colCache.set(s,v);return v;}
const REC=(()=>{const K=1000;let path=[],cur=null;const st={strokeStyle:'#fff',fillStyle:'#fff',lineWidth:1.5,globalAlpha:1,font:'12px a',textAlign:'center',dash:null};const stack=[];
  const emit=(pl,col,a,dash)=>{if(a<.01)return;for(let i=1;i<pl.length;i++){const p=pl[i-1],q=pl[i];
      if(dash&&dash.length){const L=Math.hypot(q[0]-p[0],q[1]-p[1]);const on=dash[0]/170*K,off=(dash[1]||dash[0])/170*K;let t=0;while(t<L){const t1=Math.min(L,t+on);SINK.seg(p[0]+(q[0]-p[0])*t/L,p[1]+(q[1]-p[1])*t/L,p[0]+(q[0]-p[0])*t1/L,p[1]+(q[1]-p[1])*t1/L,col,a);t=t1+off;}}
      else SINK.seg(p[0],p[1],q[0],q[1],col,a);}};
  const o={save(){stack.push({...st,dash:st.dash});},restore(){const s=stack.pop();if(s)Object.assign(st,s);},
    beginPath(){path=[];cur=null;},moveTo(x,y){cur=[[x,y]];path.push(cur);},lineTo(x,y){if(!cur){cur=[[x,y]];path.push(cur);}else cur.push([x,y]);},
    closePath(){if(cur&&cur.length)cur.push(cur[0].slice());},
    arc(x,y,r,a0,a1){if(cur&&cur.length===1)path.pop();const full=Math.abs(a1-a0)>=TAU-1e-3;const n=Math.max(8,Math.ceil(Math.abs(a1-a0)/TAU*32));cur=[];cur.circle=full?[x,y,r]:null;
      for(let k=0;k<=n;k++){const a=a0+(a1-a0)*k/n;cur.push([x+r*Math.cos(a),y+r*Math.sin(a)]);}path.push(cur);},
    rect(x,y,w,h){cur=[[x,y],[x+w,y],[x+w,y+h],[x,y+h],[x,y]];path.push(cur);cur=null;},roundRect(x,y,w,h){o.rect(x,y,w,h);},
    stroke(){const c=parseCol(st.strokeStyle);const a=c[3]*st.globalAlpha*clamp(st.lineWidth/1.6,.55,1.3);path.forEach(pl=>emit(pl,c,a,st.dash));},
    fill(){const c=parseCol(st.fillStyle);const a=c[3]*st.globalAlpha;path.forEach(pl=>{if(pl.circle&&pl.circle[2]<.05*K){SINK.pt(pl.circle[0],pl.circle[1],c,a,pl.circle[2]/K);}else emit(pl,c,a*.75,null);});},
    fillText(t,x,y){const m=/(\d+(?:\.\d+)?)px/.exec(st.font);const px=m?+m[1]:12;SINK.text(String(t),x,y,px/K,parseCol(st.fillStyle),st.globalAlpha,st.textAlign,/700/.test(st.font));},
    setLineDash(d){st.dash=d&&d.length?d.slice():null;}};
  ['strokeStyle','fillStyle','lineWidth','globalAlpha','font','textAlign'].forEach(k=>Object.defineProperty(o,k,{get(){return st[k];},set(v){st[k]=v;}}));
  o.textBaseline='middle';o.lineJoin='round';o.lineCap='round';return o;})();
MF.setCtx(REC);MF.V.cx=0;MF.V.cy=0;MF.V.R=1000;
/* the 3-D draw API for palace formations; coordinates in unit space */
const D3={line(x0,y0,z0,x1,y1,z1,c,a){D3.seg3(x0,y0,z0,x1,y1,z1,c<0?[152,166,184,1]:[...RGB[c],1],a);},
  poly(P,c,a){for(let i=1;i<P.length;i++)D3.line(P[i-1][0],P[i-1][1],P[i-1][2]||0,P[i][0],P[i][1],P[i][2]||0,c,a);},
  text(t,x,y,z,a,o={}){const c=o.c===undefined||o.c<0?[152,166,184,1]:[...RGB[o.c],1];D3.text3(String(t),x,y,z,(o.size||.06),c,a,o.align||'center');},
  moves(f,s,a){if(f.palace)f.draw(s,a,D3);else{SINK.seg=(x0,y0,x1,y1,c,al)=>D3.seg3(x0/1000,-y0/1000,0,x1/1000,-y1/1000,0,c,al);SINK.pt=(x,y,c,al,r)=>D3.pt3(x/1000,-y/1000,0,c,al,r);
    SINK.text=(t,x,y,sz,c,al,align)=>D3.text3(t,x/1000,-y/1000,0,sz,c,al,align);f.draw(s,a);}},
  seg3:null,pt3:null,text3:null};

/* ---------- object transforms and evaluation ---------- */
const o1={x:0,y:0,z:0,c:0,r:1,a:1};
function objFrame(ob,t){/* unit → plan world: origin, right, up, normal; 3-D objects turn slowly */
  const n=ob.n,sc=ob.scale*(ob.fit||1);const rx=-n[1],ry=n[0];/* right of a viewer facing -n */
  let yaw=0;if(ob.kind==='3d'&&!REDUCE)yaw=.22*Math.sin(t*.00025)+.35;
  const c=Math.cos(yaw),s=Math.sin(yaw);const R=[rx*c+n[0]*s,ry*c+n[1]*s],Nn=[n[0]*c-rx*s,n[1]*c-ry*s];
  return{o:ob.pos,R,N:Nn,sc};}
function toWorld(fr,ob,x,y,z,out){if(ob.spec&&ob.spec.floor){const hgt=ob.spec.floor.h(x,y);z=-y;y=hgt;}
  out[0]=fr.o[0]+(x*fr.R[0]+z*fr.N[0])*fr.sc;out[1]=fr.o[1]+(x*fr.R[1]+z*fr.N[1])*fr.sc;out[2]=fr.o[2]+y*fr.sc;return out;}
function evalSeq(ob,f,s,X,Y,Z,C,R,A){if(f.prep)f.prep(s);const dep=ob.spec&&ob.spec.depth;for(let i=0;i<N;i++){o1.x=0;o1.y=0;o1.z=0;o1.c=0;o1.r=1;o1.a=1;f.p(i,s,o1);
  X[i]=o1.x;Y[i]=o1.y;Z[i]=o1.z+(dep?dep(i,s,o1):0);C[i]=o1.c;R[i]=o1.r;A[i]=o1.a;}}
const PLAY=5.6,HOLD=1.8,XF=.9;
function phaseOf(ob,t){/* which formation, its progress, and the cross-fade weight to the next */
  const n=ob.seq.length,per=PLAY+HOLD,k=Math.floor(t/per)%n,lt=t%per;const s=Math.min(1,lt/PLAY);const xf=lt>per-XF?(lt-(per-XF))/XF:0;return{k,s,xf,next:(k+1)%n};}
const EA={X:new Float32Array(N),Y:new Float32Array(N),Z:new Float32Array(N),C:new Uint8Array(N),R:new Float32Array(N),A:new Float32Array(N)};
const EB={X:new Float32Array(N),Y:new Float32Array(N),Z:new Float32Array(N),C:new Uint8Array(N),R:new Float32Array(N),A:new Float32Array(N)};
function restState(ob){const f=ob.seq[ob.seq.length-1];const s=.85;evalSeq(ob,f,s,EA.X,EA.Y,EA.Z,EA.C,EA.R,EA.A);return s;}
function dustPos(ob,i,out){/* room dust: where an object's points float before it assembles */
  const r=NODES[ob.room].rect,L=ob.pos[2]-ob.base;const u=H(i+ob.k*977,411),v=H(i+ob.k*977,413),w=H(i+ob.k*977,415);
  const near=H(i,417)<.8;if(near){const rr=.25+.45*Math.cbrt(H(i,419)),th=u*TAU,ph=Math.acos(2*v-1);out[0]=ob.pos[0]+rr*Math.sin(ph)*Math.cos(th);out[1]=ob.pos[1]+rr*Math.sin(ph)*Math.sin(th);out[2]=ob.pos[2]-.15+rr*Math.cos(ph)*.8;}
  else{out[0]=lerp(r[0]+.6,r[2]-.6,u);out[1]=lerp(r[1]+.6,r[3]-.6,v);out[2]=L+.3+w*2.6;}return out;}

/* ---------- frame state ---------- */
let REDUCE=matchMedia('(prefers-reduced-motion: reduce)').matches;
const S={a:0,v:0,target:0,vmax:3,scrub:false,arrived:-1,focus:null,focusT0:0,look:{yaw:0,pit:0,dy:0,dp:0,drag:false},
  visited:new Set(),rehearse:null,timeScale:1,vnow:0,mode:'3d',last:performance.now(),lookoutT0:0,lit:new Float32Array(8).fill(1),fillLit:new Float32Array(8).fill(1),aloft:0,stats:{frames:[],js:[]},dprScale:1,slow:0};

function stopIndexAt(a){const sa=PATH.stopArc;let lo=0,hi=STOPS.length-1;while(lo<hi){const m=(lo+hi+1)>>1;if(sa[m]<=a)lo=m;else hi=m-1;}
  if(lo<STOPS.length-1&&Math.abs(sa[lo+1]-a)<Math.abs(a-sa[lo]))return lo+1;return lo;}
function arcOfT(T){const n=STOPS.length;T=clamp(T,0,n-1);const k=Math.floor(T),f=T-k;if(k>=n-1)return PATH.stopArc[n-1];return lerp(PATH.stopArc[k],PATH.stopArc[k+1],f);}
function currentT(){/* fractional stop index of the camera */const a=S.a,sa=PATH.stopArc;let k=0;while(k<STOPS.length-1&&sa[k+1]<=a)k++;if(k>=STOPS.length-1)return k;return k+(a-sa[k])/Math.max(1e-6,sa[k+1]-sa[k]);}
function goTo(i,opt={}){i=clamp(Math.round(i),0,STOPS.length-1);S.target=i;S.scrub=false;const d=Math.abs(PATH.stopArc[i]-S.a);
  S.vmax=d>24?VCAP:3.2;if(opt.instant||REDUCE){S.a=PATH.stopArc[i];S.v=0;}markDirty();}
/* a far jump (the map, the menu, a deep link) fades out and in, and walks only the last few metres */
let jumpT=0;
function jumpTo(i){i=clamp(Math.round(i),0,STOPS.length-1);const d=PATH.stopArc[i]-S.a;if(Math.abs(d)<45||REDUCE){goTo(i);return;}
  const f=$('fade');f.classList.add('on');S.target=stopIndexAt(S.a);S.scrub=false;clearTimeout(jumpT);
  jumpT=setTimeout(()=>{const prev=i>0?PATH.stopArc[i-1]:0,next=i<STOPS.length-1?PATH.stopArc[i+1]:PATH.len;S.a=d>0?Math.max(prev,PATH.stopArc[i]-5):Math.min(next,PATH.stopArc[i]+5);S.v=0;S.look.yaw=0;S.look.pit=0;goTo(i);f.classList.remove('on');},280);}
function step(dir){if(S.rehearse){rehearseStep(dir);return;}const cur=S.scrub?Math.round(S.target):S.target;goTo((Number.isInteger(cur)?cur:Math.round(currentT()))+dir);}

/* ---------- culling ---------- */
const ADJ={garden:['cellar','ground'],cellar:['garden','ground','foundations'],ground:['garden','cellar','foundations','first','yard'],foundations:['cellar','ground'],first:['ground','second'],second:['first','yard'],yard:['ground','second','tower'],tower:['yard']};
function visibleCells(zone,high){const out=[];const zs=new Set(high?ZONES:[zone,...ADJ[zone]]);const m=CAM.vp;
  for(const c of CELLS){if(c.id==='stars'||c.id==='earth'){continue;}if(!zs.has(c.zone))continue;const b=c.box;let allOut=[0,0,0,0,0];
    for(let k=0;k<8;k++){const x=k&1?b[3]:b[0],y=k&2?b[4]:b[1],h=k&4?b[5]:b[2];const X=x,Y=h,Z=-y;const cx=m[0]*X+m[4]*Y+m[8]*Z+m[12],cy=m[1]*X+m[5]*Y+m[9]*Z+m[13],cw=m[3]*X+m[7]*Y+m[11]*Z+m[15];
      if(cx<-cw)allOut[0]++;if(cx>cw)allOut[1]++;if(cy<-cw)allOut[2]++;if(cy>cw)allOut[3]++;if(cw<0)allOut[4]++;}
    if(allOut.some(v=>v===8))continue;const dx=Math.max(b[0]-CAM.x,0,CAM.x-b[3]),dy=Math.max(b[1]-CAM.y,0,CAM.y-b[4]);if(!high&&Math.hypot(dx,dy)>48)continue;out.push(c);}
  return out;}
/* which cells' fills to draw. On foot: everything in view, ceilings included, plus the earth. Aloft: no ceilings and no earth, so the house opens like a model. */
function fillCells(cells,zone,high,rec){const e=CELL_BY.earth;if(!high)return{fl:e?cells.concat([e]):cells,fc:cells};
  if(rec&&rec.k<ZONES.length){const z=ZONES[rec.k];const fl=cells.filter(c=>c.zone===z);if(z!=='cellar'&&z!=='foundations'&&e)fl.push(e);return{fl,fc:[]};}
  return{fl:e?cells.concat([e]):cells,fc:[]};}
function ranges(cells,k0,kn){const r=[];cells.slice().sort((a,b)=>a[k0]-b[k0]).forEach(c=>{if(!c[kn])return;const last=r[r.length-1];if(last&&last[0]+last[1]===c[k0])last[1]+=c[kn];else r.push([c[k0],c[kn]]);});return r;}

/* ---------- per-frame dynamic geometry ---------- */
let np=0,nl=0,texts=[];
function pushP(x,y,h,r,g,b,a,size,z){if(np>=DYNP||a<.004)return;const o=np*9;dynP[o]=x;dynP[o+1]=h;dynP[o+2]=-y;dynP[o+3]=r;dynP[o+4]=g;dynP[o+5]=b;dynP[o+6]=a;dynP[o+7]=size;dynP[o+8]=z;np++;}
function pushL(x0,y0,h0,x1,y1,h1,r,g,b,a,z){if(nl>=DYNL-1||a<.004)return;let o=nl*8;dynL.set([x0,h0,-y0,r,g,b,a,z,x1,h1,-y1,r,g,b,a,z],o);nl+=2;}
const tmpA=[0,0,0],tmpB=[0,0,0];
function drawObject(ob,t,mode){/* mode: 'focus' (animated), 'rest' (assembled, still), 'dust' (unassembled), with a point stride for far rooms */
  const z=ZI[ob.zone];const fr=objFrame(ob,t);const stride=mode.stride||1;const bright=mode.bright===undefined?1:mode.bright;
  if(ob.kind==='marker'){const f=ob.seq[0];const s=REDUCE?0:(t*.00006)%1;if(f.prep)f.prep(s);for(let i=0;i<48;i+=stride){o1.x=0;o1.y=0;o1.z=0;o1.c=3;o1.r=1;o1.a=1;f.p(i,s,o1);toWorld(fr,ob,o1.x,o1.y,o1.z,tmpA);const c=RGB[0];pushP(tmpA[0],tmpA[1],tmpA[2],c[0]/255,c[1]/255,c[2]/255,.32*bright,.03,z);}
    toWorld(fr,ob,0,-.4,0,tmpA);toWorld(fr,ob,0,-1.8,0,tmpB);pushL(tmpA[0],tmpA[1],tmpA[2],tmpB[0],tmpB[1],tmpB[2],.6,.66,.75,.25*bright,z);return;}
  let A=ob.assembled;if(mode.kind==='dust')A=0;
  let s=0,xf=0,ph=null;
  if(mode.kind==='focus'&&!REDUCE){const lt=(t-S.focusT0)/1000;ph=phaseOf(ob,Math.max(0,lt-1.2));s=ph.s;xf=ph.xf;const f=ob.seq[ph.k];evalSeq(ob,f,s,EA.X,EA.Y,EA.Z,EA.C,EA.R,EA.A);
    if(xf>0){const g=ob.seq[ph.next];evalSeq(ob,g,0,EB.X,EB.Y,EB.Z,EB.C,EB.R,EB.A);}}
  else if(A>0){if(!ob.cache){restState(ob);ob.cache={X:EA.X.slice(),Y:EA.Y.slice(),Z:EA.Z.slice(),C:EA.C.slice(),R:EA.R.slice(),A:EA.A.slice()};}
    const c=ob.cache;EA.X.set(c.X);EA.Y.set(c.Y);EA.Z.set(c.Z);EA.C.set(c.C);EA.R.set(c.R);EA.A.set(c.A);}
  const tt=REDUCE?0:t;const sc=ob.scale*(ob.fit||1);
  for(let i=0;i<N;i+=stride){let x=EA.X[i],y=EA.Y[i],zz=EA.Z[i],r=EA.R[i],a=EA.A[i],ci=EA.C[i];
    if(xf>0){const m=ease(xf);x=lerp(x,EB.X[i],m);y=lerp(y,EB.Y[i],m);zz=lerp(zz,EB.Z[i],m);r=lerp(r,EB.R[i],m);a=lerp(a,EB.A[i],m);if(m>.5)ci=EB.C[i];}
    const col=RGB[ci]||RGB[0];
    if(A<1){dustPos(ob,i,tmpB);tmpB[0]+=Math.sin(tt*.0004+i*1.7)*.12;tmpB[1]+=Math.cos(tt*.00033+i*2.3)*.12;tmpB[2]+=Math.sin(tt*.00027+i)*.08;}
    let wx,wy,wh;
    if(A>0){toWorld(fr,ob,x,y,zz,tmpA);if(tt){tmpA[0]+=Math.sin(tt*.0009+i*1.7)*.004;tmpA[2]+=Math.cos(tt*.0011+i*2.3)*.004;}}
    if(A<=0){wx=tmpB[0];wy=tmpB[1];wh=tmpB[2];a=.6;r=.85;}
    else if(A>=1){wx=tmpA[0];wy=tmpA[1];wh=tmpA[2];}
    else{const d=H(i,431)*.4,u=ease(clamp((A-d)/.6));wx=lerp(tmpB[0],tmpA[0],u);wy=lerp(tmpB[1],tmpA[1],u);wh=lerp(tmpB[2],tmpA[2],u)+Math.sin(u*Math.PI)*.35;a=lerp(.55,a,u);r=lerp(.7,r,u);}
    const dustCol=A<1&&(A<=0||H(i,431)*.4>A);const cc=dustCol?ACC[ob.zone]:col;
    pushP(wx,wy,wh,cc[0]/255,cc[1]/255,cc[2]/255,clamp(a)*bright*(dustCol?.7:1),dustCol?.034:Math.max(.012,r*sc*.026),z);}
  /* lines and text */
  if(A>=.95&&(mode.kind==='focus'||mode.kind==='rest')&&stride===1){const al=(mode.kind==='focus'?1:.55)*bright*sstep(.95,1,A);
    const ca=lineCacheFor(ob,mode.kind,ph,s,xf,t,fr);if(ca){for(let k=0;k<ca.length;k+=8){const pc=ca[k+6];pushL(ca[k],ca[k+1],ca[k+2],ca[k+3],ca[k+4],ca[k+5],((pc>>16)&255)/255,((pc>>8)&255)/255,(pc&255)/255,ca[k+7]*al,z);}}}
}
/* Lines: recorded each frame for the focused object, once for still ones. Stored as [x0,y0,h0,x1,y1,h1,packedColour,alpha]. */
function lineCacheFor(ob,kind,ph,s,xf,t,fr){
  if(kind==='rest'&&ob.lines&&ob.linesFit===ob.fit)return ob.lines;
  const out=[];const tx=[];
  D3.seg3=(x0,y0,z0,x1,y1,z1,c,a)=>{if(a<.01)return;toWorld(fr,ob,x0,y0,z0,tmpA);toWorld(fr,ob,x1,y1,z1,tmpB);out.push(tmpA[0],tmpA[1],tmpA[2],tmpB[0],tmpB[1],tmpB[2],(c[0]<<16)|(c[1]<<8)|c[2],a*(c[3]===undefined?1:c[3]));};
  D3.pt3=(x,y,z,c,a,r)=>{toWorld(fr,ob,x,y,z,tmpA);const sc=ob.scale*(ob.fit||1);pushP(tmpA[0],tmpA[1],tmpA[2],c[0]/255,c[1]/255,c[2]/255,a,Math.max(.012,r*sc*2.2),ZI[ob.zone]);};
  D3.text3=(str,x,y,z,size,c,a,align)=>{if(kind!=='focus')return;toWorld(fr,ob,x,y,z,tmpA);tx.push({t:str,p:tmpA.slice(),size:size*ob.scale*(ob.fit||1),c,a,align});};
  try{if(kind==='focus'&&ph){const f=ob.seq[ph.k];const w=1-ease(xf);if(f.draw&&w>.01)D3.moves(f,s,w);if(xf>0){const g=ob.seq[ph.next];if(g.draw)D3.moves(g,0,ease(xf));}}
    else{const f=ob.seq[ob.seq.length-1];if(f.prep)f.prep(.85);if(f.draw)D3.moves(f,.85,1);}}catch(e){console.warn(ob.id,e);}
  if(kind==='rest'){ob.lines=out;ob.linesFit=ob.fit;}
  if(kind==='focus')texts=tx;
  return out;}

/* stream points: the residual stream runs east on the ground floor, the gradient west in the cellar */
function streams(t,zones){const tt=REDUCE?0:t*.001;
  if(zones.has('ground')){const c=ACC.ground;for(let i=0;i<360;i++){const x=12+((H(i,501)*48+tt*1.4*(.7+.6*H(i,503)))%48);const y=11.25+H(i,505)*1.5;pushP(x,y,.06+.04*Math.sin(tt*2+i),c[0]/255,c[1]/255,c[2]/255,.55,.035,ZI.ground);}}
  if(zones.has('cellar')){const c=ACC.cellar;for(let i=0;i<300;i++){const x=60-((H(i,511)*48+tt*1.1*(.7+.6*H(i,513)))%48);const y=11.25+H(i,515)*1.5;pushP(x,y,-3.94+.04*Math.sin(tt*2+i),c[0]/255,c[1]/255,c[2]/255,.5,.035,ZI.cellar);}}}

/* ---------- main frame ---------- */
let dirty=true;function markDirty(){dirty=true;}
let FT=0;
function frame(now){requestAnimationFrame(frame);if(document.hidden||S.mode!=='3d'||!glOK)return;
  const t0=performance.now();const dt=Math.min(.05,(now-S.last)/1000)*S.timeScale;S.last=now;const realNow=now;S.vnow+=dt*1000;now=S.vnow;
  /* walk controller */
  const tgtArc=S.scrub?arcOfT(S.target):PATH.stopArc[S.target];const diff=tgtArc-S.a;
  if(Math.abs(diff)<.002&&Math.abs(S.v)<.05){S.a=tgtArc;S.v=0;}else{const acc=S.scrub?9:3.2;const vmax=Math.min(S.scrub?Math.max(8,S.vmax):S.vmax,Math.max(.6,PATH.VL[clamp(Math.round(S.a/DS),0,PATH.n-1)]));const want=Math.sign(diff)*Math.min(vmax,Math.sqrt(2*acc*Math.abs(diff)));
    S.v+=clamp(want-S.v,-acc*dt*1.6,acc*dt*1.6);let na=S.a+S.v*dt;if((tgtArc-na)*diff<0){na=tgtArc;S.v=0;}S.a=na;dirty=true;}
  /* look-around springs back */
  const L=S.look;if(!L.drag){L.yaw*=Math.pow(.02,dt);L.pit*=Math.pow(.02,dt);if(Math.abs(L.yaw)<1e-4)L.yaw=0;if(Math.abs(L.pit)<1e-4)L.pit=0;}
  {const k=stopIndexAt(S.a),d=Math.abs(PATH.stopArc[k]-S.a);const want=STOPS[k].kind==='room'?(VIEW.aspect<1?16:10)*(1-sstep(.5,3,d)):STOPS[k].kind==='lookout'?(lookFov()-VIEW.vfov)*(1-sstep(2,60,d)):0;CAM.fovAdd=lerp(CAM.fovAdd,want,1-Math.pow(.05,dt));
    /* at a threshold the horizon sits lower on a phone, so the room and not its floor fills the free part of the screen */
    const wo=STOPS[k].kind==='room'&&VIEW.aspect<1?lerp(1,.72,1-sstep(.5,3,d)):1;CAM.oyK=lerp(CAM.oyK,wo,1-Math.pow(.05,dt));}
  const R=railAt(S.a);setCamera(R.x,R.y,R.h,R.yaw+L.yaw,clamp(R.pit+L.pit,-1.1,1.1));
  /* arrival */
  const si=stopIndexAt(S.a);const at=Math.abs(PATH.stopArc[si]-S.a)<.03&&Math.abs(S.v)<.08&&!S.scrub;
  if(at&&S.arrived!==si)arrive(si);if(!at&&S.arrived>=0&&Math.abs(PATH.stopArc[S.arrived]-S.a)>.3)S.arrived=-1;
  const near=STOPS[si];updateCard(si,at);
  const focusOb=near.ob&&Math.abs(PATH.stopArc[si]-S.a)<4?near.ob:(near.kind==='lookout'?null:null);
  if(focusOb!==S.focus){S.focus=focusOb;S.focusT0=now;texts=[];}
  /* assembly progress */
  OBJS.forEach(ob=>{if(ob.assembling){ob.assembled=Math.min(1,ob.assembled+dt/(REDUCE?.01:1.5));if(ob.assembled>=1)ob.assembling=false;}});
  [PROLOGUE,EPILOGUE].forEach(ob=>{if(ob.assembling){ob.assembled=Math.min(1,ob.assembled+dt/1.5);if(ob.assembled>=1)ob.assembling=false;}});
  /* zone of the camera, fog, lookout lights */
  const node=R.node;const high=R.h>22;S.aloft=sstep(20,30,R.h);const zone=node==='sky'?'tower':zoneOf(node);
  const outdoor=['garden','yard'].includes(zone)||high||node==='32';
  let fn=4,ff=outdoor?52:zone==='cellar'?26:zone==='foundations'?30:36;if(high){const u=sstep(20,30,R.h);fn=lerp(fn,200,u);ff=lerp(ff,600,u);}
  /* at the lookout the zones light one at a time, in walking order, then all together; a lit zone is drawn solid and the rest as ghosts */
  let rec=null;
  if(STOPS[si].kind==='lookout'&&at){const per=REC_PER,tt=(now-S.lookoutT0)/1000,k=Math.floor(tt/per)%(ZONES.length+1);rec={k,t:tt%per};
    for(let z=0;z<8;z++){const want=k>=ZONES.length?1:(z===k?1.5:.2);S.lit[z]=lerp(S.lit[z],want,1-Math.pow(.02,dt));}recite(k,rec.t>=REC_SAY);}
  else for(let z=0;z<8;z++)S.lit[z]=lerp(S.lit[z],1,1-Math.pow(.02,dt));
  S.rec=rec;if(!!rec!==S.wasRec){S.wasRec=!!rec;$('app').classList.toggle('aloft',!!rec);}/* the mini-map steps aside for the view */
  /* dynamic geometry */
  np=0;nl=0;
  const cells=visibleCells(zone,high);const zset=new Set(cells.map(c=>c.zone));
  streams(now,zset);
  const curRoom=NODES[node]&&NODES[node].kind==='room'?node:(near.room||null);
  const ri=PAL.route.findIndex(s=>s.room===curRoom);const nearRooms=new Set([curRoom,ri>0?PAL.route[ri-1].room:null,ri>=0&&ri<38?PAL.route[ri+1].room:null]);
  const vis=new Set(cells.map(c=>c.id));
  OBJS.forEach(ob=>{if(!vis.has(ob.room)&&!high)return;const isCur=ob.room===curRoom;const isNear=nearRooms.has(ob.room);
    if(high){drawObject(ob,now,{kind:ob.assembled>0?'rest':'dust',stride:ob.kind==='marker'?6:6,bright:.9});return;}
    if(ob===S.focus&&ob.assembled>0)drawObject(ob,now,{kind:'focus'});
    else if(isCur||isNear)drawObject(ob,now,{kind:ob.assembled>0?'rest':'dust',stride:isCur?1:4,bright:isCur?.6:.4});
    else drawObject(ob,now,{kind:ob.assembled>0?'rest':'dust',stride:8,bright:.3});});
  if(vis.has('garden_path')||zone==='garden'){const ob=PROLOGUE;drawObject(ob,now,{kind:S.focus===ob&&ob.assembled>0?'focus':(ob.assembled>0?'rest':'dust'),stride:S.focus===ob?1:3,bright:S.focus===ob?1:.5});}
  if(curRoom==='32'||high){const ob=EPILOGUE;if(ob.pos)drawObject(ob,now,{kind:'rest',stride:high?3:1,bright:high?.9:.7});}
  /* draw */
  const dpr=CAM.dpr;gl.viewport(0,0,cv.width,cv.height);gl.clearColor(BG[0]/255,BG[1]/255,BG[2]/255,1);gl.depthMask(true);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  let calls=0,pts=0,lns=0,tris=0;
  /* fills first: opaque, pushed back a little so the lines that lie on them win */
  {const F=PROG.f;gl.useProgram(F);gl.disable(gl.BLEND);gl.enable(gl.POLYGON_OFFSET_FILL);gl.uniformMatrix4fv(F.loc.uvp,false,CAM.vp);gl.uniform1f(F.loc.ufn,fn);gl.uniform1f(F.loc.uff,ff);gl.uniform1fv(F.loc.ulit,S.fillLit);
    gl.uniform3f(F.loc.ueye,CAM.x,CAM.h,-CAM.y);gl.uniform3f(F.loc.ubg,BG[0]/255,BG[1]/255,BG[2]/255);gl.uniform1f(F.loc.ulamp,1-S.aloft);bindL(BUF.sf,F);
    const fc=fillCells(cells,zone,high,rec);for(let z=0;z<8;z++)S.fillLit[z]=Math.min(1.15,S.lit[z]);ranges(fc.fl,'f0','fn').concat(ranges(fc.fc,'c0','cn')).forEach(([a,n])=>{gl.drawArrays(gl.TRIANGLES,a,n);calls++;tris+=n/3;});
    gl.disable(gl.POLYGON_OFFSET_FILL);gl.enable(gl.BLEND);gl.depthMask(false);}
  gl.useProgram(PROG.l);gl.uniformMatrix4fv(PROG.l.loc.uvp,false,CAM.vp);gl.uniform1f(PROG.l.loc.ufn,fn);gl.uniform1f(PROG.l.loc.uff,ff);gl.uniform1fv(PROG.l.loc.ulit,S.lit);
  if(nl){gl.bindBuffer(gl.ARRAY_BUFFER,BUF.dl);gl.bufferSubData(gl.ARRAY_BUFFER,0,dynL.subarray(0,nl*8));}
  /* a line is one device pixel wide; on a dense screen draw it twice, one pixel apart, so it keeps the weight of a Moves line */
  const lr=ranges(cells,'l0','ln');for(let pass=0;pass<(dpr>=1.5?2:1);pass++){if(pass)gl.viewport(1,0,cv.width,cv.height);
    bindL(BUF.sl);lr.forEach(([a,n])=>{gl.drawArrays(gl.LINES,a,n);calls++;lns+=n;});
    if(nl){bindL(BUF.dl);gl.drawArrays(gl.LINES,0,nl);calls++;lns+=nl;}}
  gl.viewport(0,0,cv.width,cv.height);
  gl.useProgram(PROG.p);gl.uniformMatrix4fv(PROG.p.loc.uvp,false,CAM.vp);gl.uniform1f(PROG.p.loc.uppm,CAM.ppm);gl.uniform1f(PROG.p.loc.ucap,26*dpr);gl.uniform1fv(PROG.p.loc.ulit,S.lit);
  gl.uniform1f(PROG.p.loc.ufn,fn);gl.uniform1f(PROG.p.loc.uff,ff);
  /* the point budget: floor-grid points are drawn only for cells within 22 m, and not at all from aloft */
  const nearC=high?[]:cells.filter(c=>{const b=c.box;return Math.hypot(Math.max(b[0]-CAM.x,0,CAM.x-b[3]),Math.max(b[1]-CAM.y,0,CAM.y-b[4]))<22;}),farC=high?cells:cells.filter(c=>!nearC.includes(c));
  bindP(BUF.sp);ranges(nearC,'p0','pn').concat(ranges(farC,'q0','qn')).forEach(([a,n])=>{gl.drawArrays(gl.POINTS,a,n);calls++;pts+=n;});
  if(outdoor){const st=CELL_BY.stars;gl.uniform1f(PROG.p.loc.ufn,1e4);gl.uniform1f(PROG.p.loc.uff,2e4);gl.drawArrays(gl.POINTS,st.p0,st.pn);calls++;pts+=st.pn;gl.uniform1f(PROG.p.loc.ufn,fn);gl.uniform1f(PROG.p.loc.uff,ff);}
  if(np){bindP(BUF.dp);gl.bufferSubData(gl.ARRAY_BUFFER,0,dynP.subarray(0,np*9));gl.drawArrays(gl.POINTS,0,np);calls++;pts+=np;}
  drawOverlay(cells,curRoom,si,at,high);
  updateHUD(si,R);
  const js=performance.now()-t0;const st=S.stats;st.js.push(js);if(st.js.length>240)st.js.shift();st.frames.push(realNow);if(st.frames.length>240)st.frames.shift();st.calls=calls;st.pts=pts;st.lns=lns;st.tris=tris;st.cells=cells.length;
  /* adaptive resolution: when frames come slower than 26 ms apart for about two seconds, draw at 3/4 of the pixels per side, then at 1/2. The interval between frames counts, not script time: a phone is slow in the GPU. */
  const gap=realNow-(S.lastReal||realNow);S.lastReal=realNow;if(gap>26&&gap<250)S.slow+=1;else S.slow=Math.max(0,S.slow-2);
  if(S.slow>110&&S.dprScale>.5){S.dprScale=S.dprScale===1?.75:.5;S.slow=0;resize();}
  dirty=false;}

/* ---------- overlay: room names and formation text ---------- */
const labelPool=[];const labelsEl=$('labels');
function drawOverlay(cells,curRoom,si,at,high){const W=CAM.W,Hh=CAM.H;octx.setTransform(CAM.dpr,0,0,CAM.dpr,0,0);octx.clearRect(0,0,W,Hh);
  /* room labels */
  let used=0;const want=[];
  const atObj=at&&STOPS[si].kind==='object';const myL=Math.round((CAM.h-EYE)/STOREY);
  const rec=S.rec,doorsOfCur=NODES[curRoom]?NODES[curRoom].doors:[];
  if(high){/* aloft: the lit zone's rooms once it is time to say them; with every zone lit, the zones' names */
    if(rec&&rec.k<ZONES.length){if(rec.t>=REC_SAY)PAL.rooms.forEach(n=>{if(n.floor!==ZONES[rec.k])return;const ce=centre(n.rect);const p=project(ce[0],ce[1],lh(n.level)+2.2);if(p)want.push({n,p,op:clamp((rec.t-REC_SAY)*2),d:-p.y,isCur:false,name:VIEW.aspect>=1?n.name.replace(/^The /,''):''});});}
    else if(rec)ZONES.forEach(z=>{const a=ZANCHOR[z];const p=project(a[0],a[1],a[2]);if(p)want.push({n:{id:ZSHORT[z],floor:z},p,op:1,d:0,isCur:false,name:''});});}
  else cells.forEach(c=>{const n=NODES[c.id];if(!n||n.kind!=='room')return;const isCur=c.id===curRoom;if(isCur&&STOPS[si].kind!=='room')return;if(atObj||n.level!==myL||(at&&STOPS[si].kind==='room'&&!isCur))return;
    if(!isCur&&!doorsOfCur.includes(n.id))return;/* walls are solid now: name only the rooms a door leads to */
    const ce=centre(n.rect);const sp=wallSpec(n);const p=project(ce[0],ce[1],lh(n.level)+(sp.ceil?sp.ceil-.4:2.4));if(!p)return;const d=Math.hypot(ce[0]-CAM.x,ce[1]-CAM.y,lh(n.level)-CAM.h);
    if(p.x<-80||p.x>W+80||p.y<-20||p.y>Hh+20)return;const op=clamp(1.25-d/30);if(op<.08)return;want.push({n,p,op,d,isCur,name:n.name.replace(/^The /,'')});});
  want.sort((a,b)=>a.d-b.d).slice(0,high?12:7).forEach(w=>{let el=labelPool[used];if(!el){el=document.createElement('div');el.className='rlabel';el.setAttribute('aria-hidden','true');el.innerHTML='<b></b><span></span>';labelsEl.appendChild(el);labelPool.push(el);}
    used++;el.firstChild.textContent=w.n.id;el.lastChild.textContent=w.name;el.style.transform='translate('+(w.p.x|0)+'px,'+(w.p.y|0)+'px) translate(-50%,-100%)';el.style.opacity=w.op.toFixed(2);el.style.display='';
    el.classList.toggle('cur',w.isCur);el.classList.toggle('aloft',high);el.style.setProperty('--zc','rgb('+ACC[w.n.floor].join(',')+')');});
  for(let k=used;k<labelPool.length;k++)labelPool[k].style.display='none';
  /* at a room's threshold, number its objects in walking order */
  if(STOPS[si].kind==='room'){const room=STOPS[si].room;octx.font="500 13px 'IBM Plex Mono', ui-monospace, Menlo, monospace";octx.textAlign='center';octx.textBaseline='middle';
    OBJS.forEach(ob=>{if(ob.room!==room||!ob.pos)return;const p=project(ob.pos[0],ob.pos[1],ob.pos[2]+.95*ob.scale+.25);if(!p)return;octx.globalAlpha=.95;octx.fillStyle='rgb('+ACC[ob.zone].join(',')+')';octx.fillText(String(ob.k+1),p.x,p.y);});octx.globalAlpha=1;}
  /* formation text for the focused object */
  if(!texts.length||!S.focus||S.focus.assembled<.95||(S.rehearse&&!S.rehearse.reveal&&!S.rehearse.done))return;const ob=S.focus;const fade=sstep(.95,1,ob.assembled);
  octx.textBaseline='middle';
  texts.forEach(tx=>{const p=project(tx.p[0],tx.p[1],tx.p[2]);if(!p)return;const px=Math.max(12,tx.size*CAM.ppm/CAM.dpr/p.w*1.0);if(px>34)return;
    octx.font='500 '+px.toFixed(1)+"px 'IBM Plex Mono', ui-monospace, Menlo, monospace";octx.textAlign=tx.align;octx.globalAlpha=clamp(tx.a*fade);
    const c=tx.c;octx.fillStyle='rgb('+c[0]+','+c[1]+','+c[2]+')';octx.fillText(tx.t,p.x,p.y);});
  octx.globalAlpha=1;}

/* ---------- arrival, cards, HUD ---------- */
function arrive(i){S.arrived=i;const st=STOPS[i];
  if(st.room&&ROOMS[st.room])S.visited.add(st.room);
  const ob=st.kind==='prologue'?PROLOGUE:st.ob;if(ob&&ob.assembled<1&&!ob.assembling){ob.assembling=true;}
  if(st.kind==='lookout'){S.lookoutT0=S.vnow;EPILOGUE.assembled=1;}
  save(i);setHash(st);announce(st);drawMap();}
let cardStop=-1,cardReveal=false;
const card=$('card'),cKick=$('c-kick'),cTitle=$('c-title'),cIdea=$('c-idea'),cImg=$('c-img'),cActs=$('c-acts');
function objTitle(ob){const d=ob.data;if(d.moves&&d.moves.length&&MOVES[d.moves[0]]&&MOVES[d.moves[0]].title)return MOVES[d.moves[0]].title;return (CARDS[ob.id]&&CARDS[ob.id].title)||'';}
function linkRow(L,extra=''){const ls=[];if(L.module)ls.push(['Lesson',LINKS.lesson(L.module)]);if(L.explorable)ls.push(['Explorable',LINKS.explorable(L.explorable)]);if(L.quiz)ls.push(['Quiz',LINKS.quiz(L.quiz)]);
  return extra+ls.map(([t,u])=>'<a class="chip" href="'+esc(u)+'">'+t+'</a>').join('');}
function updateCard(si,at){const R=S.rehearse;if(R&&R.done)return;const key=si+(R?'r'+R.reveal:'');if(key===cardStop&&!R)return;if(R&&key===cardStop)return;cardStop=key;
  const st=STOPS[si];let kick='',title='',idea='',img='',acts='',cls='';
  const zone=st.zone||(st.room&&ROOMS[st.room]?ROOMS[st.room].floor:'garden');
  if(st.kind==='prologue'){kick='<span>'+esc(ZNAME.garden)+'</span><span>Gate</span>';title='The model house';idea=esc(PAL.rule);img='Before the first bed, the garden gate: the first four moves of One Hundred Moves. Walk on, or skip the garden: it is optional.';
    acts='<button class="chip" type="button" data-act="skip">Skip the garden</button><button class="chip" type="button" data-act="list">List view</button>';}
  else if(st.kind==='room'){const r=ROOMS[st.room];const n=st.step.n;kick='<span>'+esc(ZSHORT[r.floor])+'</span><span>Room '+n+' of 39</span>';title=esc(r.name)+' <small>'+esc(r.id)+'</small>';idea=esc(st.step.why);
    img=(FURNISHED.has(r.floor)?'':'Furnished in a later step: markers stand where its '+r.objects.length+' objects will be. ')+esc(r.objects.length+' objects, clockwise from the door.');
    acts=linkRow(r.links)+'<button class="chip" type="button" data-act="rehearse">Rehearse</button>';cls='room';}
  else if(st.kind==='object'){const ob=st.ob,r=ROOMS[ob.room];kick='<span>'+esc(r.name.replace(/^The /,''))+' · '+esc(r.id)+'</span><span>'+(ob.k+1)+' / '+ob.count+'</span>';
    if(!ob.furnished){title='';idea=esc(ob.data.idea);img='';acts=linkRow(ob.data.links);cls='shell';}
    else if(R&&!R.reveal){title='What does this hold?';idea='Look at it, say the idea out loud, then reveal. Its captions are hidden until you do.';img='';acts='<button class="chip solid" type="button" data-act="reveal">Reveal</button><button class="chip" type="button" data-act="stoprehearse">Stop rehearsing</button>';cls='recall';}
    else{title=esc(objTitle(ob));idea=esc(ob.data.idea);img=esc(ob.data.image);acts='<button class="chip solid" type="button" data-act="deep" aria-haspopup="dialog">Deeper</button>'+linkRow(ob.data.links);
      if(R)acts='<button class="chip solid" type="button" data-act="grade" data-g="1">I had it</button><button class="chip" type="button" data-act="grade" data-g="0">I missed it</button>'+acts.replace(' solid','')+'<button class="chip" type="button" data-act="stoprehearse">Stop rehearsing</button>';
      else if(ob.k===ob.count-1)acts+='<button class="chip" type="button" data-act="rehearse">Rehearse this room</button>';}
    if(R)kick='<span>Rehearse · '+esc(r.name.replace(/^The /,''))+'</span><span>'+(ob.k+1)+' / '+ob.count+'</span>';}
  else if(st.kind==='lookout'){kick='<span>'+esc(ZNAME.tower)+'</span><span>Lookout</span>';title='The whole house';idea='<span id="recite"></span>';img='Each zone lights in turn, in the order you walked it. Say its rooms before their names appear.';lastRecite=-2;
    acts='<button class="chip" type="button" data-act="recite">Recite again</button><button class="chip" type="button" data-act="start">Walk again</button><button class="chip" type="button" data-act="epi" aria-haspopup="dialog">What came back</button>';}
  card.className='card z-'+zone+(cls?' '+cls:'');card.style.setProperty('--zc','rgb('+ACC[zone].join(',')+')');
  cKick.innerHTML=kick;cTitle.innerHTML=title;cTitle.hidden=!title;cIdea.innerHTML=idea;cImg.innerHTML=img;cImg.hidden=!img;cActs.innerHTML=acts;
  card.classList.remove('in');void card.offsetWidth;card.classList.add('in');
  $('prev').disabled=si===0&&!S.rehearse;$('next').disabled=si===STOPS.length-1&&!S.rehearse;}
const ZSHORT={garden:'Garden',cellar:'Cellar',ground:'Ground floor',foundations:'Footings',first:'First floor',second:'Second floor',yard:'Yard',tower:'Tower'};
const ZANCHOR={garden:[90,12,2],cellar:[72,-1,-2],ground:[40,19,4.5],foundations:[6,17,-7],first:[22,5,8.5],second:[50,5,12.5],yard:[30,-10,4],tower:[-5,-10,19]};
const ZTEXT={garden:'before the house',cellar:'weights being learned',ground:'weights used once, the forward pass',foundations:'how big to build',first:'weights changed after training',second:'weights frozen; only the words change',yard:'a score from the world',tower:'weights read, never changed'};
let lastRecite=-2;const REC_PER=6.5,REC_SAY=2.6;
function recite(k,say){const key=k*2+(say?1:0);if(key===lastRecite)return;lastRecite=key;const el=$('recite');if(!el)return;if(k>=ZONES.length){el.innerHTML='Eight zones, 39 rooms, one walk. To place a lesson, ask what happens to the weights there.';return;}const z=ZONES[k];
  const rooms=PAL.route.filter(s=>ROOMS[s.room].floor===z).map(s=>s.room+' '+ROOMS[s.room].name.replace(/^The /,''));
  el.innerHTML='<b>'+esc(ZNAME[z])+'</b>: '+esc(ZTEXT[z])+'. '+(say?esc(rooms.join(' · '))+'.':'<i>Say its '+(rooms.length===1?'room':rooms.length+' rooms')+'.</i>');}
const hudWhere=$('where');let hudKey='';
function updateHUD(si,R){const st=STOPS[si];let z,room,obj='';
  if(st.kind==='prologue'){z=ZSHORT.garden;room='Gate';}else if(st.kind==='lookout'){z=ZSHORT.tower;room='Above the lookout';}
  else{const r=ROOMS[st.room];z=ZSHORT[r.floor];room=r.id+' '+r.name.replace(/^The /,'');if(st.kind==='object')obj=(st.ob.k+1)+'/'+st.ob.count;}
  const key=z+room+obj;if(key===hudKey)return;hudKey=key;hudWhere.innerHTML='<span class="z">'+esc(z)+'</span><span class="r"><i>→</i> '+esc(room)+(obj?' <i>→</i> <b>'+esc(obj)+'</b>':'')+'</span>';
  hudWhere.setAttribute('aria-label','You are in: '+z+', '+room+(obj?', object '+obj:''));}
function announce(st){const el=$('live');let t='';if(st.kind==='room'){const r=ROOMS[st.room];t=r.name+', room '+st.step.n+' of 39. '+st.step.why;}else if(st.kind==='object'){t=ROOMS[st.ob.room].name+', object '+(st.ob.k+1)+' of '+st.ob.count+'. '+st.ob.data.idea;}
  else if(st.kind==='prologue')t='The garden gate. '+PAL.rule;else t='Above the lookout: the whole house.';el.textContent=t;}

/* ---------- deeper sheet ---------- */
const DD=$('deep'),dKick=$('deep-kick'),dTitle=$('deep-title'),dBody=$('deep-body');let opener=null;
function deeperHTML(d){let h='';if(d.text)h+='<p class="d-text">'+escB(d.text)+'</p>';if(d.formula)h+='<pre class="d-formula">'+esc(d.formula)+'</pre>';
  if(d.measured)h+='<p class="d-cap"><span class="lab">Measured</span>'+escB(d.measured)+'</p>';if(d.but)h+='<p class="d-cap d-but"><span class="lab">But</span>'+escB(d.but)+'</p>';
  if(d.sources&&d.sources.length)h+='<p class="d-src">'+esc(Array.isArray(d.sources)?d.sources.join('; '):d.sources)+'</p>';return h;}
function openDeep(ob,btn){const r=ROOMS[ob.room];dKick.innerHTML='<span>'+esc(r.name)+' · '+esc(r.id)+'</span><span>'+(ob.k+1)+' / '+ob.count+'</span>';dTitle.textContent=objTitle(ob);
  let h='<p class="d-idea">'+esc(ob.data.idea)+'</p><p class="d-img"><span class="lab">Picture</span>'+esc(ob.data.image)+'</p>';
  const mv=ob.data.moves||[];if(mv.length){mv.forEach((m,j)=>{const t=MOVES[m]||{};if(mv.length>1)h+='<h3 class="d-h">'+esc(t.title||m)+'</h3>';h+=t.deeper?deeperHTML(t.deeper):(t.body?'<p class="d-text">'+escB(t.body)+'</p>':'');});
    h+='<p class="d-src">From One Hundred Moves: '+mv.map(m=>esc(m)).join(', ')+'.</p>';}
  else if(CARDS[ob.id])h+=deeperHTML(CARDS[ob.id]);
  h+='<nav class="d-links" aria-label="Go further">'+linkRow(ob.data.links).replace(/class="chip"/g,'class="d-link"')+'</nav>';
  dBody.innerHTML=h;openDlg(DD,btn);}
function openEpilogue(btn){dKick.innerHTML='<span>'+esc(ZNAME.tower)+'</span><span>Lookout</span>';dTitle.textContent='What came back';
  let h='<p class="d-idea">'+esc(PAL.epilogue.what)+'</p>';PAL.epilogue.moves.forEach(m=>{const t=MOVES[m]||{};h+='<h3 class="d-h">'+esc(t.title||m)+'</h3>'+(t.deeper?deeperHTML(t.deeper):(t.body?'<p class="d-text">'+escB(t.body)+'</p>':''));});
  h+='<p class="d-src">From One Hundred Moves: '+PAL.epilogue.moves.map(m=>esc(m)).join(', ')+'.</p>';dBody.innerHTML=h;openDlg(DD,btn);}
let curDlg=null;
function openDlg(d,from){if(curDlg)closeDlg(false);curDlg=d;opener=from||document.activeElement;d.hidden=false;void d.offsetWidth;d.classList.add('open');$('app').inert=true;const f=[...d.querySelectorAll('button,a[href]')].find(x=>x.offsetParent!==null)||d;f.focus({preventScroll:true});}
function closeDlg(refocus=true){const d=curDlg;if(!d)return;curDlg=null;d.classList.remove('open');$('app').inert=false;const done=()=>{d.hidden=true;};if(REDUCE)done();else setTimeout(done,300);if(refocus&&opener&&opener.focus)opener.focus({preventScroll:true});opener=null;}
document.querySelectorAll('.dlg').forEach(d=>d.addEventListener('click',e=>{if(e.target.closest('[data-close]'))closeDlg();}));
document.addEventListener('keydown',e=>{if(!curDlg)return;if(e.key==='Escape'){e.preventDefault();closeDlg();return;}
  if(e.key!=='Tab')return;const f=[...curDlg.querySelectorAll('a[href],button:not([disabled]),[tabindex="0"]')].filter(x=>x.offsetParent!==null);if(!f.length)return;const first=f[0],last=f[f.length-1],ae=document.activeElement;
  if(e.shiftKey&&(ae===first||!curDlg.contains(ae))){e.preventDefault();last.focus();}else if(!e.shiftKey&&(ae===last||!curDlg.contains(ae))){e.preventDefault();first.focus();}});
{const hd=DD.querySelector('.handle'),sh=DD.querySelector('.sheet');let y0=null,dy=0;
  hd.addEventListener('pointerdown',e=>{y0=e.clientY;dy=0;hd.setPointerCapture(e.pointerId);sh.style.transition='none';});
  hd.addEventListener('pointermove',e=>{if(y0===null)return;dy=Math.max(0,e.clientY-y0);sh.style.transform='translateY('+dy+'px)';});
  const end=()=>{if(y0===null)return;y0=null;sh.style.transition='';sh.style.transform='';if(dy>70)closeDlg();};hd.addEventListener('pointerup',end);hd.addEventListener('pointercancel',end);}

/* ---------- card actions ---------- */
card.addEventListener('click',e=>{const b=e.target.closest('[data-act]');if(!b)return;const act=b.dataset.act;const st=STOPS[stopIndexAt(S.a)];
  if(act==='deep'&&st.ob)openDeep(st.ob,b);
  else if(act==='epi')openEpilogue(b);
  else if(act==='grade'&&S.rehearse){S.rehearse.marks[S.rehearse.idx]=+b.dataset.g;rehearseStep(1);}
  else if(act==='missed'&&S.rehearse){const R=S.rehearse;S.rehearse=null;startRehearse(R.room,R.missed);}
  else if(act==='rehearse'&&S.rehearse){const room=S.rehearse.room;S.rehearse=null;startRehearse(room);}
  else if(act==='skip'){const i=STOPS.findIndex(s=>s.kind==='room'&&s.room==='01');jumpTo(i);}
  else if(act==='list')setMode('list');
  else if(act==='rehearse')startRehearse(st.room||st.ob.room);
  else if(act==='reveal'){S.rehearse.reveal=true;cardStop=-1;}
  else if(act==='stoprehearse')stopRehearse(true);
  else if(act==='recite'){S.lookoutT0=S.vnow;lastRecite=-2;}
  else if(act==='start')jumpTo(0);});
$('prev').addEventListener('click',()=>step(-1));$('next').addEventListener('click',()=>step(1));

/* ---------- rehearsal ---------- */
function roomObjStops(room){return STOPS.filter(s=>s.kind==='object'&&s.ob.room===room).map(s=>s.i);}
function startRehearse(room,only){const all=roomObjStops(room);const list=only&&only.length?only:all;if(!list.length)return;S.rehearse={room,list,idx:0,reveal:false,marks:[],ret:stopIndexAt(S.a)};
  OBJS.forEach(ob=>{if(ob.room===room){ob.assembled=Math.max(ob.assembled,1);}});$('app').classList.add('rehearsing');cardStop=-1;goTo(list[0]);}
function rehearseStep(dir){const R=S.rehearse;const j=R.idx+dir;if(j<0)return;if(j>=R.list.length){finishRehearse();return;}R.idx=j;R.reveal=false;cardStop=-1;goTo(R.list[j]);}
function finishRehearse(){const R=S.rehearse;R.done=true;const r=ROOMS[R.room];const n=R.list.length,had=R.marks.filter(m=>m===1).length;R.missed=R.list.filter((_,j)=>R.marks[j]!==1);
  cKick.innerHTML='<span>Rehearse · '+esc(r.name.replace(/^The /,''))+'</span><span>done</span>';cTitle.hidden=false;cTitle.textContent=had===n?'All '+n+' recalled':'You had '+had+' of '+n;
  cIdea.textContent=had===n?'Walk it again tomorrow with the cards hidden, then every few days.':'Walk the ones you missed again now, then the whole room tomorrow.';cImg.hidden=true;
  cActs.innerHTML=(R.missed.length&&had?'<button class="chip solid" type="button" data-act="missed">The '+R.missed.length+' I missed</button>':'')+'<button class="chip'+(R.missed.length&&had?'':' solid')+'" type="button" data-act="rehearse">Whole room again</button><button class="chip" type="button" data-act="stoprehearse">Back to the walk</button>';cardStop='done';}
function stopRehearse(){const R=S.rehearse;if(!R)return;S.rehearse=null;$('app').classList.remove('rehearsing');cardStop=-1;const last=R.list[R.list.length-1];goTo(Math.min(STOPS.length-1,last));}

/* ---------- input: wheel and swipe walk, drag looks, tap goes ---------- */
let scrubTimer=0;
function scrubBy(dT){if(S.rehearse)return;if(!S.scrub){S.scrub=true;S.target=currentT();}S.target=clamp(S.target+dT,0,STOPS.length-1);S.vmax=8;clearTimeout(scrubTimer);
  scrubTimer=setTimeout(()=>{const dir=Math.sign(dT);const T=S.target;let k=dir>0?Math.ceil(T-.25):Math.floor(T+.25);S.scrub=false;goTo(k);},220);}
cv.addEventListener('wheel',e=>{e.preventDefault();if(curDlg)return;const d=e.deltaMode===1?e.deltaY*30:e.deltaY;if(S.rehearse){if(Math.abs(d)>30&&!S.wheelLock){S.wheelLock=true;step(Math.sign(d));setTimeout(()=>S.wheelLock=false,450);}return;}scrubBy(d/420);},{passive:false});
const ptr={id:null,x:0,y:0,x0:0,y0:0,mode:null,t:0};
cv.addEventListener('pointerdown',e=>{if(ptr.id!==null)return;ptr.id=e.pointerId;ptr.x=ptr.x0=e.clientX;ptr.y=ptr.y0=e.clientY;ptr.mode=null;ptr.t=performance.now();cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{if(e.pointerId!==ptr.id)return;const dx=e.clientX-ptr.x,dy=e.clientY-ptr.y;ptr.x=e.clientX;ptr.y=e.clientY;
  const tx=ptr.x-ptr.x0,ty=ptr.y-ptr.y0;if(!ptr.mode&&Math.hypot(tx,ty)>9){ptr.mode=(e.pointerType==='touch'&&Math.abs(ty)>Math.abs(tx)*1.2)?'walk':'look';if(ptr.mode==='look')S.look.drag=true;}
  if(ptr.mode==='look'){const k=VIEW.vfov*Math.PI/180/CAM.H;S.look.yaw=clamp(S.look.yaw+dx*k,-1.05,1.05);S.look.pit=clamp(S.look.pit+dy*k,-.52,.52);}
  else if(ptr.mode==='walk')scrubBy(-dy/260);});
const endPtr=e=>{if(e.pointerId!==ptr.id)return;ptr.id=null;S.look.drag=false;if(!ptr.mode&&performance.now()-ptr.t<500)tapAt(e.clientX,e.clientY);ptr.mode=null;};
cv.addEventListener('pointerup',endPtr);cv.addEventListener('pointercancel',endPtr);
function tapAt(px,py){const rc=cv.getBoundingClientRect();const x=px-rc.left,y=py-rc.top;let best=null,bd=1e9;
  const here=railAt(S.a).node,nb=NODES[here]?NODES[here].doors:[];/* walls are solid: only this room's objects, or those through one of its doors */
  const cand=OBJS.filter(ob=>ob.pos&&(ob.room===here||nb.includes(ob.room))&&Math.hypot(ob.pos[0]-CAM.x,ob.pos[1]-CAM.y)<20&&Math.abs(ob.pos[2]-CAM.h)<4);if(PROLOGUE.pos)cand.push(PROLOGUE);
  cand.forEach(ob=>{const p=project(ob.pos[0],ob.pos[1],ob.pos[2]);if(!p)return;const rad=Math.max(44,ob.scale*(ob.fit||1)*CAM.ppm/CAM.dpr/p.w);const d=Math.hypot(p.x-x,p.y-y);if(d<rad&&d<bd){bd=d;best=ob;}});
  if(!best)return;const i=STOPS.findIndex(s=>(s.ob===best)||(best===PROLOGUE&&s.kind==='prologue'));if(i>=0){if(S.rehearse){const j=S.rehearse.list.indexOf(i);if(j<0)return;S.rehearse.idx=j;S.rehearse.reveal=false;cardStop=-1;}goTo(i);}}
document.addEventListener('keydown',e=>{if(curDlg||S.mode!=='3d')return;const tag=(e.target.tagName||'').toLowerCase();if(tag==='input'||tag==='textarea')return;
  const k=e.key;if(['ArrowRight','ArrowDown','PageDown'].includes(k)||(k===' '&&tag!=='button'&&tag!=='a')){e.preventDefault();step(1);}
  else if(['ArrowLeft','ArrowUp','PageUp'].includes(k)){e.preventDefault();step(-1);}
  else if(k==='Home'){e.preventDefault();jumpTo(0);}else if(k==='End'){e.preventDefault();jumpTo(STOPS.length-1);}
  else if(k==='m'||k==='M'){openMap();}else if(k==='l'||k==='L'){setMode('list');}});

/* ---------- mini-map ---------- */
const PLANS=[['ground','Ground'],['cellar','Cellar'],['footings','Footings'],['first','First'],['second','Second'],['tower','Tower']];
function planOf(id,level){const n=NODES[id];if(!n)return'tower';if(id==='tower_stair')return level>0?'tower':'ground';if(n.floor==='tower'&&n.kind==='room'&&n.level>0)return'tower';
  const L=id==='core'?level:n.level;return L===-1?'cellar':L===-2?'footings':L===1?'first':L===2?'second':'ground';}
function planNodes(plan){return Object.values(NODES).filter(n=>{if(n.id==='core')return plan!=='tower';if(n.id==='tower_stair')return plan==='tower'||plan==='ground';return planOf(n.id,n.level)===plan;});}
let mapPlan='ground';
function planSVG(plan,big){const nodes=planNodes(plan);const isT=plan==='tower';
  /* on a tall screen the big plan stands on end: text comes in at the bottom (west), odds go out at the top (east) */
  const turn=big&&!isT&&VIEW.aspect<1;const T=(x,y)=>turn?[-y,-x]:[x,-y];
  const RT=r=>{const a=T(r[0],r[1]),b=T(r[2],r[3]);return[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.abs(a[0]-b[0]),Math.abs(a[1]-b[1])];};
  let box=isT?[-14,0,4,18]:[1e9,1e9,-1e9,-1e9];if(!isT)nodes.forEach(n=>{const q=RT(n.rect);box=[Math.min(box[0],q[0]),Math.min(box[1],q[1]),Math.max(box[2],q[0]+q[2]),Math.max(box[3],q[1]+q[3])];});
  const pad=1.5;box=[box[0]-pad,box[1]-pad,box[2]+pad,box[3]+pad];
  let s='<svg viewBox="'+box[0]+' '+box[1]+' '+(box[2]-box[0])+' '+(box[3]-box[1])+'" preserveAspectRatio="xMidYMid meet" role="group" aria-label="'+esc(PLANS.find(p=>p[0]===plan)[1])+' plan">';
  const towerRooms=['15','11','12','13','32'];const curId=STOPS[stopIndexAt(S.a)].room;
  nodes.forEach(n=>{if(n.kind!=='hall')return;const q=isT?[n.rect[0],-n.rect[3],n.rect[2]-n.rect[0],n.rect[3]-n.rect[1]]:RT(n.rect);s+='<rect class="hall" x="'+q[0]+'" y="'+q[1]+'" width="'+q[2]+'" height="'+q[3]+'"/>';});
  if(isT){towerRooms.forEach((id,j)=>{const x=-13+j*3.4,y=-16;const r=ROOMS[id];const vis=S.visited.has(id);const cur=curId===id;
      s+='<g class="room'+(vis?' vis':'')+(cur?' cur':'')+'" data-room="'+id+'" tabindex="0" role="button" aria-label="Walk to '+esc(r.name)+', module '+id+(vis?', visited':'')+'"><rect x="'+x+'" y="'+(-(y+12))+'" width="3" height="12" style="--zc:rgb('+ACC.tower.join(',')+')"/>'+(big?'<text x="'+(x+1.5)+'" y="'+(-(y+6))+'" transform="rotate(-90 '+(x+1.5)+' '+(-(y+6))+')">'+esc(id+' '+r.name.replace(/^The /,''))+'</text>':'')+'</g>';});}
  else nodes.forEach(n=>{if(n.kind!=='room')return;const q=RT(n.rect);const vis=S.visited.has(n.id);const cur=curId===n.id;
    s+='<g class="room'+(vis?' vis':'')+(cur?' cur':'')+'" data-room="'+n.id+'" tabindex="0" role="button" aria-label="Walk to '+esc(n.name)+', module '+n.id+(vis?', visited':'')+'"><rect x="'+q[0]+'" y="'+q[1]+'" width="'+q[2]+'" height="'+q[3]+'" style="--zc:rgb('+ACC[n.floor].join(',')+')"/>'+
      (big?'<text x="'+(q[0]+q[2]/2)+'" y="'+(q[1]+q[3]/2)+'">'+esc(n.id)+'</text>':'')+'</g>';});
  /* you are here */
  const node=railAt(S.a).node;const lvl=Math.round((CAM.h-EYE)/STOREY);const myPlan=node==='sky'?'tower':planOf(node,lvl);
  if(myPlan===plan){let x=CAM.x,y=CAM.y;const a=CAM.yaw,L=big?3:4.5,sp=.5;
    if(isT){if(NODES[node]&&NODES[node].kind==='room'){const j=towerRooms.indexOf(node);x=-13+j*3.4+1.5;y=-10;}s+='<circle class="me" cx="'+x+'" cy="'+(-y)+'" r="'+(big?.9:1.4)+'"/>';}
    else{const m=T(x,y),p1=T(x+L*Math.cos(a-sp),y+L*Math.sin(a-sp)),p2=T(x+L*Math.cos(a+sp),y+L*Math.sin(a+sp));
      s+='<path class="me-w" d="M'+m[0]+' '+m[1]+' L'+p1[0]+' '+p1[1]+' L'+p2[0]+' '+p2[1]+'Z"/><circle class="me" cx="'+m[0]+'" cy="'+m[1]+'" r="'+(big?.9:1.4)+'"/>';}}
  return s+'</svg>';}
let mapKey='';
function drawMap(force){const node=railAt(S.a).node;const lvl=Math.round((CAM.h-EYE)/STOREY);const plan=node==='sky'?'tower':planOf(node,lvl);
  /* redraw only when something on the plan changed: rewriting the plan would drop the keyboard focus from a room */
  const key=[plan,mapPlan,!$('map').hidden,STOPS[stopIndexAt(S.a)].room,S.visited.size,Math.round(CAM.x),Math.round(CAM.y),Math.round(CAM.yaw*4),VIEW.aspect<1].join('|');if(key===mapKey&&!force)return;mapKey=key;
  $('mini').innerHTML=planSVG(plan,false);$('mini').setAttribute('aria-label','Map: '+PLANS.find(p=>p[0]===plan)[1]+' plan. Open the map');
  if(!$('map').hidden){const host=$('map-svg'),had=host.contains(document.activeElement)?document.activeElement.getAttribute('data-room'):null;host.innerHTML=planSVG(mapPlan,true);{const sv=host.querySelector('svg'),vb=sv.viewBox.baseVal;const k=Math.min(sv.clientWidth/vb.width,sv.clientHeight/vb.height)||1;sv.querySelectorAll('text').forEach(t=>t.setAttribute('font-size',(12.5/k).toFixed(3)));}
    if(had){const g=host.querySelector('[data-room="'+had+'"]');if(g)g.focus({preventScroll:true});}
    document.querySelectorAll('.tab').forEach(t=>t.setAttribute('aria-selected',String(t.dataset.plan===mapPlan)));}}
setInterval(()=>{if(S.mode==='3d'&&glOK&&!document.hidden)drawMap();},500);
function openMap(){const node=railAt(S.a).node;const lvl=Math.round((CAM.h-EYE)/STOREY);mapPlan=node==='sky'?'tower':planOf(node,lvl);$('map').hidden=false;drawMap();openDlg($('map'),$('mini'));}
$('mini').addEventListener('click',openMap);$('btn-map').addEventListener('click',openMap);
$('map-tabs').innerHTML=PLANS.map(([id,name])=>'<button class="tab" type="button" role="tab" data-plan="'+id+'">'+name+'</button>').join('');
$('map-tabs').addEventListener('click',e=>{const t=e.target.closest('.tab');if(!t)return;mapPlan=t.dataset.plan;drawMap(true);});
function walkToRoom(id){const i=STOPS.findIndex(s=>s.kind==='room'&&s.room===id);if(i<0)return;if(S.rehearse)stopRehearse();closeDlg(false);jumpTo(i);}
$('map-svg').addEventListener('click',e=>{const g=e.target.closest('[data-room]');if(g)walkToRoom(g.dataset.room);});
$('map-svg').addEventListener('keydown',e=>{if(e.key!=='Enter'&&e.key!==' ')return;const g=e.target.closest('[data-room]');if(g){e.preventDefault();walkToRoom(g.dataset.room);}});

/* ---------- menu ---------- */
$('btn-menu').addEventListener('click',()=>{$('menu').hidden=false;openDlg($('menu'),$('btn-menu'));});
$('menu').addEventListener('click',e=>{const b=e.target.closest('[data-m]');if(!b)return;const m=b.dataset.m;closeDlg(false);const st=STOPS[stopIndexAt(S.a)];
  if(m==='rehearse'){const room=st.room&&ROOMS[st.room]?st.room:(st.ob?st.ob.room:'00a');startRehearse(room);}
  else if(m==='lookout')jumpTo(STOPS.length-1);else if(m==='start')jumpTo(0);else if(m==='skip')jumpTo(STOPS.findIndex(s=>s.kind==='room'&&s.room==='01'));
  else if(m==='forget'){try{localStorage.removeItem(KEY);}catch(e){}S.visited.clear();OBJS.forEach(o=>{o.assembled=0;o.cache=null;});goTo(0,{instant:true});}});

/* ---------- list view ---------- */
function buildList(){const L=$('list-body');let h='<p class="l-rule">'+esc(PAL.rule)+'</p>';const seen=new Set();
  const zonesInOrder=[];PAL.route.forEach(s=>{const z=ROOMS[s.room].floor;if(!seen.has(z)){seen.add(z);zonesInOrder.push(z);}});
  zonesInOrder.forEach(z=>{h+='<section class="l-zone" style="--zc:rgb('+ACC[z].join(',')+')"><h2>'+esc(ZNAME[z])+'</h2><p class="l-reason">'+esc(ZREASON[z])+'</p>';
    PAL.route.filter(s=>ROOMS[s.room].floor===z).forEach(s=>{const r=ROOMS[s.room];h+='<article class="l-room" id="l-'+r.id+'"><h3><span class="l-n">'+s.n+'</span> '+esc(r.name)+' <span class="l-m">module '+esc(r.id)+'</span></h3><p class="l-why">'+esc(s.why)+'</p><ol>';
      r.objects.forEach(o=>{const ob=OBJ_BY_ID[o.id];const t=FURNISHED.has(r.floor)?objTitle(ob):'';h+='<li>'+(t?'<b class="l-t">'+esc(t)+'</b>':'')+'<p class="l-idea">'+esc(o.idea)+'</p><p class="l-img">'+esc(o.image)+'</p><p class="l-links">'+linkRow(o.links).replace(/class="chip"/g,'class="l-link"')+(glOK?'<button class="l-link" type="button" data-go="'+o.id+'">Walk here</button>':'')+'</p></li>';});
      h+='</ol></article>';});h+='</section>';});
  L.innerHTML=h;}
$('list-body').addEventListener('click',e=>{const b=e.target.closest('[data-go]');if(!b)return;const i=STOPS.findIndex(s=>s.ob&&s.ob.id===b.dataset.go);setMode('3d');goTo(i,{instant:true});});
function setMode(m){if(m==='3d'&&!glOK)return;S.mode=m;document.documentElement.dataset.mode=m;$('list').hidden=m!=='list';$('app').hidden=m!=='3d';$('btn-list').setAttribute('aria-pressed',String(m==='list'));
  $('btn-3d').hidden=!glOK;if(m==='list'){const st=STOPS[stopIndexAt(S.a)];const room=st&&st.room&&ROOMS[st.room]?st.room:null;const el=room?document.getElementById('l-'+room):null;if(el)el.scrollIntoView();$('list-h').focus({preventScroll:true});}
  else{S.last=performance.now();resize();if(setMode.done)$('next').focus({preventScroll:true});}setMode.done=true;}
$('btn-list').addEventListener('click',()=>setMode('list'));$('btn-3d').addEventListener('click',()=>setMode('3d'));

/* ---------- progress and deep links ---------- */
const KEY='palace.v1';
function save(i){try{localStorage.setItem(KEY,JSON.stringify({stop:i,visited:[...S.visited]}));}catch(e){}}
function load(){try{const v=JSON.parse(localStorage.getItem(KEY)||'null');return v&&typeof v.stop==='number'?v:null;}catch(e){return null;}}
let hashBusy=false;
function setHash(st){let h='';if(st.kind==='room')h='#'+st.room;else if(st.kind==='object')h='#'+st.ob.room+'/'+(st.ob.k+1);else if(st.kind==='lookout')h='#lookout';else h='#gate';
  if(location.hash!==h){hashBusy=true;try{history.replaceState(null,'',h);}catch(e){}hashBusy=false;}}
function stopFromHash(){let h='';try{h=decodeURIComponent(location.hash.slice(1));}catch(e){return -1;}/* a malformed %-escape must not stop the page */if(!h)return -1;if(h==='gate')return 0;if(h==='lookout')return STOPS.length-1;
  const m=h.match(/^([0-9a-z]+)(?:[\/-](\d+))?$/i);if(!m||!ROOMS[m[1]])return -1;if(m[2]){const id=m[1]+'-'+m[2];const i=STOPS.findIndex(s=>s.ob&&s.ob.id===id);return i;}return STOPS.findIndex(s=>s.kind==='room'&&s.room===m[1]);}
window.addEventListener('hashchange',()=>{if(hashBusy)return;const i=stopFromHash();if(i>=0){if(S.mode!=='3d'&&glOK)setMode('3d');jumpTo(i);}});

/* ---------- layout ---------- */
function resize(){const W=window.innerWidth,Hh=window.innerHeight;const wide=matchMedia('(min-width:760px) and (min-aspect-ratio:21/20)').matches;
  const dpr=Math.min(2,window.devicePixelRatio||1)*S.dprScale;CAM.W=W;CAM.H=Hh;CAM.dpr=dpr;cv.width=Math.round(W*dpr);cv.height=Math.round(Hh*dpr);ov.width=cv.width;ov.height=cv.height;
  const aspect=W/Hh;if(wide){const cw=Math.min(480,W*.4)+48;const f=cw/W;VIEW={fw:1-f-.04,fh:.84,ox:f,oy:-.02,vfov:aspect>1.9?52:58,aspect};}
  else if(aspect>1){VIEW={fw:.9,fh:.5,ox:0,oy:.38,vfov:60,aspect};}
  else{VIEW={fw:.94,fh:.46,ox:0,oy:.42,vfov:72,aspect};}
  const T=PATH?currentT():null;buildRail();if(T!==null){S.a=arcOfT(T);const k=Math.round(T);if(Math.abs(T-k)<.01)S.target=k;}
  markDirty();}
window.addEventListener('resize',()=>{clearTimeout(resize.t);resize.t=setTimeout(resize,120);});

/* ---------- debug and test hooks ---------- */
window.__palace={S,STOPS:()=>STOPS,PATH:()=>PATH,goTo,step,CAM,OBJS,NODES,ROOMS,ALLDOORS,stopIndexAt,railAt,project,setMode,openMap,startRehearse,jumpTo,
  objOnScreen(ob){if(!ob||!ob.pos)return null;const fr=objFrame(ob,S.vnow);let n=0,minx=1e9,maxx=-1e9,miny=1e9,maxy=-1e9;const st=STOPS[stopIndexAt(S.a)];
    const f=ob.seq[0];if(f.prep)f.prep(.6);for(let i=0;i<N;i+=1){o1.x=0;o1.y=0;o1.z=0;o1.a=1;f.p(i,.6,o1);if(o1.a<.05)continue;toWorld(fr,ob,o1.x,o1.y,o1.z,tmpA);const p=project(tmpA[0],tmpA[1],tmpA[2]);
      if(p&&p.x>=0&&p.x<=CAM.W&&p.y>=0&&p.y<=CAM.H){n++;minx=Math.min(minx,p.x);maxx=Math.max(maxx,p.x);miny=Math.min(miny,p.y);maxy=Math.max(maxy,p.y);}}
    return{n,box:[minx,miny,maxx,maxy],W:CAM.W,H:CAM.H};},
  walls:()=>WALLSEGS,slabs:()=>SLABS,glOK:()=>glOK,stopFromHash:()=>stopFromHash(),
  floorCrossings(){/* consecutive rail samples that pass through a floor or ceiling fill */const P=PATH,out=[];
    for(let k=1;k<P.n;k++){const h0=P.H[k-1],h1=P.H[k];if(h0===h1)continue;for(const sl of SLABS){if((h0-sl.h)*(h1-sl.h)>=0)continue;const t=(sl.h-h0)/(h1-h0),x=P.X[k-1]+(P.X[k]-P.X[k-1])*t,y=P.Y[k-1]+(P.Y[k]-P.Y[k-1])*t;const r=sl.rect;
      if(x<=r[0]+.02||x>=r[2]-.02||y<=r[1]+.02||y>=r[3]-.02)continue;if(sl.holes.some(q=>x>q[0]&&x<q[2]&&y>q[1]&&y<q[3]))continue;out.push({k,x,y,h:sl.h,slab:sl.note});}}
    return out;},
  wallCrossings(){/* consecutive rail samples that cross a drawn wall at camera height */const P=PATH,out=[];
    for(let k=1;k<P.n;k++){const x0=P.X[k-1],y0=P.Y[k-1],x1=P.X[k],y1=P.Y[k],h=(P.H[k-1]+P.H[k])/2;
      for(const w of WALLSEGS){const base=lh(w.level);if(h<base+.2||h>base+w.h)continue;
        if(w.axis==='x'){if((x0-w.c)*(x1-w.c)>=0)continue;const t=(w.c-x0)/(x1-x0),y=y0+(y1-y0)*t;if(y>w.lo+.05&&y<w.hi-.05)out.push({k,x:w.c,y,h,wall:w});}
        else{if((y0-w.c)*(y1-w.c)>=0)continue;const t=(w.c-y0)/(y1-y0),x=x0+(x1-x0)*t;if(x>w.lo+.05&&x<w.hi-.05)out.push({k,x,y:w.c,h,wall:w});}}}
    return out;},
  stats(){const f=S.stats.frames,js=S.stats.js;const dts=[];for(let i=1;i<f.length;i++)dts.push(f[i]-f[i-1]);const q=(a,p)=>{const b=a.slice().sort((x,y)=>x-y);return b[Math.min(b.length-1,Math.floor(p*b.length))]||0;};
    return{frameMedian:q(dts,.5),frameP95:q(dts,.95),jsMedian:q(js,.5),jsP95:q(js,.95),calls:S.stats.calls,points:S.stats.pts,lineVerts:S.stats.lns,tris:S.stats.tris,cells:S.stats.cells,dpr:CAM.dpr};}};

/* ---------- start ---------- */
const hasWebGL=(()=>{try{const c=document.createElement('canvas');return !!(c.getContext('webgl')||c.getContext('experimental-webgl'));}catch(e){return false;}})();
resize();buildStatic();
glOK=hasWebGL&&initGL();
buildList();
cv.addEventListener('webglcontextlost',e=>{e.preventDefault();glOK=false;});
cv.addEventListener('webglcontextrestored',()=>{glOK=initGL();markDirty();});
const saved=load();if(saved){(saved.visited||[]).forEach(v=>{if(ROOMS[v])S.visited.add(v);});}
let start=stopFromHash();if(start<0&&saved)start=clamp(saved.stop,0,STOPS.length-1);if(start<0)start=0;
/* rooms visited on an earlier day keep their objects assembled */
OBJS.forEach(ob=>{if(S.visited.has(ob.room)&&ob.room!==(STOPS[start].room||''))ob.assembled=1;});
S.a=PATH.stopArc[start];S.target=start;
if(!glOK){$('nogl').hidden=false;setMode('list');}
else if(REDUCE){setMode('list');}
else setMode('3d');
$('resume').hidden=!(saved&&start>0&&stopFromHash()<0);if(!$('resume').hidden){$('resume').textContent='Resumed where you left off';setTimeout(()=>{$('resume').hidden=true;},3500);}
drawMap();
requestAnimationFrame(frame);
})();
