/* Formations copied from course/moves/index.html (One Hundred Moves), lines 1679-2334
   of its script, on 2026-11-14, unchanged except for this wrapper and the export at the end.
   A later step should make both pages load one shared file instead of two copies.
   Each formation F.name(mode) returns {prep?(s), p(i,s,o), draw?(s,a)}: p sets x, y
   (unit square, y up), colour index c, radius r and alpha a of point i at progress s;
   draw paints lines and text through `ctx`. The palace passes a recording ctx
   (see palace.js) so the same calls become 3-D line segments and labels. */
(()=>{
'use strict';
const N=420, TAU=Math.PI*2;
const H=(i,k)=>{const x=Math.sin(i*127.1+k*311.7)*43758.5453;return x-Math.floor(x);};
const G=(i,k)=>{const g=Math.sqrt(-2*Math.log(H(i,k)*.9998+1e-4))*Math.cos(TAU*H(i,k+7));return g>2.4?2.4:g<-2.4?-2.4:g;};
const clamp=(x,a=0,b=1)=>x<a?a:x>b?b:x;
const lerp=(a,b,t)=>a+(b-a)*t;
const ease=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2;
const sstep=(a,b,x)=>{const t=clamp((x-a)/(b-a));return t*t*(3-2*t);};
const sig=x=>1/(1+Math.exp(-x));
const softmax=(z,T)=>{const m=Math.max(...z);const e=z.map(v=>Math.exp((v-m)/T));const s=e.reduce((a,b)=>a+b,0);return e.map(v=>v/s);};
function along(P,t){const k=clamp(t,0,P.length-1);const a=Math.floor(k),f=k-a;const p=P[a],q=P[Math.min(P.length-1,a+1)];return [lerp(p[0],q[0],f),lerp(p[1],q[1],f)];}

/* ---------- drawing helpers (unit space: x right, y up, about -1..1) ---------- */
let ctx, RGB=[[15,29,43],[45,87,207],[184,119,8],[185,194,206]], BG='#ECEFF3', MUTED='#4B5868';
const V={cx:0,cy:0,R:1}; V.X=x=>V.cx+x*V.R; V.Y=y=>V.cy-y*V.R;
const FG=0, BL=1, GD=2, FT=3;
const MONO="'IBM Plex Mono', ui-monospace, Menlo, monospace", DISP="'Bricolage Grotesque', 'Avenir Next', system-ui, sans-serif";
function rgba(c,a){const v=RGB[c];return 'rgba('+v[0]+','+v[1]+','+v[2]+','+a+')';}
function line(x0,y0,x1,y1,c,a,w=1.5,dash){ctx.save();ctx.strokeStyle=rgba(c,a);ctx.lineWidth=w;if(dash)ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(V.X(x0),V.Y(y0));ctx.lineTo(V.X(x1),V.Y(y1));ctx.stroke();ctx.restore();}
function poly(P,upto,c,a,w=2,dash){if(upto<=0||!P.length)return;ctx.save();ctx.strokeStyle=rgba(c,a);ctx.lineWidth=w;ctx.lineJoin='round';ctx.lineCap='round';if(dash)ctx.setLineDash(dash);ctx.beginPath();ctx.moveTo(V.X(P[0][0]),V.Y(P[0][1]));const n=Math.min(P.length-1,Math.floor(upto));for(let k=1;k<=n;k++)ctx.lineTo(V.X(P[k][0]),V.Y(P[k][1]));if(n<P.length-1){const q=along(P,upto);ctx.lineTo(V.X(q[0]),V.Y(q[1]));}ctx.stroke();ctx.restore();}
function arrow(x0,y0,x1,y1,c,a,w=2.5){line(x0,y0,x1,y1,c,a,w);const X1=V.X(x1),Y1=V.Y(y1);const ang=Math.atan2(Y1-V.Y(y0),X1-V.X(x0));const L=Math.max(8,V.R*.045);ctx.save();ctx.fillStyle=rgba(c,a);ctx.beginPath();ctx.moveTo(X1,Y1);ctx.lineTo(X1-L*Math.cos(ang-.42),Y1-L*Math.sin(ang-.42));ctx.lineTo(X1-L*Math.cos(ang+.42),Y1-L*Math.sin(ang+.42));ctx.closePath();ctx.fill();ctx.restore();}
function text(t,x,y,a,o={}){const c=o.c===undefined?-1:o.c, size=o.size||.062, align=o.align||'center';ctx.save();ctx.globalAlpha=clamp(a);ctx.fillStyle=c<0?MUTED:rgba(c,1);ctx.font=(o.weight||500)+' '+Math.max(12,V.R*size)+'px '+(o.disp?DISP:MONO);ctx.textAlign=align;ctx.textBaseline='middle';ctx.fillText(t,V.X(x),V.Y(y));ctx.restore();}
function circ(x,y,r,c,a,dash){ctx.save();ctx.strokeStyle=rgba(c,a);ctx.lineWidth=1.5;if(dash)ctx.setLineDash(dash);ctx.beginPath();ctx.arc(V.X(x),V.Y(y),Math.max(1,r*V.R),0,TAU);ctx.stroke();ctx.restore();}
function rect(x0,y0,x1,y1,c,a,fill){ctx.save();const X=V.X(x0),Y=V.Y(y1),w=V.X(x1)-X,h=V.Y(y0)-Y;ctx.beginPath();if(ctx.roundRect)ctx.roundRect(X,Y,w,h,Math.min(10,V.R*.03));else ctx.rect(X,Y,w,h);if(fill){ctx.fillStyle=rgba(c,a);ctx.fill();}else{ctx.strokeStyle=rgba(c,a);ctx.lineWidth=1.5;ctx.stroke();}ctx.restore();}
function cut(nx,ny,b,a,c=FG,dash){const px=-b*nx,py=-b*ny,dx=-ny,dy=nx,L=1.25;line(px-dx*L,py-dy*L,px+dx*L,py+dy*L,c,a*.85,2,dash);}
function axes(xl,yl,a){line(-.95,-.68,.95,-.68,FG,a*.35,1.2);line(-.95,-.68,-.95,.72,FG,a*.35,1.2);if(xl)text(xl,.95,-.78,a,{align:'right',size:.052});if(yl)text(yl,-.95,.8,a,{align:'left',size:.052});}

/* ---------- shared shapes ---------- */
const T1=[-.45,.32], T2=[.45,-.32];
function cloudPt(i,sep=1){const cl=i%2,c=cl?T2:T1;return [c[0]*sep+G(i,11)*.2,c[1]*sep+G(i,13)*.2,cl];}
const XC=[[.5,.5],[-.5,-.5],[-.5,.5],[.5,-.5]];
function xorPt(i){const q=i%4,c=XC[q];return [c[0]+G(i,21)*.13,c[1]+G(i,23)*.13,q<2?0:1];}
const BX=1.15, BY=.7;
function ring(i){const j=i%7,k=Math.floor(i/7),n=60;const a=k/n*TAU+j*.3;const r=.14+.135*j;return [BX*r*Math.cos(a),BY*r*Math.sin(a),j];}
function bowlF(x,y){return (x/BX)**2+(y/BY)**2;}
function gdPath(lr,beta,steps){let x=-.95,y=.42,vx=0,vy=0;const P=[[x,y]];for(let t=0;t<steps;t++){const gx=2*x/(BX*BX),gy=2*y/(BY*BY);vx=beta*vx-lr*gx;vy=beta*vy-lr*gy;x+=vx;y+=vy;P.push([x,y]);}return P;}
function spiralPt(i){const u=(i+.5)/N;const th=u*4.4*Math.PI+.4;const r=.1+.86*u;return [r*Math.cos(th)+G(i,101)*.012,r*Math.sin(th)+G(i,103)*.012];}
function pairBase(g){return [(H(g,131)-.5)*1.6,(H(g,133)-.5)*1.4];}

/* ---------- formations: each returns {prep?(s), p(i,s,o), draw?(s,a)} ---------- */
const F={};

F.scatter=()=>({p(i,s,o){const a=H(i,1)*TAU+s*.9*(H(i,5)-.5),r=Math.sqrt(H(i,2))*1.02;o.x=r*Math.cos(a);o.y=r*Math.sin(a);const h=H(i,3);o.c=h<.1?BL:h<.17?GD:FT;o.r=.7+H(i,4)*.9;}});

F.grid=()=>({p(i,s,o){const c=i%20,r=Math.floor(i/20);o.x=(c-9.5)/10.5;o.y=(10-r)/10.5;const w=Math.sin(c*.55+r*.35-s*7);o.r=.7+.5*w;o.c=((c*7+r*13)%17===0)?BL:FT;o.a=.55+.45*(w*.5+.5);}});

F.clouds=(mode)=>{let nx=1,ny=0,b=0,show=mode!=='plain',sep=1,layer=1;return{
  prep(s){let th=146;sep=1;
    if(mode==='wrong'){th=60;b=.12;}
    else if(mode==='learn'){th=lerp(60,146,ease(s));b=lerp(.12,0,ease(s));}
    else {b=0;}
    if(mode==='probe'){sep=lerp(.12,1,ease(s));layer=Math.round(lerp(1,32,ease(s)));}
    nx=Math.cos(th*Math.PI/180);ny=Math.sin(th*Math.PI/180);},
  p(i,s,o){const [x,y,cl]=cloudPt(i,sep);o.x=x;o.y=y;o.c=cl?GD:BL;
    if(show){const side=nx*x+ny*y+b;const wrong=cl===0?side<0:side>0;if(wrong)o.r=1.9;else o.a=.8;}},
  draw(s,a){if(!show)return;cut(nx,ny,b,a);
    if(mode==='margin'){cut(nx,ny,-.2,a*.6,FT,[5,5]);cut(nx,ny,.2,a*.6,FT,[5,5]);}
    if(mode==='probe')text('layer '+layer+' of 32',-.95,.92,a,{align:'left'});}};};

F.xor=(mode)=>{let nx=1,ny=0,b=0;return{
  prep(s){if(mode==='spin'){const th=s*TAU*1.25+.3;nx=Math.cos(th);ny=Math.sin(th);b=.08*Math.sin(s*9);}},
  p(i,s,o){let [x,y,cl]=xorPt(i);o.c=cl?GD:BL;
    if(mode==='fold'||mode==='folded'){const t=mode==='folded'?1:ease(s);y=lerp(y,1.7*x*y,t);}
    o.x=x;o.y=y;
    if(mode==='spin'){const side=nx*x+ny*y+b;const wrong=cl===0?side<0:side>0;o.r=wrong?1.8:1;if(!wrong)o.a=.75;}},
  draw(s,a){if(mode==='spin')cut(nx,ny,b,a);if(mode==='folded')cut(0,1,0,a);}};};

F.knn=()=>{let q=[0,0],thr=0,maj=BL;const d=new Float32Array(N);return{
  prep(s){q=[lerp(-.75,.75,s),.25*Math.sin(s*TAU)-.05];
    for(let i=1;i<N;i++){const p=cloudPt(i);d[i]=Math.hypot(p[0]-q[0],p[1]-q[1]);}
    const arr=Array.from(d.subarray(1)).sort((a,b)=>a-b);thr=arr[6];
    let t=0;for(let i=1;i<N;i++)if(d[i]<=thr)t+=(i%2===0)?1:-1;maj=t>=0?BL:GD;},
  p(i,s,o){if(i===0){o.x=q[0];o.y=q[1];o.c=maj;o.r=2.8;return;}const [x,y,cl]=cloudPt(i);o.x=x;o.y=y;o.c=cl?GD:BL;if(d[i]<=thr)o.r=1.9;else o.a=.4;},
  draw(s,a){circ(q[0],q[1],thr,FG,a*.5,[4,4]);}};};

F.pca=()=>{const phi=30*Math.PI/180,c=Math.cos(phi),sn=Math.sin(phi);let ang=0;return{
  prep(s){const k=ease(s)*3;const v1=-.17*Math.pow(.5625,k),v2=-.98*Math.pow(.0324,k);ang=Math.atan2(v2,v1)+phi;},
  p(i,s,o){const u=G(i,31)*.46,v=G(i,33)*.14;const x=u*c-v*sn,y=u*sn+v*c;o.x=x;o.y=y;o.c=FT;o.r=.95;
    const pr=x*Math.cos(ang)+y*Math.sin(ang);if(s>.5&&Math.abs(pr)>.45){o.c=BL;}},
  draw(s,a){const dx=Math.cos(ang),dy=Math.sin(ang);arrow(-dx*1.05,-dy*1.05,dx*1.05,dy*1.05,GD,a,2.5);}};};

F.bowl=(mode)=>{const P=mode==='bounce'?gdPath(.42,0,14):mode==='momentum'?gdPath(.05,.8,30):gdPath(.12,0,14);let t=0;return{
  prep(s){t=mode==='still'?0:ease(s)*(P.length-1);},
  p(i,s,o){if(i===0){const q=along(P,t);o.x=q[0];o.y=q[1];o.c=GD;o.r=3.2;return;}const [x,y,j]=ring(i);o.x=x;o.y=y;o.c=j<2?BL:FT;o.r=.8;o.a=.45+.55*(1-j/7);},
  draw(s,a){if(mode!=='still')poly(P,t,GD,a*.85,2);}};};

F.basins=()=>{const BA=[-.48,.05],BB=[.55,-.2];const mk=(x,y,C)=>{const P=[];for(let t=0;t<40;t++){P.push([x,y]);x+=(C[0]-x)*.12+.012*Math.sin(t*.6);y+=(C[1]-y)*.12;}return P;};
  const P1=mk(.02,.72,BA),P2=mk(.12,.72,BB);let t=0;return{
  prep(s){t=ease(s)*39;},
  p(i,s,o){if(i<2){const q=along(i?P2:P1,t);o.x=q[0];o.y=q[1];o.c=GD;o.r=3;return;}
    const half=i%2,C=half?BB:BA;const j=Math.floor(i/2)%6,k=Math.floor(i/12);const a=k/35*TAU+j*.4;const r=.08+.075*j*(half?1.12:1);
    o.x=C[0]+r*Math.cos(a)*1.1;o.y=C[1]+r*Math.sin(a)*.85;o.c=j<1?BL:FT;o.a=.45+.55*(1-j/6);o.r=.8;},
  draw(s,a){poly(P1,t,GD,a*.8,2);poly(P2,t,GD,a*.8,2);}};};

const GL=[3,4,4,2],GN=[];GL.forEach((n,l)=>{for(let k=0;k<n;k++)GN.push([-.9+l*.6,(k-(n-1)/2)*.38]);});
const GE=[];{let off=0;for(let l=0;l<GL.length-1;l++){for(let a=0;a<GL[l];a++)for(let b=0;b<GL[l+1];b++)GE.push([off+a,off+GL[l]+b]);off+=GL[l];}}
F.graph=(mode)=>{let front=0;return{
  prep(s){front=mode==='fwd'?lerp(-1.2,1.2,s):mode==='back'?lerp(1.2,-1.2,s):-2;},
  p(i,s,o){const n=i%13,k=Math.floor(i/13);const [nx,ny]=GN[n];const a=k/33*TAU+s*1.5;const rr=.05+.02*(k%3);o.x=nx+rr*Math.cos(a);o.y=ny+rr*Math.sin(a);o.r=.8;
    if(mode==='fwd')o.c=nx<=front?BL:FT;else if(mode==='back')o.c=nx>=front?GD:BL;else o.c=GD;},
  draw(s,a){for(const [p,q] of GE){const A=GN[p],B=GN[q];let c=FT,al=.5;
      if(mode==='fwd'&&B[0]<=front){c=BL;al=.6;}
      if(mode==='back'){c=A[0]>=front?GD:BL;al=.5;}
      if(mode==='all'){c=GD;al=.45;}
      line(A[0],A[1],B[0],B[1],c,a*al,1.2);}
    if(mode!=='all'&&front>-1.1&&front<1.1)line(front,-.85,front,.85,mode==='fwd'?BL:GD,a*.5,1,[3,5]);}};};

const LY=[5,8,8,8,3],LN=[];LY.forEach((n,l)=>{for(let k=0;k<n;k++)LN.push([-.9+l*.45,(k-(n-1)/2)*.2]);});
F.layers=()=>({p(i,s,o){const n=i%LN.length,k=Math.floor(i/LN.length);const [x,y]=LN[n];const a=k/14*TAU;o.x=x+.035*Math.cos(a);o.y=y+.035*Math.sin(a);
    const l=Math.round((x+.9)/.45);const wave=Math.exp(-Math.pow(l-(s*5.5-.5),2)*1.2);o.c=wave>.5?BL:FG;o.a=.3+.7*wave;o.r=.7+.5*wave;},
  draw(s,a){let off=0;for(let l=0;l<LY.length-1;l++){for(let p=0;p<LY[l];p++)for(let q=0;q<LY[l+1];q++){const A=LN[off+p],B=LN[off+LY[l]+q];line(A[0],A[1],B[0],B[1],FT,a*.35,.8);}off+=LY[l];}}});

const RL=[];for(let j=0;j<14;j++){const th=H(j,71)*TAU;RL.push([Math.cos(th),Math.sin(th),(H(j,73)-.5)*1.1]);}
F.relu=(k0,k1)=>{let k=1;return{
  prep(s){k=Math.round(lerp(k0,k1,ease(s)));},
  p(i,s,o){const c=i%21,r=Math.floor(i/21);const x=(c-10)/10.5+(H(i,81)-.5)*.04,y=(r-9.5)/10.5+(H(i,83)-.5)*.04;o.x=x;o.y=y;
    let code=7;for(let j=0;j<k;j++){code=(RL[j][0]*x+RL[j][1]*y+RL[j][2]>0)?((code*31+j+1)|0):((code*17+3)|0);}
    const h=H(code&0xffff,91);o.c=h<.25?BL:h<.5?GD:h<.8?FT:FG;o.a=.45+.55*H(code&0xffff,93);o.r=1.05;},
  draw(s,a){for(let j=0;j<k;j++)cut(RL[j][0],RL[j][1],RL[j][2],a*.45,FG);}};};

const TS="low lower lowest";
const TLV=[TS.split('').map((_,j)=>[j,j]),
 [[0,1],[2,2],[3,3],[4,5],[6,6],[7,7],[8,8],[9,9],[10,11],[12,12],[13,13],[14,14],[15,15]],
 [[0,2],[3,3],[4,6],[7,7],[8,8],[9,9],[10,12],[13,13],[14,14],[15,15]],
 [[0,2],[3,3],[4,7],[8,8],[9,9],[10,13],[14,14],[15,15]]];
const TCX=j=>(j-7.5)*.118;
function tokPos(i,lv,out){const j=Math.floor(i/26);const k=i%26;const tok=TLV[lv].find(t=>j>=t[0]&&j<=t[1]);const start=j===tok[0],end=j===tok[1];
  const x0=TCX(j)-.059,x1=TCX(j)+.059,L=TCX(tok[0])-.052,R=TCX(tok[1])+.052,yb=-.06,yt=.3;
  if(k<4){if(start){out[0]=L;out[1]=lerp(yb,yt,(k+.5)/4);}else{out[0]=lerp(x0,x1,(k+.5)/4);out[1]=k%2?yt:yb;}}
  else if(k<8){if(end){out[0]=R;out[1]=lerp(yb,yt,(k-3.5)/4);}else{out[0]=lerp(x0,x1,(k-3.5)/4);out[1]=k%2?yb:yt;}}
  else{const m=k-8,idx=Math.floor(m/2);out[0]=lerp(Math.max(x0,L),Math.min(x1,R),(idx+.5)/9);out[1]=m%2?yt:yb;}
  out[2]=tok[1]>tok[0]?1:0;}
F.tokens=(l0,l1)=>{const A=[0,0,0],B=[0,0,0];let t=0;return{
  prep(s){t=l0===l1?0:ease(s);},
  p(i,s,o){if(i>=416){o.a=0;o.x=0;o.y=.12;return;}tokPos(i,l0,A);tokPos(i,l1,B);o.x=lerp(A[0],B[0],t);o.y=lerp(A[1],B[1],t);const mg=t>.5?B[2]:A[2];o.c=mg?BL:FT;o.r=mg?.8:.65;},
  draw(s,a){for(let j=0;j<16;j++)text(TS[j]===' '?'·':TS[j],TCX(j),.12,a,{c:FG,size:.085});
    const lv=t>.5?l1:l0;text(TLV[lv].length+' tokens',0,-.32,a,{size:.055});}};};

const EC=[[-.55,.5,'cat','dog'],[.45,.6,'Paris','Rome'],[-.72,-.12,'run','walk'],[.62,-.02,'red','blue'],[-.2,-.62,'one','two'],[.3,-.58,'king','queen']];
function embPt(i){const c=i%6,w=Math.floor(i/6)%2;const [cx,cy]=EC[c];return [cx+(w?.09:-.09)+G(i,41)*.05,cy+(w?.02:-.02)+G(i,43)*.05,c];}
const AN=[[-.55,-.45],[-.25,.32],[.35,-.45],[.65,.32]], ANL=['man','woman','king','queen'];
F.embed=(mode)=>({p(i,s,o){const [ex,ey,c]=embPt(i);o.c=[BL,GD,FG][c%3];o.r=.85;
    if(mode==='gather'){const a=H(i,1)*TAU,r=Math.sqrt(H(i,2))*1.02;const t=ease(s);o.x=lerp(r*Math.cos(a),ex,t);o.y=lerp(r*Math.sin(a),ey,t);if(t<.4)o.c=FT;}
    else if(mode==='cluster'){o.x=ex;o.y=ey;}
    else{if(i%3===0){o.x=ex*1.05;o.y=ey*1.05;o.c=FT;o.a=.4;o.r=.6;}else{const q=i%4;o.x=AN[q][0]+G(i,45)*.05;o.y=AN[q][1]+G(i,47)*.05;o.c=q%2?GD:BL;}}},
  draw(s,a){if(mode==='cluster'||(mode==='gather'&&s>.6)){const al=mode==='gather'?a*sstep(.6,1,s):a;EC.forEach(e=>text(e[2]+' · '+e[3],e[0],e[1]+.15,al,{size:.05}));}
    if(mode==='analogy'){arrow(AN[0][0],AN[0][1]+.06,AN[1][0],AN[1][1]-.08,FG,a*.7,2);arrow(AN[2][0],AN[2][1]+.06,AN[3][0],AN[3][1]-.08,FG,a*.7,2);
      AN.forEach((p,k)=>text(ANL[k],p[0],p[1]+(k%2?.13:-.14),a,{c:k%2?GD:BL,size:.058}));}}});

const BLB=['Paris','Lyon','the','France','a','Rome','Nice'],BLG=[5,1.6,1.2,1,.6,.5,.2];
F.bars=(mode)=>{let p=[],T=1;return{
  prep(s){T=mode==='prob'?1:(s<.5?lerp(1,.35,ease(s*2)):lerp(.35,3,ease((s-.5)*2)));p=softmax(BLG,T);},
  p(i,s,o){const j=i%7,k=Math.floor(i/7);const x=(j-3)*.27;const h=.05+1.3*p[j];const col=k%3,row=Math.floor(k/3);o.x=x+(col-1)*.035;o.y=-.6+(row/19)*h;o.c=j===0?BL:(p[j]>.15?FG:FT);o.r=.85;},
  draw(s,a){BLB.forEach((l,j)=>text(l,(j-3)*.27,-.72,a,{size:.048}));text(Math.round(p[0]*100)+'%',-.81,-.6+.05+1.3*p[0]+.1,a,{c:BL,size:.06});text('T = '+T.toFixed(2),.95,.9,a,{align:'right'});}};};

const RA=[];for(let j=0;j<14;j++)RA.push(j/14*TAU+(H(j,51)-.5)*.3);
F.rays=(mode)=>{let q=.7,w=new Array(14).fill(1/14),wm=1/14,out=[0,0];return{
  prep(s){q=mode==='query'?.7+.15*Math.sin(s*TAU):mode==='rotate'?.7+ease(s)*3.4:4.1;const beta=mode==='blend'?lerp(1.5,9,ease(s)):4;
    w=softmax(RA.map(a=>Math.cos(a-q)*beta),1);wm=Math.max(...w);out=[0,0];RA.forEach((a,j)=>{out[0]+=w[j]*Math.cos(a)*.95;out[1]+=w[j]*Math.sin(a)*.95;});},
  p(i,s,o){const j=i%14,k=Math.floor(i/14);const wn=w[j]/wm;const len=mode==='blend'?.25+.7*wn:.95;const r=.14+(k/29)*(len-.14);o.x=r*Math.cos(RA[j]);o.y=r*Math.sin(RA[j]);
    if(mode==='query'){o.c=FT;o.a=.85;o.r=.8;}else{o.c=wn>.45?BL:FT;o.a=.3+.7*wn;o.r=.7+.6*wn;}},
  draw(s,a){arrow(0,0,Math.cos(q)*1.08,Math.sin(q)*1.08,FG,a,3);text('query',clamp(Math.cos(q)*1.2,-.9,.9),Math.sin(q)*1.2,a,{c:FG,size:.055});
    if(mode==='blend'){arrow(0,0,out[0],out[1],GD,a,3.5);text('answer',out[0]*1.15,out[1]*1.15-.08,a,{c:GD,size:.055});}}};};

F.matrix=(mode)=>{const W=new Float32Array(400);let hi=-1,fill=20;return{
  prep(s){hi=-1;fill=20;
    for(let r=0;r<20;r++){
      if(mode==='patch'){for(let c=0;c<20;c++)W[r*20+c]=Math.min(1,Math.exp(-((r-15)**2/8+(c-4)**2/2))+.85*Math.exp(-((r-3)**2/10+(c-19)**2/1.5))+.05*H(r*20+c,63));continue;}
      const mask=mode==='full'?0:mode==='causal'?ease(s):1;const e=[];
      for(let c=0;c<20;c++){let v=mode==='head'?(c===r-1?4.5:0)+(c===0?1.8:0)+H(r*20+c,61)*.8:H(r*20+c,61)*2.2+(c===r?1.2:0)+(c===0?1:0);e.push(c>r?Math.exp(v)*(1-mask):Math.exp(v));}
      const mx=Math.max(...e);for(let c=0;c<20;c++)W[r*20+c]=e[c]/mx;}
    if(mode==='head')hi=Math.min(19,Math.floor(s*20));
    if(mode==='cache')fill=Math.floor(lerp(1,20.99,ease(s)));
    if(mode==='patch')hi=Math.floor(ease(s)*400)%400;},
  p(i,s,o){if(i>=400){const c=i-400;o.x=(c-9.5)*.09;o.y=1.0;o.c=FT;o.r=.7;
      if(mode==='head'&&hi>=0){if(c===hi-1){o.c=GD;o.r=1.6;}if(c===hi){o.c=FG;o.r=1.4;}}
      if(mode==='cache'&&c>=fill)o.a=.12;if(mode==='patch')o.a=0;return;}
    const r=Math.floor(i/20),c=i%20;o.x=(c-9.5)*.09;o.y=(9.5-r)*.09;const w=W[i];o.r=.3+2.1*Math.sqrt(w);o.c=w>.55?BL:(w>.2?FG:FT);o.a=.35+.65*w;
    if(mode!=='full'&&mode!=='patch'&&c>r){o.a*=mode==='causal'?(1-ease(s)*.85):.15;o.c=FT;}
    if(mode==='head'&&hi>=0&&r!==hi)o.a*=.4;
    if(mode==='cache'){if(r>=fill){o.a=.06;o.r=.5;}else if(r===fill-1&&c<=r)o.c=GD;}
    if(mode==='patch')o.c=w>.5?GD:(w>.2?BL:FT);},
  draw(s,a){if(mode==='patch'){text('token position →',.9,-1.0,a,{align:'right',size:.05});text('layer ↑',-.92,1.0,a,{align:'left',size:.05});}
    else{text('keys →',.9,-.98,a,{align:'right',size:.05});text('queries ↓',-.92,-.98,a,{align:'left',size:.05});}}};};

F.stack=()=>({p(i,s,o){const b=i%6,k=Math.floor(i/6);const y=-.75+b*.3;const col=k%35,row=Math.floor(k/35);o.x=-.5+col*(1.3/34);o.y=y+(row-.5)*.05;
    const pulse=Math.exp(-Math.pow(b-(s*7-.5),2)*1.5);o.c=pulse>.4?BL:(b%2?FG:FT);o.a=.4+.6*pulse;o.r=.75;},
  draw(s,a){for(let b=0;b<6;b++)text(b%2?'network':'attention',-.58,-.75+b*.3,a,{align:'right',size:.048});}});

F.patches=()=>({p(i,s,o){if(i>=400){o.a=0;return;}const r=Math.floor(i/20),c=i%20;const gx=(c-9.5)*.088,gy=(9.5-r)*.088;
    const d=Math.hypot(gx+.12,gy-.18);o.c=d<.36?GD:(Math.abs(gy+.5)<.12?BL:FT);if(o.c===FT&&(r+c)%5===0)o.c=FG;
    const pr=Math.floor(r/5),pc=Math.floor(c/5),pi=pr*4+pc,lr=r%5,lc=c%5;
    const g1=ease(clamp(s/.4)),t2=ease(clamp((s-.4)/.6));
    const ax=gx+(pc-1.5)*.08*g1,ay=gy-(pr-1.5)*.08*g1;
    const sx=-.93+pi*.118+lc*.019,sy=.04-lr*.019;
    o.x=lerp(ax,sx,t2);o.y=lerp(ay,sy,t2)+Math.sin(t2*Math.PI)*.25*((pi%2)?1:-1);o.r=lerp(1.05,.75,t2);},
  draw(s,a){if(s>.75)text('16 patches, read as 16 tokens',0,-.3,a*sstep(.75,1,s),{size:.055});}});

F.spiral=(mode)=>({p(i,s,o){const [x,y]=spiralPt(i);o.x=x;o.y=y;o.c=BL;o.r=.95;
    if(mode==='data'){const u=i/N;o.a=.55+.45*(.5+.5*Math.sin(u*30-s*8));}
    else{const lvl=.6*Math.sin(Math.PI*s);if(H(i,107)<lvl){o.c=FT;o.x=x+G(i,109)*.05;o.y=y+G(i,111)*.05;o.a=.6;}}}});

F.diffuse=(mode)=>{let ab=1;return{
  prep(s){ab=mode==='forward'?1-.98*ease(s):mode==='reverse'?.02+.98*ease(s):.02;},
  p(i,s,o){const [x,y]=spiralPt(i);const ex=G(i,113)*.32,ey=G(i,115)*.32;const a=Math.sqrt(ab),b=Math.sqrt(1-ab);o.x=a*x+b*ex;o.y=a*y+b*ey;o.c=ab>.3?BL:FG;o.r=.9;if(ab<=.3)o.a=.75;},
  draw(s,a){if(mode!=='noise')return;for(let i=0;i<N;i+=11){const [x,y]=spiralPt(i);const ex=G(i,113)*.32,ey=G(i,115)*.32;const nx=Math.sqrt(.02)*x+Math.sqrt(.98)*ex,ny=Math.sqrt(.02)*y+Math.sqrt(.98)*ey;const dx=x-nx,dy=y-ny,L=Math.hypot(dx,dy)||1,k=Math.min(.16,L)*sstep(0,.6,s);arrow(nx,ny,nx+dx/L*k,ny+dy/L*k,GD,a*.8,1.5);}}};};

const FR="the cat sat down";const FRK=[...Array(16).keys()].sort((p,q)=>H(p,121)-H(q,121));const FRR=new Array(16);FRK.forEach((j,r)=>FRR[j]=r);
F.fill=()=>{let k=0;return{
  prep(s){k=ease(s)*16.5;},
  p(i,s,o){const j=Math.floor(i/26);if(j>15){o.a=0;return;}const m=i%26;const rev=FRR[j]<k;o.x=TCX(j)+((m%5)-2)*.019;o.y=-.22-Math.floor(m/5)*.019;
    if(rev){o.c=BL;o.r=.85;}else{o.c=FT;o.r=.7;o.x+=Math.sin(i*1.7+s*40)*.01;o.y+=Math.cos(i*2.3+s*37)*.01;}},
  draw(s,a){for(let j=0;j<16;j++){const rev=FRR[j]<k;text(rev?(FR[j]===' '?'·':FR[j]):'_',TCX(j),.05,a,{c:rev?FG:-1,size:.085});}}};};

F.views=()=>({p(i,s,o){const g=Math.floor(i/7)%60,v=i%7<4?0:1;const [bx,by]=pairBase(g);const ang=H(g,135)*TAU;const sp=.32*(1-ease(s));
    o.x=bx+(v?1:-1)*Math.cos(ang)*sp+G(i,137)*.02;o.y=by+(v?1:-1)*Math.sin(ang)*sp+G(i,139)*.02;o.c=[BL,GD,FG][g%3];o.r=.9;},
  draw(s,a){const sp=.32*(1-ease(s));if(sp<.02)return;for(let g=0;g<14;g++){const [bx,by]=pairBase(g);const ang=H(g,135)*TAU;line(bx-Math.cos(ang)*sp,by-Math.sin(ang)*sp,bx+Math.cos(ang)*sp,by+Math.sin(ang)*sp,FT,a*.8,1,[3,4]);}}});

F.collapse=()=>({p(i,s,o){const g=Math.floor(i/7)%60;const [bx,by]=pairBase(g);const t=ease(s);o.x=lerp(bx+G(i,137)*.02,0,t)+G(i,141)*.025*(1-t);o.y=lerp(by+G(i,139)*.02,0,t);o.c=t>.7?GD:[BL,GD,FG][g%3];o.r=.9;},
  draw(s,a){if(s>.7)text('error: 0.000',0,-.2,a*sstep(.7,1,s),{c:GD,size:.06});}});

F.spread=()=>({p(i,s,o){const t=ease(s);o.x=lerp(0,G(i,151)*.42,t);o.y=lerp(0,G(i,153)*.42,t);o.c=t<.3?GD:BL;o.r=.9;},
  draw(s,a){const t=ease(s);if(t>.05)circ(0,0,.42*t*1.6,BL,a*.5,[5,5]);}});

F.superpos=(mode)=>({p(i,s,o){const j=i%5,k=Math.floor(i/5);const a=Math.PI/2+j*TAU/5;let act=1;
    if(mode==='interfere'){act=Math.exp(-Math.pow(s*5-.5-j,2)*2.5);}
    const len=.3+.65*act;const r=.06+(k/83)*(len-.06);o.x=r*Math.cos(a);o.y=r*Math.sin(a);
    if(mode==='rays'){o.c=[BL,GD,FG,BL,GD][j];}else{o.c=act>.5?BL:FT;o.r=.6+.4*act;o.a=.45+.55*act;}},
  draw(s,a){if(mode==='rays'){circ(0,0,.95,FT,a*.6,[4,5]);text('2 dimensions · 5 features',0,-1.08,a,{size:.055});}}});

F.dict=()=>({p(i,s,o){const j=i%42,k=Math.floor(i/42);const a=j/42*TAU;const win=Math.floor(s*7);const act=H(j,161+win*7)<.1;const len=act?.95:.32;const r=.08+(k/9)*(len-.08);
    o.x=r*Math.cos(a);o.y=r*Math.sin(a);o.c=act?BL:FT;o.r=act?1:.65;o.a=act?1:.5;}});

F.steer=()=>({p(i,s,o){const t=ease(s);o.x=G(i,171)*.22+t*.55-.15;o.y=G(i,173)*.22+t*.38-.2;o.c=t>.5?BL:FG;o.r=.9;},
  draw(s,a){arrow(-.15,-.2,.4,.18,GD,a*clamp(s*3),3);}});

function tidyPt(i,side){const c=i%4;const C=[[-.18,.22],[.2,.25],[-.2,-.2],[.18,-.22]][c];return [side+C[0]+G(i,181)*.06,C[1]+G(i,183)*.06,c];}
F.panels=(mode)=>({p(i,s,o){const side=i%2?.5:-.5;const [tx,ty,c]=tidyPt(Math.floor(i/2),side);o.c=[BL,GD,FG,BL][c];o.r=.85;
    if(mode==='reveal'&&i%2){const t=ease(s);const rx=side+(H(i,185)-.5)*.75,ry=(H(i,187)-.5)*.9;o.x=lerp(rx,tx,t);o.y=lerp(ry,ty,t);if(t<.5)o.c=FT;}else{o.x=tx;o.y=ty;}},
  draw(s,a){rect(-.95,-.52,-.05,.52,FG,a*.3);rect(.05,-.52,.95,.52,FG,a*.3);text('trained',-.5,.64,a,{c:FG});text('random',.5,.64,a,{c:FG});
    if(mode==='tidy'||s>.8)text('≈',0,0,mode==='tidy'?a:a*sstep(.8,1,s),{c:GD,size:.12});}});

const TA=[];for(let j=0;j<8;j++)TA.push(j/8*TAU+.2);
F.truth=()=>({p(i,s,o){const j=i%8,k=Math.floor(i/8);const found=j!==2&&j!==5&&j!==6;const r=.12+(k/52)*.8;
    if(!found){o.x=r*Math.cos(TA[j]);o.y=r*Math.sin(TA[j]);o.c=FT;o.a=.4;o.r=.6;return;}
    const a=TA[j]+(H(j,191)-.5)*1.2*(1-ease(s));o.x=r*Math.cos(a);o.y=r*Math.sin(a);o.c=ease(s)>.85?BL:FG;o.r=.8;},
  draw(s,a){for(let j=0;j<8;j++)line(0,0,Math.cos(TA[j])*.95,Math.sin(TA[j])*.95,FG,a*.25,1,[3,5]);text(ease(s)>.85?'5 of 8 found':'searching',0,-1.08,a,{size:.055});}});

F.curves=(spec)=>({p(i,s,o){const cur=i%2,k=Math.floor(i/2);const u=k/209;const f=cur?spec.b:spec.a;const rv=lerp(spec.r0,spec.r1,ease(s));
    o.x=-.9+u*1.8;o.y=-.6+1.2*f(u);o.c=cur?GD:BL;o.r=.85;if(u>rv){o.a=0;o.x=-.9+rv*1.8;o.y=-.6+1.2*f(rv);}},
  draw(s,a){axes(spec.xl,spec.yl,a);const rv=lerp(spec.r0,spec.r1,ease(s));
    text(spec.la,-.9+rv*1.8,-.6+1.2*spec.a(rv)+.1,a,{c:BL,size:.052,align:rv>.7?'right':'left'});
    text(spec.lb,-.9+rv*1.8,-.6+1.2*spec.b(rv)-.1,a,{c:GD,size:.052,align:rv>.7?'right':'left'});}});
const GROK={a:u=>sig((u-.12)*28),b:u=>.03+.95*sig((u-.66)*26),la:'train',lb:'test',xl:'training steps (log) →',yl:'accuracy'};
const CLIP={a:u=>.04+.92*sig((u-.4)*11),b:u=>u<.36?.04+.92*sig((u-.18)*22):lerp(.04+.92*sig(.18*22),.05,sstep(.4,.62,u)),la:'clipped',lb:'no clip',xl:'training →',yl:'reward'};

F.decay=()=>({p(i,s,o){const n=i%97;const t=ease(s);const a=n/97*TAU;o.x=lerp(G(i,201)*.4,.72*Math.cos(a)+G(i,205)*.015,t);o.y=lerp(G(i,203)*.4,.72*Math.sin(a)+G(i,207)*.015,t);
    o.c=t>.6?(n<33?BL:n<66?GD:FG):FT;o.r=.85;}});

F.forget=(mode)=>({p(i,s,o){const t=ease(s);
    if(i%2===0){const d=mode==='replay'?.18:1;o.x=-.5+G(i,211)*.13+d*(G(i,213)*.26*t+.42*t);o.y=.05+G(i,215)*.13+d*(G(i,217)*.26*t-.22*t);o.c=BL;o.a=1-.4*t*d;}
    else{o.x=lerp((H(i,219)-.5)*1.8,.5+G(i,221)*.13,t);o.y=lerp((H(i,223)-.5)*1.4,.05+G(i,225)*.13,t);o.c=GD;}o.r=.85;},
  draw(s,a){circ(-.5,.05,.3,BL,a*.4,[4,5]);circ(.5,.05,.3,GD,a*.4,[4,5]);text('old skill',-.5,-.42,a,{c:BL,size:.052});text('new skill',.5,-.42,a,{c:GD,size:.052});}});

const MZ=(()=>{let x=-.9,y=.42;const P=[[x,y]],D=[];for(let k=0;k<40;k++){const th=H(k,231)*TAU;const dx=Math.cos(th),dy=Math.sin(th);const e=.08;
  const g=(bowlF(x+e*dx,y+e*dy)-bowlF(x-e*dx,y-e*dy))/(2*e);x-=.1*g*dx;y-=.1*g*dy;P.push([x,y]);D.push([dx,dy]);}return{P,D};})();
F.mezo=(mode)=>{let t=0,d=[1,0],e=.1,c=[0,0];return{
  prep(s){t=mode==='probe'?0:ease(s)*40;d=MZ.D[mode==='probe'?Math.floor(s*4)%40:Math.min(39,Math.floor(t))];e=mode==='probe'?.06+.16*Math.abs(Math.sin(s*Math.PI*4)):.1;c=along(MZ.P,t);},
  p(i,s,o){if(i===0){o.x=c[0];o.y=c[1];o.c=FG;o.r=2.6;return;}
    if(i<3){const g=i===1?1:-1;o.x=c[0]+g*e*d[0];o.y=c[1]+g*e*d[1];o.c=i===1?BL:GD;o.r=2;return;}
    const [x,y,j]=ring(i);o.x=x;o.y=y;o.c=FT;o.r=.8;o.a=.45+.55*(1-j/7);},
  draw(s,a){line(c[0]-d[0]*.5,c[1]-d[1]*.5,c[0]+d[0]*.5,c[1]+d[1]*.5,FG,a*.4,1,[4,4]);if(mode==='step')poly(MZ.P,t,GD,a*.8,2);
    text('+ε',c[0]+d[0]*(e+.1),c[1]+d[1]*(e+.1),a,{c:BL,size:.05});text('−ε',c[0]-d[0]*(e+.1),c[1]-d[1]*(e+.1),a,{c:GD,size:.05});}};};

F.state=()=>({p(i,s,o){const t=ease(s);
    if(i<210){const row=Math.floor(i/15),col=i%15;const rows=1+Math.floor(t*13.99);o.x=-.92+col*.032;o.y=-.66+row*.1;o.c=BL;o.r=.8;if(row>=rows){o.a=0;o.y=-.66+rows*.1;}}
    else{const k=i-210;const col=k%14,row=Math.floor(k/14);o.x=.3+col*.042;o.y=-.66+row*.042;o.c=GD;o.r=.8;o.a=.55+.45*Math.sin(k*.9+s*14);}},
  draw(s,a){text('cache',-.7,-.8,a,{c:BL,size:.055});text('state',.57,-.8,a,{c:GD,size:.055});
    {const n=1+Math.floor(ease(s)*13.99);text(n+(n>1?' tokens':' token')+' kept',-.7,.82,a,{size:.048});}text('same size',.57,.08,a,{size:.048});}});

const DK=[],DV=[];for(let w=0;w<5;w++){const k=[],v=[];for(let j=0;j<20;j++){k.push(G(w*20+j,241));v.push(G(w*20+j,243));}const nk=Math.hypot(...k);DK.push(k.map(a=>a/nk));DV.push(v);}DK[4]=DK[1];
F.delta=()=>{const S=new Float32Array(400);let mx=1,w=0;return{
  prep(s){S.fill(0);w=ease(s)*5;for(let q=0;q<5;q++){const f=clamp(w-q);if(f<=0)break;const k=DK[q],v=DV[q];
      for(let r=0;r<20;r++){let sk=0;for(let c=0;c<20;c++)sk+=S[r*20+c]*k[c];const e=(v[r]-sk)*f;for(let c=0;c<20;c++)S[r*20+c]+=e*k[c];}}
    mx=0;for(let i=0;i<400;i++)mx=Math.max(mx,Math.abs(S[i]));mx=mx||1;},
  p(i,s,o){if(i>=400){o.a=0;return;}const r=Math.floor(i/20),c=i%20;o.x=(c-9.5)*.09;o.y=(9.5-r)*.09;const v=S[i]/mx;o.r=.3+2*Math.pow(Math.abs(v),.7);o.c=v>0?BL:GD;o.a=.3+.7*Math.abs(v);},
  draw(s,a){const n=Math.min(5,Math.ceil(w-.001));text(n>0?'write '+n+' of 5':'empty memory',0,1.06,a,{size:.055});if(w>4.2)text('same key as write 2: old value replaced',0,-1.06,a,{c:GD,size:.048});}};};

const PF=j=>.35-j*.17, PX=j=>-.75+j*.3;
function envAt(c){let m=9;for(let j=0;j<6;j++){const c0=-.95+j*.2;if(c>=c0)m=Math.min(m,PF(j)+(.75-PF(j))*Math.exp(-(c-c0)*3));}return m;}
F.power=(mode)=>({p(i,s,o){const j=i%6,k=Math.floor(i/6);const u=k/69;o.c=j<2?FT:(j<4?FG:BL);o.r=.85;
    if(mode==='envelope'){const c0=-.95+j*.2;const c=c0+u*(.9-c0);const y=PF(j)+(.75-PF(j))*Math.exp(-(c-c0)*3);o.x=c;o.y=y;o.c=(y-envAt(c)<.015)?GD:FT;return;}
    const x=-.9+u*1.8,y=PF(j)+(.7-PF(j))*Math.exp(-u*4.2);
    if(mode==='curves'){const rv=ease(s);o.x=x;o.y=y;if(u>rv){o.a=0;o.x=-.9+rv*1.8;o.y=PF(j)+(.7-PF(j))*Math.exp(-rv*4.2);}}
    else{const t=ease(s);o.x=lerp(x,PX(j)+G(i,261)*.025,t);o.y=lerp(y,PF(j)+G(i,263)*.025,t);}},
  draw(s,a){if(mode==='curves')axes('data seen (log) →','loss (log)',a);
    else if(mode==='line'){axes('model size (log) →','loss (log)',a);const t=ease(s);line(PX(-.4),PF(-.4),PX(5.4),PF(5.4),GD,a*t,2.5);}
    else{axes('compute (log) →','loss (log)',a);const P=[];for(let c=-.95;c<=.9;c+=.02)P.push([c,envAt(c)]);poly(P,P.length,GD,a*.9,2.5);}}});

const IX=[],IY=[];for(let j=0;j<24;j++){const x=(H(j,251)-.5)*1.8;IX.push(x);IY.push(.55*x-.1+G(j,253)*.09);}
function fitK(k){if(k<1)return[0,0];if(k<2)return[0,IY[0]];let sx=0,sy=0,sxx=0,sxy=0;for(let j=0;j<k;j++){sx+=IX[j];sy+=IY[j];sxx+=IX[j]*IX[j];sxy+=IX[j]*IY[j];}const d=k*sxx-sx*sx;const m=(k*sxy-sx*sy)/d;return[m,(sy-m*sx)/k];}
F.icl=(k0,k1)=>{let kf=0,m=0,b=0;return{
  prep(s){kf=lerp(k0,k1,ease(s));const a=Math.floor(kf),f=kf-a;const A=fitK(a),B=fitK(Math.min(24,a+1));m=lerp(A[0],B[0],f);b=lerp(A[1],B[1],f);},
  p(i,s,o){if(i<24){o.x=IX[i];o.y=IY[i];o.c=GD;const v=clamp(kf-i);o.r=.5+1.7*v;o.a=v;return;}const u=(i-24)/395;const x=-.95+u*1.9;o.x=x;o.y=m*x+b;o.c=BL;o.r=.6;o.a=.85;},
  draw(s,a){text(Math.round(kf)+(Math.round(kf)===1?' example':' examples')+' in the prompt',-.95,.92,a,{align:'left',size:.052});}};};

const CE=[.25,.25,.7,.25,-.45,.25,-.78,.25],CHA={'.25':'42','.7':'48','-.45':'36','-.78':'40'};
F.chains=(mode)=>({p(i,s,o){if(i>=400){o.a=0;return;}const j=i%8,k=Math.floor(i/8);const t=k/49;const rv=mode==='think'?ease(s):1;
    const yE=CE[j];const yf=u=>lerp(0,yE,sstep(0,1,u))+.1*Math.sin(u*Math.PI*3+j*1.3)*Math.sin(Math.PI*u)*(j%2?1:-1);
    o.x=lerp(-.9,.8,t);o.y=yf(t);o.r=.75;
    if(mode==='think'){o.c=FG;if(t>rv){o.a=0;o.x=lerp(-.9,.8,rv);o.y=yf(rv);}else if(rv-t<.05)o.c=BL;}
    else{o.c=yE===.25?BL:GD;if(k===49)o.r=2;}},
  draw(s,a){text('question',-.9,.14,a,{size:.05});
    if(mode==='vote'){[.25,.7,-.45,-.78].forEach(y=>text(CHA[String(y).replace('0.','.')],.92,y,a,{c:y===.25?BL:GD,size:.065,align:'left'}));circ(.8,.25,.09,BL,a*.8);text('5 of 8 agree',.8,.42,a,{c:BL,size:.05});}}});

const PSC=[],PHO=[];for(let i=0;i<N;i++){const g=Math.floor(i/60);const sc=clamp(.12+g*.1+G(i,321)*.07,0,.98);PSC.push(sc);PHO.push(clamp(.25+.5*sc+G(i,323)*.1,0,.98));}
const PBG=[];for(let g=0;g<7;g++){let b=g*60;for(let i=g*60;i<g*60+60;i++)if(PSC[i]>PSC[b])b=i;PBG.push(b);}
let PBEST=0,PBHO=0;for(let i=0;i<N;i++){if(PSC[i]>PSC[PBEST])PBEST=i;if(PHO[i]>PHO[PBHO])PBHO=i;}
F.prompts=(mode)=>({p(i,s,o){const g=Math.floor(i/60);
    if(mode==='search'){const gv=ease(s)*6.99;o.x=-.9+PSC[i]*1.8;o.y=-.7+g*.22+(H(i,325)-.5)*.08;const best=PBG[g]===i;o.c=best?BL:FT;o.r=best?1.8:.8;o.a=g>gv?0:(g>gv-1?1:.55);if(g>gv){o.x=-.9;}}
    else{o.x=-.9+PSC[i]*1.8;o.y=-.65+PHO[i]*1.4;o.c=i===PBEST?GD:(i===PBHO?BL:FT);o.r=(i===PBEST||i===PBHO)?2.2:.75;}},
  draw(s,a){if(mode==='search'){axes('score on 60 questions →',null,a);const gv=Math.floor(ease(s)*6.99);text('round '+(gv+1),-.95,.85,a,{align:'left',size:.052});}
    else{axes('score on the 60 →','score on fresh questions',a);line(-.9,-.65,.86,.72,FT,a*.8,1,[4,5]);
      text('best on the 60',-.9+PSC[PBEST]*1.8,-.65+PHO[PBEST]*1.4-.1,a,{c:GD,size:.048,align:'right'});}}});

const PS=[];for(let j=0;j<14;j++)PS.push(.15+.75*H(j,261));PS[9]=1;
const PRANK=[...Array(14).keys()].sort((a,b)=>PS[b]-PS[a]);const PSLOT=new Array(14);PRANK.forEach((j,r)=>PSLOT[j]=r);
F.passages=(mode)=>({p(i,s,o){const j=i%14,k=Math.floor(i/14);const col=k%15,row=Math.floor(k/15);const t=ease(s);const top=PSLOT[j]===0;
    const slot=mode==='rank'?lerp(j,PSLOT[j],t):PSLOT[j];const len=mode==='rank'?lerp(.55,.3+.55*PS[j],t):.3+.55*PS[j];
    let x=-.15+col/14*len,y=.8-slot*.123+(row-.5)*.035;o.c=(mode==='rank'&&t<.3)?FT:(top?BL:FT);o.r=.8;o.a=top?1:.75;
    if(mode==='pull'){if(top){x=lerp(x,-.55+col/14*1.1,t);y=lerp(y,-.1+(row-.5)*.05,t);o.r=1.15;}else{o.a=.75*(1-.8*t);x+=.3*t;}}
    o.x=x;o.y=y;},
  draw(s,a){if(mode==='rank'){rect(-.98,.12,-.52,.32,FG,a*.5);text('question',-.75,.22,a,{c:FG,size:.05});for(let j=0;j<14;j++){const sl=lerp(j,PSLOT[j],ease(s));line(-.52,.22,-.17,.8-sl*.123,FT,a*(.2+.6*PS[j]*ease(s)),1);}}
    else text('pasted into the prompt',0,-.3,a*sstep(.3,.9,s),{c:BL,size:.052});}});

const QW=[];for(let i=0;i<N;i++)QW.push(clamp(G(i,271)*.3,-.93,.93));
function stackPos(binOf,centerOf,cols,dx){const cnt={};return QW.map(x=>{const b=binOf(x);const k=cnt[b]||0;cnt[b]=k+1;const col=k%cols,row=Math.floor(k/cols);return[centerOf(b)+(col-(cols-1)/2)*dx,-.62+row*.026];});}
const QMAX=Math.max(...QW.map(Math.abs)),QMEAN=QW.reduce((a,x)=>a+Math.abs(x),0)/N;
const QH=stackPos(x=>Math.round(x/.035),b=>b*.035,1,0);
const Q16=stackPos(x=>Math.round((x+QMAX)/(2*QMAX)*15),l=>-QMAX+l*(2*QMAX/15),3,.02);
const Q3L=x=>Math.abs(x)<QMEAN*.7?0:(x>0?1:-1);const Q3=stackPos(Q3L,l=>l*QMEAN*2.2,10,.02);
const QTHR=[...QW].map(Math.abs).sort((a,b)=>a-b)[42];
F.quant=(mode)=>({p(i,s,o){const t=ease(s);let A=QH[i],B=QH[i];if(mode==='q16'){B=Q16[i];}if(mode==='q3'){A=Q16[i];B=Q3[i];}
    o.x=lerp(A[0],B[0],t);o.y=lerp(A[1],B[1],t);o.r=.85;
    if(mode==='hist')o.c=FG;else if(mode==='q16')o.c=t>.5?BL:FG;else if(mode==='q3'){const l=Q3L(QW[i]);o.c=t>.5?(l>0?BL:l<0?GD:FT):BL;}
    else{if(Math.abs(QW[i])<QTHR){o.c=GD;o.a=1-t;o.y-=.2*t;}else o.c=BL;}},
  draw(s,a){line(-1,-.66,1,-.66,FG,a*.35,1.2);
    if(mode==='q16'&&s>.3){for(let l=0;l<16;l++){const x=-QMAX+l*(2*QMAX/15);line(x,-.66,x,-.7,FG,a*.6,1);}text('16 levels',0,-.82,a,{size:.052});}
    else if(mode==='q3'){[-1,0,1].forEach(l=>text(String(l).replace('-','−'),l*QMEAN*2.2,-.78,a,{c:FG,size:.06}));}
    else text(mode==='prune'?'smallest tenth removed':'weight value: 0 in the middle',0,-.82,a,{size:.05});}});

/* Module 24: five arms paying with probabilities 0.2, 0.4, 0.5, 0.6 and 0.7, in shuffled order. */
const BP=[.4,.2,.7,.5,.6],BPULL=[],BWIN=[],BIDX=[];
(()=>{const n=[0,0,0,0,0],w=[0,0,0,0,0];for(let t=0;t<N;t++){let a=0;if(t<5)a=t;else{let best=-1;for(let j=0;j<5;j++){const u=w[j]/n[j]+Math.sqrt(1.2*Math.log(t+1)/n[j]);if(u>best){best=u;a=j;}}}
  const win=H(t,281)<BP[a];BIDX.push(n[a]);n[a]++;if(win)w[a]++;BPULL.push(a);BWIN.push(win);}})();
F.bandit=(n0,n1)=>{let n=0;return{
  prep(s){n=lerp(n0,n1,ease(s));},
  p(i,s,o){const a=BPULL[i],k=BIDX[i];const ax=-.72+a*.36;const col=k%8,row=Math.floor(k/8);const x=ax+(col-3.5)*.032,y=-.68+row*.032;
    const f=clamp((n-i)/6);o.x=x;o.y=lerp(1.0,y,ease(f));o.a=f;o.c=BWIN[i]?BL:FT;o.r=.95;},
  draw(s,a){['A','B','C','D','E'].forEach((l,j)=>text(l,-.72+j*.36,-.8,a,{c:FG,size:.062}));text(Math.round(n)+' pulls',.95,.92,a,{align:'right',size:.052});}};};

F.policy=()=>{let p=[];return{
  prep(s){const t=ease(s)*3.2;p=softmax([0,.2,-.1,.3,0,-.2,1.2,.1,-.3].map(v=>v*t),1);},
  p(i,s,o){const j=i%9,k=Math.floor(i/9);const h=.12+1.6*p[j];const col=k%2,row=Math.floor(k/2);o.x=(j-4)*.2+(col-.5)*.04;o.y=-.62+(row/22)*h;o.c=j===6?BL:FT;o.r=.85;},
  draw(s,a){text('nine possible actions',0,-.76,a,{size:.052});}};};

const GW=(()=>{const wall=new Set();for(let r=2;r<=5;r++)wall.add(r*8+4);const start=[5,1],goal=[2,6];const mv=[[0,1],[0,-1],[1,0],[-1,0]];
  const step=(p,a)=>{const r=p[0]+mv[a][0],c=p[1]+mv[a][1];if(r<0||r>7||c<0||c>7||wall.has(r*8+c))return p;return[r,c];};
  const dist=p=>Math.hypot(p[0]-goal[0],p[1]-goal[1]);const A=[start],RO=[],BEST=[];let cur=start,seed=1;
  for(let t=0;t<24;t++){const ros=[];let bi=0,bs=1e9;for(let q=0;q<56;q++){let p=cur,sc=0;const path=[];for(let h=0;h<6;h++){p=step(p,Math.floor(H(seed++,301)*4));path.push(p);sc+=dist(p);}if(sc<bs){bs=sc;bi=q;}ros.push(path);}
    RO.push(ros);BEST.push(bi);cur=ros[bi][0];A.push(cur);if(cur[0]===goal[0]&&cur[1]===goal[1])break;}
  return{wall,start,goal,A,RO,BEST};})();
const gx=c=>(c-3.5)*.22, gy=r=>(3.5-r)*.22;
F.grid8=(mode)=>{let t=0,ti=0;return{
  prep(s){t=mode==='plan'?ease(s)*(GW.A.length-1):0;ti=Math.min(GW.RO.length-1,Math.floor(t));},
  p(i,s,o){if(i<64){const r=Math.floor(i/8),c=i%8;o.x=gx(c);o.y=gy(r);if(GW.wall.has(i)){o.c=FG;o.r=2.4;}else if(r===GW.goal[0]&&c===GW.goal[1]){o.c=BL;o.r=2.6;}else{o.c=FT;o.r=1;}return;}
    const ag=along(GW.A.map(p=>[gx(p[1]),gy(p[0])]),t);
    if(i<400){const q=i-64,ro=q%56,st=Math.floor(q/56);if(mode!=='plan'){o.x=ag[0];o.y=ag[1];o.a=0;return;}
      const p=GW.RO[ti][ro][st];o.x=gx(p[1])+(H(ro,307)-.5)*.07;o.y=gy(p[0])+(H(ro,309)-.5)*.07;const best=ro===GW.BEST[ti];o.c=best?BL:FT;o.r=best?1.3:.8;o.a=best?1:.35*(1-st/7);return;}
    const k=i-400;if(k===0){o.x=ag[0];o.y=ag[1];o.c=GD;o.r=2.8;return;}const tt=Math.max(0,t-k*.5);const q=along(GW.A.map(p=>[gx(p[1]),gy(p[0])]),tt);o.x=q[0];o.y=q[1];o.c=GD;o.r=.9;o.a=t>k*.5?.6*(1-k/20):0;},
  draw(s,a){if(mode==='plan'){const P=[[gx(GW.A[ti][1]),gy(GW.A[ti][0])]].concat(GW.RO[ti][GW.BEST[ti]].map(p=>[gx(p[1]),gy(p[0])]));poly(P,P.length,BL,a*.7,2);}
    text('goal',gx(GW.goal[1]),gy(GW.goal[0])+.14,a,{c:BL,size:.05});}};};

const CR=[1,0,1,1,0,1,0,1];
F.checker=(mode)=>{let R=CR,mean=.625,show=0;return{
  prep(s){if(mode==='grade'){R=CR;show=ease(s);}else{const w=Math.floor(s*9);R=CR.map((_,j)=>H(j,331+w*3)<.5?1:0);show=1;}mean=R.reduce((a,b)=>a+b,0)/8;},
  p(i,s,o){if(i>=416){o.a=0;return;}const j=i%8,k=Math.floor(i/8);const col=k%4,row=Math.floor(k/4);const top=lerp(.45,(R[j]-mean)*1.1,show);
    o.x=(j-3.5)*.22+(col-1.5)*.032;o.y=(row/12)*top+(top>=0?.02:-.02);o.c=show<.5?FT:(R[j]?BL:GD);o.r=.85;},
  draw(s,a){line(-.95,0,.95,0,FG,a*.4,1.2,[4,4]);text('group mean',-.95,.07,a,{align:'left',size:.045});
    if(show>.5)for(let j=0;j<8;j++){const top=lerp(.45,(R[j]-mean)*1.1,show);text(R[j]?'1':'0',(j-3.5)*.22,top+(top>=0?.12:-.12),a*sstep(.5,1,show),{c:R[j]?BL:GD,size:.06});}
    if(mode==='noise')text('59.1% → 4.6%',0,-.94,a*sstep(.25,.6,s),{c:GD,size:.11,disp:true,weight:700});}};};

const EVf=x=>.6*Math.exp(-Math.pow((x-.55)/.17,2))+.32*Math.exp(-Math.pow((x+.45)/.22,2))+.06*(1+Math.cos(x*11));
const EVY=x=>-.55+1.1*EVf(x)/.75;
const EVX=(()=>{const X=[];for(let k=0;k<20;k++)X.push(-.95+H(k,311)*1.9);
  for(let g=1;g<21;g++){const prev=X.slice((g-1)*20,g*20);const w=prev.map(x=>Math.pow(EVf(x),3)+1e-4);const tot=w.reduce((a,b)=>a+b,0);
    for(let k=0;k<20;k++){let r=H(g*20+k,313)*tot,j=0;while(j<19&&r>w[j]){r-=w[j];j++;}X.push(clamp(prev[j]+G(g*20+k,317)*.12,-1,1));}}return X;})();
F.evo=(mode)=>{let g=0;return{
  prep(s){g=mode==='run'?ease(s)*20:20;},
  p(i,s,o){const gen=Math.floor(i/20);const x=EVX[i];const y=EVY(x)+(H(i,319)-.5)*.03;o.x=x;o.y=y;o.r=.95;
    if(mode==='run'){if(gen>g){o.a=0;return;}const cur=gen>g-1;o.c=cur?BL:FT;o.a=cur?1:Math.exp(-(g-gen)*.35)*.8+.1;}
    else{if(gen>=12&&i%2){const t=ease(s);o.y=lerp(y,.85+(H(i,327)-.5)*.1,t);o.c=GD;}else{o.c=gen>=18?BL:FT;o.a=gen>=12?1:.3;}}},
  draw(s,a){const P=[];for(let x=-1;x<=1.001;x+=.02)P.push([x,EVY(x)]);poly(P,P.length,FG,a*.35,1.5);
    if(mode==='run')text('generation '+Math.round(g),-.95,.92,a,{align:'left',size:.052});
    else{text('real score',.95,-.68,a,{align:'right',c:BL,size:.05});text('score it claims',.95,1.0,a*sstep(.2,.7,s),{align:'right',c:GD,size:.05});}}};};

F.motifs=()=>{const sub=[F.spiral('data'),F.graph('all'),F.rays('rotate'),F.chains('vote'),F.panels('tidy')];const L=['hide, predict','one local rule','lookup','propose, keep','controls'];const tmp={};let rot=0;return{
  prep(s){rot=ease(s)*.5;sub.forEach(f=>f.prep&&f.prep(.5));},
  p(i,s,o){const j=i%5;tmp.x=0;tmp.y=0;tmp.c=0;tmp.r=1;tmp.a=1;sub[j].p(i,.5,tmp);const ang=Math.PI/2-j*TAU/5+rot;o.x=.62*Math.cos(ang)+tmp.x*.25;o.y=.62*Math.sin(ang)+tmp.y*.25;o.c=tmp.c;o.r=tmp.r*.8;o.a=tmp.a;},
  draw(s,a){for(let j=0;j<5;j++){const ang=Math.PI/2-j*TAU/5+rot;text(L[j],.62*Math.cos(ang),.62*Math.sin(ang)-.32,a,{size:.045});}}};};

/* ---------- version 2 formations: n1–n36 ---------- */
function dots(P,rp,c,a){if(a<.01)return;ctx.save();ctx.fillStyle=rgba(c,a);ctx.beginPath();const r=Math.max(.8,V.R*rp);for(const q of P){const X=V.X(q[0]),Y=V.Y(q[1]);ctx.moveTo(X+r,Y);ctx.arc(X,Y,r,0,TAU);}ctx.fill();ctx.restore();}
function solveLin(M,b){const n=b.length,A=M.map((r,i)=>r.concat([b[i]]));
  for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;const t=A[c];A[c]=A[p];A[p]=t;const d=A[c][c]||1e-12;
    for(let r=c+1;r<n;r++){const f=A[r][c]/d;if(f)for(let k=c;k<=n;k++)A[r][k]-=f*A[c][k];}}
  const x=new Array(n).fill(0);for(let r=n-1;r>=0;r--){let s=A[r][n];for(let k=r+1;k<n;k++)s-=A[r][k]*x[k];x[r]=s/(A[r][r]||1e-12);}return x;}
/* Chebyshev basis keeps the 12×12 normal equations well conditioned on [-1,1]. */
function cheb(x,deg,out){out[0]=1;if(deg>0)out[1]=x;for(let k=2;k<=deg;k++)out[k]=2*x*out[k-1]-out[k-2];return out;}
const CHB=new Array(12);
function chebVal(c,x){cheb(x,c.length-1,CHB);let y=0;for(let k=0;k<c.length;k++)y+=c[k]*CHB[k];return y;}
function polyFit(X,Y,deg,lam){const n=deg+1,M=[],b=new Array(n).fill(0),r=new Array(n);for(let a=0;a<n;a++)M.push(new Array(n).fill(0));
  for(let j=0;j<X.length;j++){cheb(X[j],deg,r);for(let a=0;a<n;a++){b[a]+=r[a]*Y[j];for(let c=0;c<n;c++)M[a][c]+=r[a]*r[c];}}
  for(let a=1;a<n;a++)M[a][a]+=lam+1e-12;return solveLin(M,b);}
const PTRUE=x=>.42*Math.sin(2.6*x)+.12*x+.05;
const PFX=[],PFY=[],PTX=[],PTY=[];for(let j=0;j<12;j++){PFX.push(.88*(-.5*Math.cos(Math.PI*(j+.5)/12)+.5*(-1+2*j/11)));PFY.push(PTRUE(PFX[j])+G(j,823)*.2);}
for(let j=0;j<11;j++){const x=(PFX[j]+PFX[j+1])/2+(H(j,405)-.5)*.03;PTX.push(x);PTY.push(PTRUE(x)+G(j,407)*.06);}
const PFC=[null];for(let d=1;d<=11;d++)PFC.push(polyFit(PFX,PFY,d,0));
F.polyfit=(mode)=>{let c0=PFC[1],c1=PFC[1],f=0,deg=1,lam=0;return{
  prep(s){if(mode==='degree'){deg=lerp(1,11,ease(s));const d=Math.min(10,Math.floor(deg));c0=PFC[d];c1=PFC[d+1];f=deg-d;}
    else if(mode==='penalty'){lam=Math.pow(10,lerp(-7,-.6,ease(s)));c0=c1=polyFit(PFX,PFY,11,lam);f=0;}
    else{c0=c1=PFC[11];f=0;}},
  y(x){const a=chebVal(c0,x);return f?lerp(a,chebVal(c1,x),f):a;},
  p(i,s,o){if(i<12){o.x=PFX[i];o.y=PFY[i];o.c=GD;o.r=2.4;return;}
    if(i<23){const k=i-12;o.x=PTX[k];o.y=PTY[k];o.c=BL;o.r=2;o.a=mode==='test'?clamp(s*3.2-k*.18):0;return;}
    const u=(i-23)/396,x=lerp(PFX[0]-.02,PFX[11]+.02,u);o.x=x;o.y=this.y(x);o.c=FG;o.r=.75;o.a=.85;},
  draw(s,a){if(mode==='degree')text('degree '+Math.round(deg),-.95,.92,a,{align:'left',size:.052});
    if(mode==='test'){for(let k=0;k<11;k++){const al=clamp(s*3.2-k*.18);if(al<.02)continue;line(PTX[k],PTY[k],PTX[k],this.y(PTX[k]),GD,a*al*.9,1.5,[3,3]);}
      text('new points',.95,.92,a*clamp(s*3),{c:BL,align:'right',size:.052});}
    if(mode==='penalty'){const t=ease(s);text('weight penalty',-.95,.92,a,{align:'left',size:.048});rect(-.95,.76,-.25,.8,FT,a*.8,true);rect(-.95,.76,-.95+.7*Math.max(.02,t),.8,GD,a,true);}}};};

/* Module 02: validation loss lowest at epoch 17 (0.360); train loss ends at 0.216 (epoch 25). */
const LTR=e=>.216+.38*(Math.exp(-(e-1)/6)-Math.exp(-4))/(1-Math.exp(-4));
const LVA=e=>e>=17?.36+.051*((e-17)/8)**2:.36+.25*((17-e)/16)**2;
const LSX=e=>-.9+(e-1)/24*1.8, LSY=v=>-.6+1.2*(v-.15)/.55;
F.losses=(mode)=>{let em=1;return{
  prep(s){em=lerp(1,17,ease(s));},
  p(i,s,o){if(i===0){if(mode==='stop'){o.x=LSX(em);o.y=LSY(LVA(em));o.c=GD;o.r=3;}else{o.x=LSX(1);o.y=LSY(LVA(1));o.a=0;}return;}
    const cur=i%2,k=Math.floor(i/2),u=k/209,e=1+24*u;const rv=mode==='reveal'?ease(s):1;
    o.x=LSX(e);o.y=LSY(cur?LTR(e):LVA(e));o.c=cur?BL:GD;o.r=.85;
    if(mode==='reveal'&&u>rv){const er=1+24*rv;o.a=0;o.x=LSX(er);o.y=LSY(cur?LTR(er):LVA(er));}
    if(mode==='stop'&&e>17)o.a=1-.65*sstep(.6,1,s);},
  draw(s,a){axes('epochs →','loss',a);const rv=mode==='reveal'?ease(s):1,er=1+24*rv,al=rv>.7?'right':'left';
    text('validation',LSX(er),LSY(LVA(er))+.1,a,{c:GD,size:.052,align:al});text(mode==='stop'?'train 0.216':'train',LSX(er),LSY(LTR(er))-.1,a,{c:BL,size:.052,align:al});
    if(mode==='stop'){const b=sstep(.55,.9,s);line(LSX(17),-.68,LSX(17),LSY(.36),GD,a*b*.7,1.2,[4,4]);text('lowest: epoch 17, 0.360',LSX(17),LSY(.36)-.12,a*b,{c:GD,size:.05});}}};};

F.ucurve=()=>{const TE=u=>.2+.6*(u<.4?((.4-u)/.4)**2:((u-.4)/.6)**2),TR=u=>.03+.75*Math.exp(-3.2*u);return{
  p(i,s,o){const cur=i%2,u=(Math.floor(i/2))/209,rv=ease(s);const f=cur?TR:TE;o.x=-.9+u*1.8;o.y=-.6+1.2*f(u);o.c=cur?BL:GD;o.r=.85;
    if(u>rv){o.a=0;o.x=-.9+rv*1.8;o.y=-.6+1.2*f(rv);}},
  draw(s,a){axes('model size →','error',a);const rv=ease(s),al=rv>.7?'right':'left';
    text('test',-.9+rv*1.8,-.6+1.2*TE(rv)+.1,a,{c:GD,size:.052,align:al});text('train',-.9+rv*1.8,-.6+1.2*TR(rv)-.1,a,{c:BL,size:.052,align:al});
    const b=sstep(.45,.8,s);if(b>0){const x=-.9+.4*1.8,y=-.6+1.2*.2;circ(x,y,.06,GD,a*b);line(x,-.68,x,y-.06,GD,a*b*.6,1.2,[4,4]);text('best size',x,y-.16,a*b,{c:GD,size:.05});}}};};

/* Module 00c: test MSE of minimum-norm polynomial fits to n = 20 points (one dataset). */
const DDT=[[3,.0183],[10,.3309],[18,4.38e6],[19,1.11e12],[20,5.57e7],[25,37.95],[30,3.019],[47,.4825],[80,.5991]];
const DDX=d=>-.9+(d-3)/77*1.8, DDY=m=>-.6+1.3*(Math.log10(m)+2)/14.5;
function ddAt(d){let k=0;while(k<DDT.length-2&&d>DDT[k+1][0])k++;const [d0,m0]=DDT[k],[d1,m1]=DDT[k+1];const t=sstep(0,1,(d-d0)/(d1-d0));return lerp(DDY(m0),DDY(m1),t);}
F.dd=(mode)=>{let dm=20;return{
  prep(s){dm=lerp(20,47,ease(s));},
  p(i,s,o){if(i===0){if(mode==='floor'){o.x=DDX(dm);o.y=ddAt(dm);o.c=GD;o.r=3;}else o.a=0;return;}
    const u=(i-1)/418,d=3+77*u,rv=mode==='reveal'?ease(s):1;o.x=DDX(d);o.y=ddAt(d);o.c=BL;o.r=.85;
    if(mode==='floor'&&d>19&&d<=dm)o.c=GD;
    if(u>rv){const dr=3+77*rv;o.a=0;o.x=DDX(dr);o.y=ddAt(dr);}},
  draw(s,a){axes('polynomial degree →','test error (log)',a);const rv=mode==='reveal'?ease(s):1,dr=3+77*rv;
    const sp=mode==='reveal'?sstep(.18,.26,rv):1;if(sp>0){line(DDX(19),-.68,DDX(19),DDY(1.11e12),FG,a*sp*.4,1,[4,4]);text('degree 19: as many weights as points',0,.94,a*sp,{size:.045});}
    if(mode==='reveal'&&rv>.05)circ(DDX(dr),ddAt(dr),.035,BL,a);
    if(mode==='floor'){line(-.9,DDY(.0183),.9,DDY(.0183),BL,a*.6,1.2,[5,5]);text('0.0183 at degree 3',-.9,DDY(.0183)-.08,a,{c:BL,align:'left',size:.046});
      const b=sstep(.6,.95,s);text('0.4825 at degree 47',DDX(47),DDY(.4825)+.1,a*b,{c:GD,size:.046});}}};};

/* Minimum-norm interpolation: every curve passes the 5 points; blue has the smallest coefficients. */
const IPX=[-.78,-.38,.02,.4,.8],IPY=IPX.map((x,j)=>PTRUE(x)*.9+G(j,411)*.08);
const IPN=(()=>{const D=8,A=IPX.map(x=>{const r=[];for(let k=0;k<D;k++)r.push(Math.pow(x,k));return r;});
  const M=A.map(r=>A.map(q=>r.reduce((s,v,k)=>s+v*q[k],0)));const z=solveLin(M,IPY);const cmin=new Array(D).fill(0);A.forEach((r,j)=>r.forEach((v,k)=>cmin[k]+=v*z[j]));
  let q=[1];IPX.forEach(x0=>{const n=new Array(q.length+1).fill(0);q.forEach((v,k)=>{n[k+1]+=v;n[k]-=x0*v;});q=n;});
  const nulls=[0,1,2].map(m=>{const c=new Array(D).fill(0);q.forEach((v,k)=>c[k+m]+=v);let mx=0;for(let x=-.95;x<=.95;x+=.01){let y=0;c.forEach((v,k)=>y+=v*Math.pow(x,k));mx=Math.max(mx,Math.abs(y));}return c.map(v=>v/mx);});
  return{cmin,nulls};})();
function ipVal(c,x){let y=0,p=1;for(let k=0;k<c.length;k++){y+=c[k]*p;p*=x;}return y;}
F.interp=()=>{const C=[IPN.cmin,[],[],[]];return{
  prep(s){for(let k=1;k<4;k++){const am=(.25+.08*k)*Math.sin(s*Math.PI*1.6+k*2.1)+(k%2?.2:-.2);C[k]=IPN.cmin.map((v,j)=>v+am*IPN.nulls[k-1][j]);}},
  p(i,s,o){if(i<5){o.x=IPX[i];o.y=IPY[i];o.c=GD;o.r=2.6;return;}const k=(i-5)%4,u=Math.floor((i-5)/4)/103,x=-.84+1.7*u;o.x=x;o.y=ipVal(C[k],x);
    if(k===0){o.c=BL;o.r=1;}else{o.c=FG;o.r=.65;o.a=.45;}},
  draw(s,a){text('smallest weights',.95,.92,a,{c:BL,align:'right',size:.05});text('all pass through every point',-.95,-.98,a,{align:'left',size:.045});}};};

/* Splits. */
F.split=()=>({p(i,s,o){const t=ease(s);const a0=H(i,1)*TAU,r0=Math.sqrt(H(i,2))*1.02;let tx,ty;
    if(i<294){tx=-.95+(i%14)*.083;ty=.6-Math.floor(i/14)*.06;o.c=BL;}else if(i<357){const k=i-294;tx=.3+(k%3)*.083;ty=.6-Math.floor(k/3)*.06;o.c=GD;}else{const k=i-357;tx=.72+(k%3)*.083;ty=.6-Math.floor(k/3)*.06;o.c=FG;}
    o.x=lerp(r0*Math.cos(a0),tx,t);o.y=lerp(r0*Math.sin(a0),ty,t);if(t<.3)o.c=FT;o.r=.85;},
  draw(s,a){const b=sstep(.5,.9,s);text('train',-.41,.76,a*b,{c:BL,size:.05});text('validation',.38,.76,a*b,{c:GD,size:.05});text('test',.8,.76,a*b,{c:FG,size:.05});}});

const CLS=[];{let l=0,r=0;for(let g=0;g<70;g++){const left=g%10<7;const k=left?l++:r++;CLS.push({cx:(H(g,421)-.5)*1.6,cy:(H(g,423)-.5)*1.2,tx:left?-.85+(k%7)*.17:.5+(k%3)*.19,ty:.6-Math.floor(k/(left?7:3))*.19,left});}}
F.clumps=()=>({p(i,s,o){const g=Math.floor(i/6),m=i%6,C=CLS[g];const t=ease(clamp((s-H(g,425)*.25)/.75));const ox=.04*Math.cos(m/6*TAU+g),oy=.04*Math.sin(m/6*TAU+g);
    o.x=lerp(C.cx,C.tx,t)+ox;o.y=lerp(C.cy,C.ty,t)+oy;o.c=C.left?BL:GD;o.r=.9;},
  draw(s,a){const b=sstep(.4,.9,s);line(.33,-.75,.33,.8,FG,a*b*.5,1.2,[4,4]);text('train',-.34,.84,a*b,{c:BL,size:.05});text('test',.69,.84,a*b,{c:GD,size:.05});text('one clump: one person',0,-.92,a,{size:.045});}});

F.leak=()=>({p(i,s,o){const lit=sstep(.15,.55,s);
    if(i>=300){const k=i-300;o.x=.15+H(i,431)*.8;o.y=(H(i,433)-.5)*1.1;o.c=BL;o.r=.9;if(k<12){o.c=lit>.5?GD:BL;o.r=.9+1.2*lit;}return;}
    if(i<12){const j=300+i;o.x=.15+H(j,431)*.8-1.1+G(i,435)*.01;o.y=(H(j,433)-.5)*1.1+.03;o.c=lit>.5?GD:FT;o.r=.9+1.2*lit;return;}
    o.x=-.95+H(i,437)*.8;o.y=(H(i,439)-.5)*1.1;o.c=FT;o.r=.85;},
  draw(s,a){const lit=sstep(.15,.55,s);line(.05,-.65,.05,.62,FG,a*.4,1.2,[4,4]);text('train',-.55,-.78,a,{size:.05});text('test',.55,-.78,a,{size:.05});
    for(let k=0;k<12;k++){const j=300+k;const x=.15+H(j,431)*.8,y=(H(j,433)-.5)*1.1;line(x,y,x-1.1,y+.03,GD,a*lit*.45,1,[2,4]);}
    const sc=lerp(.4,.92,sstep(.35,.85,s));text('test score',.15,.94,a,{align:'left',size:.048});rect(.15,.74,.95,.8,FT,a*.7,true);rect(.15,.74,.15+.8*sc,.8,sc>.7?GD:BL,a,true);}});

/* Confusion counts: 120 true positives, 40 false negatives, 60 false positives, 200 true negatives. */
const CFN=[120,40,60,200],CFC=[[-.42,.36],[.42,.36],[-.42,-.44],[.42,-.44]],CFL=['true positive','false negative','false positive','true negative'];
F.confusion=()=>({p(i,s,o){let k=0,j=i;while(j>=CFN[k]){j-=CFN[k];k++;}const n=CFN[k],cols=Math.ceil(Math.sqrt(n)),sp=.6/cols;const r=Math.floor(j/cols),c=j%cols;
    const tx=CFC[k][0]+(c-(cols-1)/2)*sp,ty=CFC[k][1]+.3-sp/2-r*sp;const a0=H(i,1)*TAU,r0=Math.sqrt(H(i,2))*1.02;const t=ease(s);
    o.x=lerp(r0*Math.cos(a0),tx,t);o.y=lerp(r0*Math.sin(a0),ty,t);o.c=t<.3?FT:(k===0||k===3?BL:GD);o.r=.85;},
  draw(s,a){const b=sstep(.5,.9,s);CFC.forEach((p,k)=>{rect(p[0]-.36,p[1]-.36,p[0]+.36,p[1]+.36,FG,a*b*.25);text(CFL[k],p[0],p[1]-.44,a*b,{c:k===0||k===3?BL:GD,size:.045});});
    text('predicted yes',-.42,.9,a*b,{size:.045});text('predicted no',.42,.9,a*b,{size:.045});}});

/* Module 05a: threshold 0.43 chosen on validation. */
const THS=[];for(let i=0;i<N;i++){const pos=i%2===0;THS.push(clamp((pos?.63:.37)+G(i,441)*.17,.02,.98));}
F.thresh=()=>{let th=.15;return{
  prep(s){th=s<.6?lerp(.15,.85,ease(s/.6)):lerp(.85,.43,ease((s-.6)/.4));},
  p(i,s,o){const pos=i%2===0,sc=THS[i];o.x=-.9+sc*1.8;o.y=(pos?.38:-.38)+(H(i,443)-.5)*.42;const flag=sc>=th;o.r=.9;
    if(flag){o.c=pos?BL:GD;}else{o.c=pos?FG:FT;o.a=pos?.55:.6;}},
  draw(s,a){const x=-.9+th*1.8;line(x,-.68,x,.7,FG,a*.75,2);text('threshold '+th.toFixed(2),x,.82,a,{c:FG,size:.05});
    line(-.95,-.72,.95,-.72,FG,a*.3,1.2);text('score →',.95,-.82,a,{align:'right',size:.048});text('positive',.95,.68,a,{align:'right',c:BL,size:.045});text('negative',.95,-.08,a,{align:'right',c:GD,size:.045});
    if(s>.92)text('chosen on validation',x,-.9,a*sstep(.92,1,s),{c:FG,size:.045});}};};

F.calib=()=>({p(i,s,o){const m=i%2,k=Math.floor(i/2),b=k%10,j=Math.floor(k/10);const c=.05+.1*b;const acc=m?clamp(.5+.45*(c-.5)-.06):clamp(c+G(b,451)*.025);
    const X=-.95+c*1.9,Y=-.68+acc*1.4;o.x=X+G(i,453)*.014+(m?.018:-.018);o.y=Y+G(i,455)*.018;o.c=m?GD:BL;o.r=.8;
    const rv=1+ease(s)*9;if(b>rv){o.a=0;}else if(b>rv-1)o.a=rv-b;},
  draw(s,a){axes('confidence →','accuracy',a);line(-.95,-.68,.95,.72,FG,a*.5,1,[4,5]);text('calibrated',.5,.55,a,{c:BL,size:.048});text('overconfident',.55,-.05,a*sstep(.4,.8,s),{c:GD,size:.048});}});

/* Module 00b: 1,000 people, prevalence 0.4% (4 sick), 96% sensitivity, 8% false positives: P(sick | flagged) = 4.6%. */
const RG=[];for(let k=0;k<1000;k++)RG.push([-.95+(k%40)*(1.9/39),.56-Math.floor(k/40)*(1.12/24)]);
const RSICK=[],RFP=[];{const used=new Set();let s=0;while(RSICK.length<4){const k=Math.floor(H(s++,461)*1000);if(!used.has(k)){used.add(k);RSICK.push(k);}}while(RFP.length<80){const k=Math.floor(H(s++,461)*1000);if(!used.has(k)){used.add(k);RFP.push(k);}}}
const RREST=RG.filter((_,k)=>!RSICK.includes(k)&&!RFP.includes(k));
F.rare=()=>({p(i,s,o){if(i<4){const q=RG[RSICK[i]];o.x=q[0];o.y=q[1];o.c=BL;o.r=2.4;return;}
    if(i<84){const k=i-4,q=RG[RFP[k]];o.x=q[0];o.y=q[1];o.c=GD;o.r=1.6;o.a=clamp(s*2.2-k/80*1.1);return;}const q=RREST[(i-84)*2];o.x=q[0];o.y=q[1];o.c=FT;o.r=.7;o.a=0;},
  draw(s,a){dots(RREST,.007,FT,a*.9);text('1,000 people',-.95,.72,a,{align:'left',size:.048});text('4 sick',.95,.72,a,{align:'right',c:BL,size:.048});
    text('flagged but healthy',-.95,-.72,a*sstep(.2,.6,s),{align:'left',c:GD,size:.048});text('4.6% of flags are sick',0,-.9,a*sstep(.6,.95,s),{c:FG,size:.055});}});

/* Module 05a: slice F1 at 0.5 with 95% bootstrap intervals. */
const BS=[[.806,.722,.872,'daylight',.32],[.727,.428,.909,'low-light',-.38]],BX2=f=>-.9+(f-.35)/.6*1.8;
F.boot=()=>({p(i,s,o){const r=i<210?0:1,[e,lo,hi,,y0]=BS[r];const g=G(i,471);const v=e+(g<0?g*(e-lo):g*(hi-e))/1.96;const t=ease(s);
    const t2=.15+.85*t;o.x=lerp(BX2(e),BX2(v),t2);o.y=y0+(H(i,473)-.5)*.24*t2;o.c=r?GD:BL;o.r=.85;o.a=.55+.45*(1-t);},
  draw(s,a){line(-.95,-.72,.95,-.72,FG,a*.3,1.2);text('F1 →',.95,-.82,a,{align:'right',size:.048});
    BS.forEach(([e,lo,hi,l,y0],r)=>{const c=r?GD:BL;text(l+' '+e.toFixed(3),-.95,y0+.24,a,{align:'left',c,size:.05});
      const b=sstep(.55,.9,s);line(BX2(lo),y0-.17,BX2(hi),y0-.17,c,a*b,2.5);line(BX2(lo),y0-.21,BX2(lo),y0-.13,c,a*b,2);line(BX2(hi),y0-.21,BX2(hi),y0-.13,c,a*b,2);});
    text('the intervals overlap',0,-.92,a*sstep(.75,1,s),{c:FG,size:.05});}});

F.vault=()=>{let open=0;return{
  prep(s){open=sstep(.72,.95,s);},
  p(i,s,o){if(i<300){const bx=-.95+H(i,481)*.95,by=(H(i,483)-.5)*1.1;o.x=bx+.04*Math.sin(s*14+i);o.y=by+.04*Math.cos(s*11+i*1.3);o.c=i%3?FT:BL;o.r=.85;return;}
    const k=i-300;o.x=.32+(k%10)*.068;o.y=.4-Math.floor(k/10)*.075;o.c=open>.5?BL:FT;o.r=.85+.4*open;o.a=.4+.6*open;},
  draw(s,a){rect(.26,-.5,.98,.5,FG,a*.6);const an=open*1.1;line(.26,.5,.26+.72*Math.cos(an),.5+.72*Math.sin(an),FG,a*.9,2.5);
    text('tune here',-.48,-.78,a,{size:.05});text('test set',.62,-.62,a,{size:.05});text('opened once',.62,-.78,a*open,{c:BL,size:.05});}};};

/* Weights. */
F.wgrid=()=>({p(i,s,o){if(i>=400){o.a=0;return;}const r=Math.floor(i/20),c=i%20;const t=ease(s);const w0=G(i,491)*.45,dw=G(i,493)*.4;const w=w0+t*dw;
    o.x=(c-9.5)*.09+t*.012*Math.sign(dw);o.y=(9.5-r)*.09-.05+t*.012*Math.sign(dw);o.r=.35+1.7*Math.min(1,Math.abs(w));o.c=w>0?BL:GD;o.a=.45+.55*Math.min(1,Math.abs(w));},
  draw(s,a){text('every weight changes',0,-1.0,a*sstep(.2,.6,s),{size:.05});}});

const LRU=[],LRV=[];for(let k=0;k<20;k++){LRU.push(G(k,501));LRV.push(G(k,503));}
F.lowrank=()=>({p(i,s,o){const t=ease(s);
    if(i<380){const r=Math.floor(i/20),c=i%20;const lo=LRU[r]*LRV[c]*.28,fu=G(i,505)*.45;const v=lerp(fu,lo,t);o.x=-.62+c*.078;o.y=.68-r*.078;o.r=.35+1.6*Math.min(1,Math.abs(v));o.c=v>0?BL:GD;o.a=.4+.6*Math.min(1,Math.abs(v));return;}
    if(i<399){const r=i-380,v=LRU[r]*.5;o.x=-.92;o.y=.68-r*.078;o.r=.4+1.4*Math.min(1,Math.abs(v));o.c=v>0?BL:GD;o.a=t;return;}
    if(i<419){const c=i-399,v=LRV[c]*.5;o.x=-.62+c*.078;o.y=.92;o.r=.4+1.4*Math.min(1,Math.abs(v));o.c=v>0?BL:GD;o.a=t;return;}o.a=0;},
  draw(s,a){text('change ≈ column × row',0,-.86,a*sstep(.4,.9,s),{size:.05});}});

/* Module 06: LoRA at rank 16 trains 8.7M parameters, 2.3% of the model. */
F.lora=(mode)=>({p(i,s,o){
    if(i<324){const r=Math.floor(i/18),c=i%18;o.x=-.95+c*.055;o.y=.47-r*.055;o.c=FT;o.r=1;o.a=1;
      if(mode==='zero'){const col=Math.floor(s*18.99);if(c===col){o.c=FG;o.a=.9;}}return;}
    if(i<360){const k=i-324,r=Math.floor(k/2),c=k%2;o.x=.2+c*.055;o.y=.47-r*.055;
      if(mode==='zero'){o.c=FT;o.r=.3;o.a=.5;}else{const v=G(k,511)+.6*Math.sin(s*7+k);o.c=v>0?BL:GD;o.r=.5+.9*Math.min(1,Math.abs(v)*.6);}return;}
    if(i<396){const k=i-360,r=Math.floor(k/18),c=k%18;o.x=.4+c*.033;o.y=.74-r*.055;const v=G(k,513)+.6*Math.sin(s*7+k*.7);o.c=v>0?BL:GD;o.r=.5+.9*Math.min(1,Math.abs(v)*.6);return;}
    o.a=0;o.x=0;o.y=0;},
  draw(s,a){text('W, frozen',-.48,-.62,a,{size:.05});text('B',.225,-.62,a,{c:BL,size:.055});text('A',.68,.9,a,{c:BL,size:.055});text('+',.09,0,a,{c:FG,size:.08});
    if(mode==='zero')text('B = 0, so B·A adds nothing',0,-.86,a,{c:GD,size:.05});
    else text('8.7M trainable, 2.3%, rank 16',0,-.86,a,{c:BL,size:.05});}});

/* Bar charts drawn with points. spec.bars: {x, v, v0?, c, top, lab}. */
F.vbars=(spec)=>{const nb=spec.bars.length,per=Math.floor(N/nb),cols=spec.cols||6;const rows=Math.ceil(per/cols);let hs=[];return{
  prep(s){const t=ease(s);hs=spec.bars.map(b=>lerp(b.v0===undefined?0:b.v0,b.v,t)/spec.max);},
  p(i,s,o){const b=Math.floor(i/per);if(b>=nb){o.a=0;o.x=0;o.y=-.6;return;}const k=i%per,c=k%cols,r=Math.floor(k/cols);const B0=spec.bars[b];
    o.x=B0.x+(c-(cols-1)/2)*.032;o.y=-.6+hs[b]*1.25*(r/(rows-1));o.c=B0.c;o.r=.85;},
  draw(s,a){line(-.95,-.64,.95,-.64,FG,a*.35,1.2);
    if(spec.chance!==undefined){const y=-.6+spec.chance/spec.max*1.25;line(-.95,y,.95,y,FG,a*.55,1.2,[5,5]);text('chance',.95,y+.07,a,{align:'right',size:.044});}
    spec.bars.forEach((B0,j)=>{text(B0.top,B0.x,-.6+hs[j]*1.25+.1,a*(B0.v0===B0.v?1:sstep(.6,.95,s)),{c:B0.c===FT?FG:B0.c,size:.05});if(B0.lab)text(B0.lab,B0.x,-.74,a,{size:.044});});
    (spec.groups||[]).forEach(g=>text(g[1],g[0],-.88,a,{c:FG,size:.048}));if(spec.head)text(spec.head,0,.92,a,{size:.048});}};};

/* Distillation. Module 23: after "window b" the teacher gives e 0.48, y 0.12, u 0.11; the rest share 0.29. */
const DTE=[.48,.12,.11,.29],DST=[.22,.3,.26,.22],DOH=[1,0,0,0],DLB=['e','y','u','rest'],DTV=['0.48','0.12','0.11','0.29'];
F.distill=(mode)=>{let L=DST,R=DTE;return{
  prep(s){const t=ease(s);if(mode==='teach'){L=DTE;R=DST.map((v,k)=>lerp(v,DTE[k],t));}else{L=DOH;R=DTE.map(v=>v*t);}},
  p(i,s,o){if(i>=416){o.a=0;o.x=0;o.y=-.55;return;}const g=i<208?0:1,k=i%208,b=Math.floor(k/52),j=k%52,c=j%4,r=Math.floor(j/4);const h=(g?R:L)[b];
    o.x=(g?.5:-.5)+(b-1.5)*.2+(c-1.5)*.026;o.y=-.55+h*1.3*(r/12);o.c=g?(mode==='teach'?GD:BL):(mode==='teach'?BL:FG);o.r=.8;},
  draw(s,a){[-.5,.5].forEach(cx=>DLB.forEach((l,b)=>text(l,cx+(b-1.5)*.2,-.67,a,{size:.044})));
    const tv=mode==='teach'?-.5:.5,al=mode==='teach'?a:a*sstep(.6,.95,s);DTE.forEach((v,b)=>b<3&&text(DTV[b],tv+(b-1.5)*.2,-.55+v*1.3+.09,al,{c:BL,size:.04}));
    if(mode==='teach'){text('teacher',-.5,.92,a,{c:BL,size:.052});text('student',.5,.92,a,{c:GD,size:.052});}
    else{text('label',-.5,.92,a,{c:FG,size:.052});text('teacher',.5,.92,a,{c:BL,size:.052});text('1',-.5-.3,-.55+1.3+.09,a,{c:FG,size:.04});}}};};

F.prefs=()=>({p(i,s,o){const g=Math.floor(i/14),m=i%14,ch=m<7,k=m%7;const t=ease(s);const x=-.9+g*(1.8/29);const base=(H(g,521)-.5)*.25;
    const lift=(.32+.15*H(g,523))*t;o.x=x+(ch?-.014:.014);o.y=base+(ch?lift:-lift)+(k-3)*.022;o.c=ch?BL:GD;o.r=.75;},
  draw(s,a){const t=ease(s);text('chosen',-.95,.75,a*sstep(.3,.7,s),{align:'left',c:BL,size:.05});text('rejected',-.95,-.75,a*sstep(.3,.7,s),{align:'left',c:GD,size:.05});
    if(t>.05)for(let g=0;g<30;g+=3){const x=-.9+g*(1.8/29),base=(H(g,521)-.5)*.25,lift=(.32+.15*H(g,523))*t;line(x,base+lift,x,base-lift,FT,a*.6,1,[2,3]);}}});

/* Agents. */
const TKW=['It','can','only','write','the','next','word','…'];
F.tokline=()=>{let k=0;return{
  prep(s){k=lerp(1,8.4,ease(s));},
  p(i,s,o){const j=Math.floor(i/52);if(j>7){o.a=0;return;}const m=i%52,cx=-.81+j*.232;
    let x,y;if(m<40){const side=Math.floor(m/10);const u=(m%10)/9;x=side<2?cx-.1+u*.2:(side===2?cx-.1:cx+.1);y=side===0?.1:side===1?-.1:-.1+u*.2;}else{x=cx-.07+((m-40)%6)*.028;y=Math.floor((m-40)/6)*.05-.03;}
    o.x=x;o.y=y;const v=clamp(k-j);o.a=v;o.c=j===Math.floor(k)-1||(v<1&&v>0)?BL:FG;o.r=.8;if(v<1)o.x+=(1-v)*.25;},
  draw(s,a){TKW.forEach((w,j)=>{const v=clamp(k-j);if(v>.05)text(w,-.81+j*.232,.24,a*v,{c:FG,size:.05});});text('text in, text out',0,-.5,a,{size:.05});}};};

const AGC=[0,.08],AGR=.55,AGA={th:Math.PI/2,ac:-Math.PI/6,ob:Math.PI*7/6};
const agP=ang=>[AGC[0]+AGR*Math.cos(ang),AGC[1]+AGR*Math.sin(ang)];
const AGANS=[0,-.92];
F.agent=(mode)=>({p(i,s,o){
    if(i<90){const nd=Math.floor(i/30),k=i%30;const ang=[AGA.th,AGA.ac,AGA.ob][nd];let [cx,cy]=agP(ang);
      if(mode==='code'&&nd===1){o.x=-.9+(k%15)*.06;o.y=-.75-Math.floor(k/15)*.12;o.c=BL;o.r=.75;return;}
      const a=k/30*TAU+s*2;o.x=cx+.075*Math.cos(a);o.y=cy+.075*Math.sin(a);o.c=[FG,BL,GD][nd];o.r=.9;return;}
    const k=i-90;const ph=H(k,531)*TAU;const sp=mode==='stuck'?2.6:1.3;let ang=ph-s*TAU*sp;const jr=G(k,533)*.02;let [x,y]=[AGC[0]+(AGR+jr)*Math.cos(ang),AGC[1]+(AGR+jr)*Math.sin(ang)];
    o.c=FT;o.r=.75;o.a=.7;
    if(mode==='exit'){const t=clamp((ease(s)-H(k,535)*.85)*4);if(t>0){const q=[AGANS[0]+G(k,537)*.05,AGANS[1]+G(k,539)*.04];x=lerp(x,q[0],ease(t));y=lerp(y,q[1],ease(t));o.c=t>.5?BL:FT;o.a=.7+.3*t;}}
    if(mode==='check'&&H(k,541)<.25){const f=(H(k,543)+s*2)%1;ang=AGA.ac-Math.PI/3*(1-Math.abs(1-2*f));x=AGC[0]+(AGR+jr)*Math.cos(ang);y=AGC[1]+(AGR+jr)*Math.sin(ang);o.c=GD;o.a=.9;}
    if(mode==='stuck'){const w=1+.06*Math.sin(s*9+k);x=AGC[0]+(x-AGC[0])*w;y=AGC[1]+(y-AGC[1])*w;}
    o.x=x;o.y=y;},
  draw(s,a){const [tx,ty]=agP(AGA.th),[ax,ay]=agP(AGA.ac),[ox,oy]=agP(AGA.ob);circ(AGC[0],AGC[1],AGR,FT,a*.6,[3,5]);
    text('thought',tx,ty+.17,a,{c:FG,size:.05});text(mode==='code'?'code':'action',ax,ay+.17,a,{c:BL,size:.05});text('observation',ox,oy+.17,a,{c:GD,size:.05});
    if(mode==='exit'){circ(AGANS[0],AGANS[1],.1,BL,a*sstep(.1,.5,s));text('Final Answer',AGANS[0]+.16,AGANS[1],a*sstep(.3,.7,s),{c:BL,align:'left',size:.05});}
    if(mode==='code'){rect(-.95,-1.0,.95,-.6,BL,a*.5);line(ax,ay-.08,ax,-.6,BL,a*.5,1,[3,4]);text("r = kb_lookup('seconds in an hour')",-.91,-.69,a,{c:FG,align:'left',size:.042});text("print(calculator(f'{r} * 2'))",-.91,-.81,a,{c:FG,align:'left',size:.042});}
    if(mode==='check'){const [cx,cy]=agP(-Math.PI/2);circ(cx,cy,.07,GD,a);text('check',cx,cy-.17,a,{c:GD,size:.05});}
    if(mode==='stuck'){circ(AGANS[0],AGANS[1],.1,FT,a*.8,[3,4]);text('no answer yet: step '+(1+Math.floor(s*29.99)),0,AGANS[1]-.18,a,{size:.05});}}});

const TRL=[['Thought: I will look up seconds in a day.',FG],['Action: kb_lookup[seconds in a day]',BL],['Observation: 86400',GD],['Thought: Now I multiply by 3.',FG],['Action: calculator[86400 * 3]',BL],['Observation: 259200',GD],['Final Answer: 259200',BL]];
F.trace=()=>{let k=0;return{
  prep(s){k=lerp(.6,7.4,ease(s));},
  p(i,s,o){const j=Math.floor(i/60);if(j>6){o.a=0;return;}const m=i%60,len=Math.min(2.1,TRL[j][0].length*.046);const y=.86-j*.27-.08;const v=clamp(k-j);
    o.x=-1.08+len*(m/59)*v;o.y=y;o.c=TRL[j][1];o.r=.7;o.a=v;},
  draw(s,a){TRL.forEach(([t,c],j)=>{const v=clamp(k-j);if(v>.05)text(t,-1.08,.86-j*.27,a*v,{c,align:'left',size:.04});});}};};

const HBT=['calculator','files','search','database','calendar','browser'],HBC=[-.2,0],HBM=[-.82,0];
const hbN=j=>{const an=(-70+j*28)*Math.PI/180;return[HBC[0]+.62*Math.cos(an),HBC[1]+.62*Math.sin(an)];};
F.hub=()=>({p(i,s,o){
    if(i<30){const a=i/30*TAU+s*2;o.x=HBM[0]+.08*Math.cos(a);o.y=HBM[1]+.08*Math.sin(a);o.c=FG;o.r=.9;return;}
    if(i<70){const a=(i-30)/40*TAU-s*2;o.x=HBC[0]+.1*Math.cos(a);o.y=HBC[1]+.1*Math.sin(a);o.c=BL;o.r=.9;return;}
    if(i<190){const j=Math.floor((i-70)/20),a=(i-70)%20/20*TAU+s*2;const [x,y]=hbN(j);o.x=x+.05*Math.cos(a);o.y=y+.05*Math.sin(a);o.c=GD;o.r=.8;return;}
    const k=i-190,sp=k%7,ph=(H(k,551)+s*1.5)%1;const A=sp===6?HBM:HBC,B=sp===6?HBC:hbN(sp);const back=H(k,553)<.5;const t=back?1-ph:ph;
    o.x=lerp(A[0],B[0],t);o.y=lerp(A[1],B[1],t);o.c=sp===6?FG:FT;o.r=.65;o.a=.75;},
  draw(s,a){for(let j=0;j<6;j++){const [x,y]=hbN(j);line(HBC[0],HBC[1],x,y,FT,a*.6,1);text(HBT[j],x+.09,y,a,{align:'left',size:.044});}
    line(HBM[0],HBM[1],HBC[0],HBC[1],FT,a*.6,1);text('model',HBM[0],HBM[1]+.17,a,{c:FG,size:.05});text('one protocol',HBC[0],HBC[1]-.2,a,{c:BL,size:.046});}});


window.MovesF={F,N,H,G,clamp,lerp,ease,sstep,sig,softmax,along,V,FG,BL,GD,FT,
  setCtx(c){ctx=c;},setRGB(rgb,muted){RGB=rgb;MUTED=muted;}};
})();
