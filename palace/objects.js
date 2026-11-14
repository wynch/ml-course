/* Palace objects: the formations the Moves page does not have, and how each furnished
   object stands in its room.

   A palace formation has the Moves shape {prep?(s), p(i,s,o), draw?(s,a,D)} with two
   additions: p may set o.z (unit depth, toward the viewer), and draw gets D, which
   draws in three dimensions: D.line(x0,y0,z0,x1,y1,z1,c,a), D.poly(P,c,a) with
   P = [[x,y,z],...], D.text(t,x,y,z,a,{c,size,align}). Units: the Moves unit square,
   about -1..1, y up; one unit is SPEC.scale metres in the room.

   OBJ[id] = {make, kind, base, look, depth, note}
     make  () => formation or [formations played in turn]; default: the data's moves_formation
     kind  'panel' (a standing sheet of points) or '3d' (uses depth; the camera looks down at it)
     base  height of the object's centre in metres (default 1.45)
     depth (i,s,o,k) => extra o.z for a Moves formation given depth in the palace
*/
(()=>{
'use strict';
const M=window.MovesF, {F,N,H,G,clamp,lerp,ease,sstep,sig}=M;
const FG=0,BL=1,GD=2,FT=3,TAU=Math.PI*2;
const P={};

/* 00a-4 · least squares as a projection, then gradient descent past its bound. */
P.sundial=()=>{const e1=[1,0,0],e2=[0,.28,-.96];const n2=Math.hypot(...e2);e2[0]/=n2;e2[1]/=n2;e2[2]/=n2;
  const Y=[.28,.95,-.05];const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  const ws=[dot(Y,e1),dot(Y,e2)];const at=(a,b)=>[a*e1[0]+b*e2[0],a*e1[1]+b*e2[1],a*e1[2]+b*e2[2]];
  const FOOT=at(ws[0],ws[1]);
  const run=eta=>{let w=[-.75,-.55];const Q=[w.slice()];for(let k=0;k<12;k++){w=[ws[0]+(1-eta)*(w[0]-ws[0]),ws[1]+(1-eta)*(w[1]-ws[1])];Q.push(w.slice());}return Q;};
  const R1=run(.55),R2=run(2.12);let path=R1,t=0,stable=true;
  return{palace:true,
  prep(s){stable=s<.5;path=stable?R1:R2;t=ease(stable?s*2:(s-.5)*2)*12;},
  p(i,s,o){
    if(i<324){const a=((i%18)-8.5)/9.4,b=(Math.floor(i/18)-8.5)/9.4;const q=at(a,b);o.x=q[0];o.y=q[1];o.z=q[2];o.c=FT;o.r=.75;o.a=.8;return;}
    if(i<340){const k=i-324;const q=at(...path[Math.min(12,Math.floor(t))]);const r=.025;o.x=q[0]+r*Math.cos(k);o.y=q[1]+r*Math.sin(k*1.3);o.z=q[2]+r*Math.sin(k);o.c=GD;o.r=1.4;
      const k0=Math.floor(t),f=t-k0,A=path[Math.min(12,k0)],B=path[Math.min(12,k0+1)];const w=[lerp(A[0],B[0],ease(f)),lerp(A[1],B[1],ease(f))];
      if(Math.abs(w[0])>1.6||Math.abs(w[1])>1.6){o.a=0;return;}const q2=at(w[0],w[1]);o.x=q2[0]+r*Math.cos(k);o.y=q2[1]+r*Math.sin(k*1.3);o.z=q2[2]+r*Math.sin(k);return;}
    if(i<370){const k=i-340,u=k/29;o.x=lerp(Y[0],FOOT[0],u);o.y=lerp(Y[1],FOOT[1],u);o.z=lerp(Y[2],FOOT[2],u);o.c=BL;o.r=.8;return;}
    if(i<380){const k=i-370;o.x=Y[0]+G(k,7)*.02;o.y=Y[1]+G(k,8)*.02;o.z=Y[2]+G(k,9)*.02;o.c=BL;o.r=2.2;return;}
    o.a=0;},
  draw(s,a,D){D.line(Y[0],Y[1],Y[2],FOOT[0],FOOT[1],FOOT[2],BL,a*.8);
    const u=[.07,0,0],v=[e2[0]*.07,e2[1]*.07,e2[2]*.07];const nrm=[Y[0]-FOOT[0],Y[1]-FOOT[1],Y[2]-FOOT[2]];const L=Math.hypot(...nrm);const m=[nrm[0]/L*.07,nrm[1]/L*.07,nrm[2]/L*.07];
    D.line(FOOT[0]+u[0],FOOT[1]+u[1],FOOT[2]+u[2],FOOT[0]+u[0]+m[0],FOOT[1]+u[1]+m[1],FOOT[2]+u[2]+m[2],FG,a*.7);
    D.line(FOOT[0]+m[0],FOOT[1]+m[1],FOOT[2]+m[2],FOOT[0]+u[0]+m[0],FOOT[1]+u[1]+m[1],FOOT[2]+u[2]+m[2],FG,a*.7);
    const k0=Math.min(12,Math.floor(t));const Q=[];for(let k=0;k<=k0;k++){const w=path[k];if(Math.abs(w[0])>1.6||Math.abs(w[1])>1.6)break;Q.push(at(w[0],w[1]));}
    D.poly(Q,GD,a*.75);
    D.text('target y',Y[0]+.08,Y[1]+.1,Y[2],a,{size:.05,align:'left'});
    D.text('fit Xw',FOOT[0]+.1,FOOT[1]-.1,FOOT[2],a,{size:.05,align:'left',c:BL});
    D.text(stable?'step below the bound: settles':'step past the bound: flies off',0,-1.02,0,a,{size:.05,c:stable?FG:GD});}};};

/* 00b-2 · naive Bayes on two tilted clouds: its axis-aligned ellipses and the boundary they give. */
P.naive=()=>{const th=.62,sx=.36,sy=.07,c=Math.cos(th),sn=Math.sin(th);const M0=[-.22,-.3],M1=[.22,.3];
  const vx=(sx*c)**2+(sy*sn)**2,vy=(sx*sn)**2+(sy*c)**2,cxy=(sx*sx-sy*sy)*c*sn;
  const pt=i=>{const cl=i%2,m=cl?M1:M0,u=G(i,601)*sx,v=G(i,603)*sy;return[m[0]+u*c-v*sn,m[1]+u*sn+v*c,cl];};
  const dm=[M1[0]-M0[0],M1[1]-M0[1]],mid=[0,0];
  const wN=[dm[0]/vx,dm[1]/vy];
  const det=vx*vy-cxy*cxy,wT=[(vy*dm[0]-cxy*dm[1])/det,(-cxy*dm[0]+vx*dm[1])/det];
  const side=(w,x,y)=>w[0]*(x-mid[0])+w[1]*(y-mid[1]);
  const cutLine=(w,D,col,a,lab)=>{const L=Math.hypot(w[0],w[1]),dx=-w[1]/L,dy=w[0]/L;D.line(-dx*1.1,-dy*1.1,0,dx*1.1,dy*1.1,0,col,a);D.text(lab,dx*1.02+.04,dy*1.02,0,a,{size:.048,c:col,align:'left'});};
  return{palace:true,
  p(i,s,o){const [x,y,cl]=pt(i);o.x=x;o.y=y;o.c=cl?GD:BL;o.r=.85;
    if(s>.55){const wrong=(side(wN,x,y)>0)!==(cl===1);if(wrong){o.r=1.9;o.a=1;}else o.a=.55;}},
  draw(s,a,D){const e=sstep(.05,.35,s);if(e>0){for(const m of [M0,M1]){const Q=[];for(let k=0;k<=48;k++){const t=k/48*TAU;Q.push([m[0]+Math.sqrt(vx)*1.7*Math.cos(t),m[1]+Math.sqrt(vy)*1.7*Math.sin(t),0]);}D.poly(Q,FG,a*.45*e);}}
    if(s>.4)cutLine(wN,D,GD,a*sstep(.4,.55,s),'naive');
    if(s>.7)cutLine(wT,D,FG,a*.6*sstep(.7,.85,s),'full covariance');
    D.text(s<.55?'one upright bell per feature':'big points: wrong by the naive cut',0,-1.04,0,a,{size:.05});}};};

/* 00b-5 · k-means: three centres walk to the mean of their points. */
P.kmeans=()=>{const CC=[[-.55,.35],[.5,.45],[.05,-.5]];const X=[],Y=[];for(let i=0;i<N;i++){const c=CC[i%3];X.push(c[0]+G(i,611)*.17);Y.push(c[1]+G(i,613)*.17);}
  const steps=[[[-.9,-.85],[-.75,-.95],[-.55,-.8]]];const asg=[];
  for(let it=0;it<8;it++){const C=steps[it];const A=new Uint8Array(N);const S=C.map(()=>[0,0,0]);
    for(let i=0;i<N;i++){let b=0,bd=9;for(let k=0;k<3;k++){const d=(X[i]-C[k][0])**2+(Y[i]-C[k][1])**2;if(d<bd){bd=d;b=k;}}A[i]=b;S[b][0]+=X[i];S[b][1]+=Y[i];S[b][2]++;}
    asg.push(A);steps.push(C.map((c,k)=>S[k][2]?[S[k][0]/S[k][2],S[k][1]/S[k][2]]:c));}
  let t=0;const cen=k=>{const a=Math.min(8,Math.floor(t)),f=ease(t-a),A=steps[a][k],B=steps[Math.min(8,a+1)][k];return[lerp(A[0],B[0],f),lerp(A[1],B[1],f)];};
  return{palace:true,
  prep(s){t=clamp(s*1.15)*8;},
  p(i,s,o){if(i>=N-9){const k=(i-(N-9))%3,j=Math.floor((i-(N-9))/3);const c=cen(k);o.x=c[0]+.02*Math.cos(j*2.1);o.y=c[1]+.02*Math.sin(j*2.1);o.c=GD;o.r=2.6;return;}
    o.x=X[i];o.y=Y[i];const A=asg[Math.min(7,Math.floor(t))];o.c=[BL,FG,FT][A[i]];o.r=.85;if(A[i]===2){o.c=BL;o.a=.5;}},
  draw(s,a,D){for(let k=0;k<3;k++){const Q=[];for(let j=0;j<=Math.min(8,Math.floor(t));j++)Q.push([steps[j][k][0],steps[j][k][1],0]);Q.push([...cen(k),0]);D.poly(Q,GD,a*.6);}
    D.text('round '+Math.min(8,Math.floor(t)+1),0,1.0,0,a,{size:.052});}};};

/* 00c-2 · the kernel lift: rings rise by x² + y², and a flat sheet slides between them. */
P.lift=()=>{let h=0,sheet=0;return{palace:true,
  prep(s){h=ease(sstep(.05,.55,s));sheet=sstep(.55,.85,s);},
  p(i,s,o){if(i<300){const inner=i<110,k=inner?i:i-110,n=inner?110:190;const r=(inner?.3:.78)+G(i,621)*.035,th=k/n*TAU;
      const x=r*Math.cos(th),z=r*Math.sin(th);o.x=x;o.z=z;o.y=-.55+h*1.5*(x*x+z*z);o.c=inner?BL:GD;o.r=1;return;}
    const k=i-300,gx=(k%12-5.5)/5.5*.95,gz=(Math.floor(k/12)-4.5)/4.5*.95;o.x=gx;o.z=gz;o.y=-.55+1.5*.29;o.c=FG;o.r=.6;o.a=.55*sheet;},
  draw(s,a,D){D.text(h<.5?'no line splits two rings':'height = x² + y²',0,-.95,0,a,{size:.055});if(sheet>.5)D.text('a flat cut',.98,-.1,0,a*sheet,{size:.05,align:'left'});}};};

/* 03-4 and others · play two Moves formations' text over a sequence. */
P.caption=(f,msg,c=FG)=>({prep:f.prep&&(s=>f.prep(s)),p:(i,s,o)=>f.p(i,s,o),palace:true,
  draw(s,a,D){if(f.draw)D.moves(f,s,a);D.text(msg,0,-1.0,0,a,{size:.05,c});}});

/* 03-5 · tokens per word, SmolLM3 (results/fertility.json, module 03). */
P.fertility=()=>F.vbars({max:2.2,cols:8,head:'SmolLM3 · tokens per word',bars:[{x:-.4,v:1.14,c:BL,top:'1.14',lab:'English'},{x:.4,v:1.97,c:GD,top:'1.97',lab:'Spanish'}]});

/* 08-1 · a 3 × 3 kernel slides over a 14 × 14 image and fills a 12 × 12 output. */
P.conv=()=>{const img=(r,c)=>{const x=(c-6.5)/6.5,y=(r-6.5)/6.5;return Math.exp(-((x+.2)**2+(y-.1)**2)/.25)+.6*(Math.abs(x-y)<.18?1:0);};
  const out=(r,c)=>{let v=0;for(let a=0;a<3;a++)for(let b=0;b<3;b++)v+=(b-1)*img(r+a,c+b);return v;};
  let k=0;return{palace:true,
  prep(s){k=Math.floor(ease(s)*144);},
  p(i,s,o){if(i<196){const r=Math.floor(i/14),c=i%14;o.x=-.95+c*.068;o.y=.45-r*.068;const v=img(r,c);o.r=.5+1.2*Math.min(1,v);o.c=FG;o.a=.35+.6*Math.min(1,v);
      const kr=Math.floor(Math.min(143,k)/12),kc=Math.min(143,k)%12;if(r>=kr&&r<kr+3&&c>=kc&&c<kc+3){o.c=GD;o.a=1;}return;}
    if(i<340){const j=i-196,r=Math.floor(j/12),c=j%12;o.x=.22+c*.068;o.y=.38-r*.068;if(j>k){o.r=.4;o.c=FT;o.a=.5;return;}const v=out(r,c);o.r=.4+1.1*Math.min(1,Math.abs(v));o.c=v>0?BL:GD;return;}
    if(i<349){const j=i-340;o.x=-.12+(j%3)*.07;o.y=.85-Math.floor(j/3)*.07;o.c=(j%3)===1?FT:((j%3)===0?GD:BL);o.r=1.6;return;}o.a=0;},
  draw(s,a,D){const kk=Math.min(143,k),kr=Math.floor(kk/12),kc=kk%12;const x0=-.95+kc*.068-.034,y0=.45-kr*.068+.034;
    D.poly([[x0,y0,0],[x0+.204,y0,0],[x0+.204,y0-.204,0],[x0,y0-.204,0],[x0,y0,0]],GD,a*.9);
    const ox=.22+kc*.068,oy=.38-kr*.068;D.line(x0+.204,y0-.1,0,ox,oy,0,GD,a*.35);
    D.text('image',-.5,-.62,0,a,{size:.05});D.text('output',.6,-.55,0,a,{size:.05});D.text('one 3 × 3 kernel, shared',.08,.98,0,a,{size:.048});}};};

/* 08-2 · the first layer's eight 5 × 5 kernels sharpen from noise into edges and blobs (cnn.py: conv1 = Conv2d(1, 8, 5)). */
P.filters=()=>{const pat=(k,r,c)=>{const x=(c-2)/2,y=(r-2)/2;if(k<6){const th=k/6*Math.PI;return Math.tanh(3*(x*Math.cos(th)+y*Math.sin(th)));}const d=x*x+y*y;return (k%2?1:-1)*(Math.exp(-d*2.5)*2-.6);};
  return{palace:true,
  p(i,s,o){if(i>=400){o.a=0;return;}const q=i>>1,k=Math.floor(q/25),j=q%25,r=Math.floor(j/5),c=j%5;const t=ease(s);
    const v=lerp(G(q,631)*.7,pat(k,r,c),t);const gx=k%4,gy=Math.floor(k/4);o.x=-.75+gx*.5+(c-2)*.085+(i&1)*.012;o.y=.38-gy*.62-(r-2)*.085+(i&1)*.012;o.c=v>0?BL:GD;o.r=.4+1.4*Math.min(1,Math.abs(v));o.a=.4+.6*Math.min(1,Math.abs(v));},
  draw(s,a,D){D.text(s<.4?'eight first-layer kernels at the start':'after training: edges and blobs',0,-1.06,0,a,{size:.05});}};};

/* 08-4 · [CLS] attention over a ViT's 14 × 14 patches. */
P.attnmap=()=>{const w=(r,c)=>{const x=(c-6.5)/6.5,y=(6.5-r)/6.5;const body=Math.exp(-((x+.05)**2/.12+(y+.1)**2/.06));const head=Math.exp(-((x-.42)**2+(y-.32)**2)/.025);return Math.max(body,head*1.2);};
  return{palace:true,
  p(i,s,o){if(i>=392){const k=i-392;o.x=-.05+(k%7)*.022;o.y=1.0+Math.floor(k/7)*.022;o.c=GD;o.r=1.2;return;}const q=i>>1,r=Math.floor(q/14),c=q%14;o.x=(c-6.5)*.126+(i&1)*.016;o.y=(6.5-r)*.126-.06+(i&1)*.016;
    const t=ease(s),v=lerp(.25,w(r,c),t);o.r=.45+2.2*v;o.c=v>.45?GD:(v>.2?BL:FT);o.a=.35+.65*v;},
  draw(s,a,D){D.text('[CLS]',.12,1.02,0,a,{size:.048,align:'left',c:GD});D.text('last-layer attention from [CLS]',0,-1.0,0,a,{size:.048});}};};

/* 08-5 · CLIP: one image and six captions on a sphere; the nearest caption lights. */
P.clip=()=>{const CAP=['a dog','a cat','a car','a tree','a boat','a bird'];const CP=[];for(let j=0;j<6;j++){const th=j/6*TAU+.3,ph=(H(j,641)-.5)*1.2;CP.push([Math.cos(ph)*Math.cos(th),Math.sin(ph),Math.cos(ph)*Math.sin(th)]);}
  const IMG0=[.2,.8,-.56],IMG1=(()=>{const c=CP[0];return [c[0]*.9+.1*IMG0[0],c[1]*.9+.1*IMG0[1],c[2]*.9+.1*IMG0[2]];})();
  const nrm=v=>{const L=Math.hypot(...v);return v.map(x=>x/L);};let im=IMG0,rot=0,best=0;
  return{palace:true,
  prep(s){const t=ease(sstep(0,.6,s));im=nrm([lerp(IMG0[0],IMG1[0],t),lerp(IMG0[1],IMG1[1],t),lerp(IMG0[2],IMG1[2],t)]);rot=s*.8;
    let bd=-2;CP.forEach((c,j)=>{const d=c[0]*im[0]+c[1]*im[1]+c[2]*im[2];if(d>bd){bd=d;best=j;}});},
  R(v){const c=Math.cos(rot),s=Math.sin(rot);return[v[0]*c-v[2]*s,v[1],v[0]*s+v[2]*c].map(x=>x*.82);},
  p(i,s,o){if(i<330){const u=(i+.5)/330,ph=Math.acos(1-2*u),th=i*2.39996;const q=this.R([Math.sin(ph)*Math.cos(th),Math.cos(ph),Math.sin(ph)*Math.sin(th)]);o.x=q[0];o.y=q[1];o.z=q[2];o.c=FT;o.r=.55;o.a=.7;return;}
    if(i<390){const j=Math.floor((i-330)/10),k=(i-330)%10;const q=this.R(CP[j]);o.x=q[0]+.025*Math.cos(k*.63);o.y=q[1]+.025*Math.sin(k*.63);o.z=q[2];o.c=(j===best&&s>.6)?GD:FG;o.r=1.5;return;}
    const k=i-390;const q=this.R(im);o.x=q[0]+.03*Math.cos(k*.21);o.y=q[1]+.03*Math.sin(k*.21);o.z=q[2]+.02*Math.sin(k);o.c=BL;o.r=1.6;},
  draw(s,a,D){const q=this.R(im);CP.forEach((c,j)=>{const p=this.R(c);const lit=j===best&&s>.6;D.line(q[0],q[1],q[2],p[0],p[1],p[2],lit?GD:FT,a*(lit?.9:.45));D.text(CAP[j],p[0]+.06,p[1]+.06,p[2],a,{size:.046,align:'left',c:lit?GD:-1});});
    D.text('photo',q[0]-.06,q[1]-.08,q[2],a,{size:.046,align:'right',c:BL});D.text('nearest caption wins',0,-1.0,0,a,{size:.048});}};};

/* 05-1 · SmolLM2-360M by role: 32 layers of attention and MLP over the embedding table (results/anatomy.json). */
P.params=()=>{const NL=8;/* 32 layers drawn as 8 slabs of 4 */const EMB=55,ATT=Math.round(.22*N/NL),MLP=Math.floor((N-EMB-ATT*NL)/NL);
  return{palace:true,
  p(i,s,o){const t=ease(s);let x,y,z;
    if(i<EMB){x=-.7+(i%11)*.14;z=-.35+Math.floor(i/11)*.17;y=-.9;o.c=FG;}
    else{const j=i-EMB,l=Math.floor(j/(ATT+MLP));const k=j%(ATT+MLP);if(l>=NL){o.a=0;return;}
      const yl=-.62+l*.2;if(k<ATT){x=-.72+(k%5)*.06;z=-.2+Math.floor(k/5)*.09;o.c=BL;}else{const m=k-ATT;x=-.25+(m%10)*.09;z=-.33+Math.floor(m/10)*.13;o.c=GD;}y=yl;}
    const g=G(i,651)*.6;o.x=lerp(g,x,t);o.y=lerp(G(i,653)*.6,y,t);o.z=lerp(G(i,655)*.5,z,t);o.r=1;},
  draw(s,a,D){const e=a*sstep(.5,.9,s);D.text('embeddings ~13%',.9,-.9,0,e,{size:.046,align:'left'});D.text('attention ~22%',-.85,.95,0,e,{size:.046,c:BL});D.text('MLP ~2/3',.55,.95,0,e,{size:.046,c:GD});D.text('32 layers, drawn 4 to a slab',0,-1.12,0,e,{size:.044});}};};

/* 05-5 · the logit lens: eight layer slices, each a bar chart of next-token odds. */
P.lens=()=>{const NL=8,NB=5,TOK=['Paris','','other tokens','',''];const odds=l=>{/* results/logit_lens.json: Paris is at 0.0% through layer 25, 71% at layer 27, 98.7% at 28, 98.5% at 32. Slices show layers 4, 8, … 32. */const p0=l<6?0:l===6?.987:.985;const r=[p0];for(let b=1;b<NB;b++)r.push((1-p0)*(.15+.6*H(l*7+b,661))/(NB-1)*1.6);return r;};
  const O=[];for(let l=0;l<NL;l++)O.push(odds(l));let shown=0;
  return{palace:true,
  prep(s){shown=sstep(0,.9,s)*NL;},
  p(i,s,o){const l=Math.floor(i/52),k=i%52;if(l>=NL){o.a=0;return;}const b=Math.floor(k/10),m=k%10;if(b>=NB){o.a=0;return;}
    const z=.75-l*.21,x=-.45+b*.22+(l-3.5)*.07,h=Math.min(1,O[l][b])*1.1,yb=-.75+(l-3.5)*.09;o.x=x+(m%2)*.03;o.y=yb+(Math.floor(m/2)/4)*h;o.z=z;
    const top=O[l].indexOf(Math.max(...O[l]));o.c=(b===top&&l>=6)?GD:(b===0?BL:FT);o.r=.95;o.a=l<shown?1:.08;},
  draw(s,a,D){for(let l=0;l<NL;l++){const z=.75-l*.21,dx=(l-3.5)*.07,dy=(l-3.5)*.09;D.line(-.55+dx,-.77+dy,z,.55+dx,-.77+dy,z,FT,a*.5);if(l%2===1)D.text('layer '+((l+1)*4),.6+dx,-.75+dy,z,a*(l<shown?1:.3),{size:.04,align:'left'});}
    for(let b=0;b<NB;b++)if(TOK[b])D.text(TOK[b],-.45+b*.22+.245,-.88+.315,-.72,a,{size:.038,c:b===0?BL:-1});D.text('Paris tops from layer 27 of 32',.2,1.0,-.75,a*sstep(.75,.95,s),{size:.046,c:GD});}};};

/* 07-1 · one flat ribbon of numbers splits into named tensors. */
P.manifest=()=>{const parts=[['embed = head',112],['L0 qkv',45],['L0 out',15],['L0 mlp',90],['norm',8],['L1 qkv',45],['L1 out',15],['L1 mlp',90]];/* no head block: the LM head is wte.weight reused (module 04, EXPORT_FORMAT.md) */
  const start=[];let acc=0;parts.forEach(p=>{start.push(acc);acc+=p[1];});
  const rowOf=j=>j<3?0:j<6?1:2;
  return{palace:true,
  p(i,s,o){const t=ease(sstep(.2,.8,s));let j=0;while(j<parts.length-1&&i>=start[j+1])j++;const k=i-start[j];
    const x0=-.95+i/N*1.9,y0=.1+.05*Math.sin(i*.3);
    const row=rowOf(j);let px=-.95;for(let q=0;q<j;q++)if(rowOf(q)===row)px+=parts[q][1]/90*.42+.08;const x1=px+(k%Math.ceil(parts[j][1]/6))*.07*.6,y1=.55-row*.5-Math.floor(k/Math.ceil(parts[j][1]/6))*.045;
    o.x=lerp(x0,x1,t);o.y=lerp(y0,y1,t);o.c=j%3===0?BL:(j%3===1?FG:GD);o.r=.8;},
  draw(s,a,D){const e=sstep(.7,.95,s);parts.forEach((p,j)=>{const row=rowOf(j);let px=-.95;for(let q=0;q<j;q++)if(rowOf(q)===row)px+=parts[q][1]/90*.42+.08;D.text(p[0],px,.64-row*.5,0,a*e,{size:.042,align:'left'});});
    D.text(s<.25?'one blob of bytes':'the manifest names each piece',0,-1.0,0,a,{size:.05});}};};

/* 07-3 · prefill fills the lower triangle at once; decode adds one full row per step. */
P.prefill=()=>{const n=20,P0=12;let rows=0,flash=0;return{palace:true,
  prep(s){flash=sstep(0,.15,s);rows=P0+Math.floor(sstep(.3,.95,s)*(n-P0+.99));},
  p(i,s,o){if(i>=400){o.a=0;return;}const r=Math.floor(i/20),c=i%20;o.x=(c-9.5)*.088;o.y=(9.5-r)*.088;
    if(c>r){o.c=FT;o.r=.4;o.a=.35;return;}if(r<P0){o.c=BL;o.r=.9;o.a=flash;return;}if(r<rows){o.c=r===rows-1?GD:BL;o.r=r===rows-1?1.3:.9;return;}o.c=FT;o.r=.4;o.a=.25;},
  draw(s,a,D){D.text(s<.3?'prefill: the prompt fills the cache':'decode: one row per token, nothing masked',0,1.0,0,a,{size:.05});}};};

/* 07-6 · the engine's outputs laid over the reference's: all within tolerance. */
P.parity=()=>{let t=0;return{palace:true,
  prep(s){t=ease(sstep(0,.7,s));},
  p(i,s,o){const k=i>>1,ref=i&1;const u=k/210,x=-.9+u*1.8,y=.45*Math.sin(u*9)*Math.exp(-u*.6)+G(k,671)*.05;
    o.x=ref?x+(1-t)*.9:x;o.y=y;o.c=ref?FG:BL;o.r=ref?.6:1.2;o.a=ref?.9:.75;},
  draw(s,a,D){D.text('engine',-.9,.75,0,a,{size:.048,c:BL,align:'left'});D.text('PyTorch',.9,.75,0,a,{size:.048,align:'right'});
    if(s>.75)D.text('every value within tolerance',0,-.8,0,a*sstep(.75,.9,s),{size:.05,c:GD});}};};

/* 18-1 · a GRU's update gate z is the share of the old state kept; 1 − z comes from the candidate (module 18: h ← (1 − z) ⊙ n + z ⊙ h). */
P.gru=()=>{const Z=[.15,.8,.3,.05,.6,.4];const C=Z.map((_,k)=>[(H(k,681)-.5)*1.1,(H(k,683)-.5)*.6+.1]);
  const S=[[0,.1]];for(let k=0;k<6;k++){const p=S[k];S.push([lerp(p[0],C[k][0],1-Z[k]),lerp(p[1],C[k][1],1-Z[k])]);}
  let t=0;return{palace:true,
  prep(s){t=s*6;},
  p(i,s,o){const k=Math.min(5,Math.floor(t)),f=ease(clamp((t-k)*1.6));
    if(i<240){const a=S[k],b=S[k+1];o.x=lerp(a[0],b[0],f)+G(i,685)*.12;o.y=lerp(a[1],b[1],f)+G(i,687)*.12;o.c=BL;o.r=.9;return;}
    if(i<400){const c=C[k];o.x=c[0]+G(i,689)*.1;o.y=c[1]+G(i,691)*.1;o.c=FT;o.r=.8;o.a=.7*(1-f*.5);return;}
    const j=i-400;o.x=-.9+(j%10)*.02;o.y=-.8+Math.floor(j/10)*.02*Z[k]*10;o.c=GD;o.r=.9;},
  draw(s,a,D){const k=Math.min(5,Math.floor(t));D.text('state',S[k][0],S[k][1]+.3,0,a,{size:.046,c:BL});D.text('candidate',C[k][0],C[k][1]-.22,0,a,{size:.046});
    D.text('token '+(k+1)+' · keeps z = '+Z[k].toFixed(2)+' of the old state',0,-1.0,0,a,{size:.05,c:GD});}};};

/* 18-3 and 18-4 · a decayed sum of the inputs, by kernel and by steps; selective decay breaks the kernel. */
P.scan=(selective)=>{const T=20,X=[],A=[];for(let k=0;k<T;k++){X.push(H(k,701)<.35?(.5+.5*H(k,703)):0);A.push(selective?(H(k,705)<.3?.15:.92):.8);}
  const Hs=[];let h=0;for(let k=0;k<T;k++){h=A[k]*h+X[k];Hs.push(h);}const mx=Math.max(...Hs);
  let t=0;return{palace:true,
  prep(s){t=sstep(.05,.9,s)*T;},
  p(i,s,o){const k=i%T,m=Math.floor(i/T);const x=-.9+k*(1.8/(T-1));
    if(m<5){o.x=x;o.y=-.75+m*.06*X[k]*3;o.c=FG;o.r=.8;o.a=X[k]?1:.25;return;}
    if(m<12){const y=-.2+.9*Hs[k]/mx;o.x=x+(m-8)*.008;o.y=y+(m-8)*.008;o.c=BL;o.r=.7;o.a=selective?.15:.6;return;}
    if(m<19){const y=-.2+.9*Hs[k]/mx;o.x=x+(m-15)*.006;o.y=y;o.c=GD;o.r=.9;o.a=k<t?1:0;return;}
    if(selective){o.x=x+(m-19)*.01;o.y=-.95+A[k]*.15;o.c=GD;o.r=.7;return;}o.a=0;},
  draw(s,a,D){const k=Math.min(T-1,Math.floor(t));const Q=[];for(let j=0;j<=k;j++)Q.push([-.9+j*(1.8/(T-1)),-.2+.9*Hs[j]/mx,0]);D.poly(Q,GD,a*.8);
    if(!selective){const K=[];for(let j=0;j<8;j++)K.push([.55+j*.05,.85-.3*(1-Math.pow(.8,j)),0]);D.poly(K,BL,a*.7);D.text('kernel 0.8ʲ',.72,.95,0,a,{size:.044,c:BL});
      D.text('kernel (blue) = step by step (gold)',0,-1.04,0,a,{size:.048});}
    else{D.text('decay per token',-.9,-.86,0,a,{size:.042,align:'left',c:GD});D.text('no fixed kernel: a scan',0,.95,0,a,{size:.05,c:GD});}
    D.text('inputs',-.9,-.65,0,a,{size:.042,align:'left'});}};};

/* 18-5 · memory kept per token: the cache grows, the recurrent state stays flat. */
P.memory=()=>({palace:true,
  p(i,s,o){const cur=i%2,k=i>>1,u=k/209,rv=ease(s);if(u>rv){o.a=0;}o.x=-.85+u*1.7;o.y=cur?-.45:-.6+1.3*u;o.c=cur?GD:BL;o.r=.9;},
  draw(s,a,D){D.line(-.9,-.7,0,.9,-.7,0,FG,a*.35);D.line(-.9,-.7,0,-.9,.8,0,FG,a*.35);D.text('context length →',.9,-.8,0,a,{size:.046,align:'right'});D.text('memory kept',-.88,.88,0,a,{size:.046,align:'left'});
    D.text('KV cache',.5,.5,0,a,{size:.048,c:BL});D.text('recurrent state',.5,-.35,0,a,{size:.048,c:GD});}});

/* 27 · a 20 × 20 state matrix S, written with outer products. */
const KEYS=[],VALS=[];for(let w=0;w<16;w++){const k=[],v=[];for(let j=0;j<20;j++){k.push(G(w*20+j,711));v.push(G(w*20+j,713));}const nk=Math.hypot(...k);KEYS.push(k.map(a=>a/nk));VALS.push(v);}
function grid(i,o){const r=Math.floor(i/20),c=i%20;o.x=(c-9.5)*.08-.12;o.y=(9.5-r)*.08;}

/* 27-1 · a one-hot key reads one column of S. */
P.slate=()=>{const S=new Float32Array(400);for(let i=0;i<400;i++)S[i]=G(i,721)*.6;let col=0;return{palace:true,
  prep(s){col=Math.floor(sstep(.05,.95,s)*19.99);},
  p(i,s,o){if(i<400){grid(i,o);const c=i%20,v=S[i];o.r=.3+1.2*Math.abs(v);o.c=c===col?BL:FT;o.a=c===col?1:.55;return;}
    const r=i-400;o.x=.85;o.y=(9.5-r)*.08;const v=S[r*20+col];o.r=.4+1.8*Math.abs(v);o.c=v>0?BL:GD;},
  draw(s,a,D){const x=(col-9.5)*.08-.12;D.line(x,.95,0,x,.82,0,BL,a);D.text('key',x,1.02,0,a,{size:.046,c:BL});D.text('S k',.85,.92,0,a,{size:.046});D.text('S',-1.0,0,0,a,{size:.06,c:FG});}};};

/* 27-2 · linear attention fills S with chalk; decay fades the oldest writes. */
P.chalk=()=>{const S=new Float32Array(400);let n=0,decay=false;return{palace:true,
  prep(s){decay=s>=.5;const u=decay?(s-.5)*2:s*2;n=Math.floor(ease(u)*16);S.fill(0);const al=decay?.95:1;for(let w=0;w<n;w++){for(let q=0;q<400;q++)S[q]*=al;const k=KEYS[w],v=VALS[w];for(let r=0;r<20;r++)for(let c=0;c<20;c++)S[r*20+c]+=v[r]*k[c];}},
  p(i,s,o){if(i>=400){o.a=0;return;}grid(i,o);const v=S[i]/2.2;o.r=.25+1.6*Math.min(1,Math.abs(v));o.c=v>0?BL:GD;o.a=.3+.7*Math.min(1,Math.abs(v));},
  draw(s,a,D){D.text((decay?'with decay 0.95 · ':'only adding · ')+n+' pairs written',0,.98,0,a,{size:.05});D.text(decay?'old writes fade':'the slate turns grey',0,-.98,0,a,{size:.05,c:decay?BL:GD});}};};

/* 27-4 · the gated delta rule, one chunk of four writes at a time. */
P.chunks=()=>{const al=.9,be=1;const SS=[];{const S=new Float32Array(400);SS.push(S.slice());for(let w=0;w<16;w++){for(let q=0;q<400;q++)S[q]*=al;const k=KEYS[w],v=VALS[w];
    for(let r=0;r<20;r++){let sk=0;for(let c=0;c<20;c++)sk+=S[r*20+c]*k[c];const e=be*(v[r]-sk);for(let c=0;c<20;c++)S[r*20+c]+=e*k[c];}if(w%4===3)SS.push(S.slice());}}
  let ch=0;return{palace:true,
  prep(s){ch=Math.min(4,Math.floor(sstep(.05,.95,s)*4.99));},
  p(i,s,o){if(i<400){grid(i,o);o.y+=.12;const v=SS[ch][i]/1.5;o.r=.25+1.5*Math.min(1,Math.abs(v));o.c=v>0?BL:GD;o.a=.3+.7*Math.min(1,Math.abs(v));return;}
    const j=i-400;if(j>=16){o.a=0;return;}o.x=-.72+j*.1+Math.floor(j/4)*.04;o.y=-.85;o.c=Math.floor(j/4)<ch?GD:FT;o.r=1.6;},
  draw(s,a,D){for(let c=0;c<4;c++){const x0=-.77+c*.44,x1=x0+.38;D.poly([[x0,-.78,0],[x1,-.78,0],[x1,-.92,0],[x0,-.92,0],[x0,-.78,0]],c<ch?GD:FT,a*(c===ch-1?1:.5));}
    D.text(ch?'chunk '+ch+' of 4: one triangular solve':'empty state',0,1.04,0,a,{size:.05});}};};

/* 27-5 · recall against pairs written, 32 × 32 state, random keys (module 27 table). */
P.recall=()=>{const NP=[8,16,32,64,128],R=[['linear',[1,1,.99,.83,.46],FG],['delta',[.99,.98,.81,.47,.22],BL],['decay',[1,1,.75,.4,.2],FT],['gated delta',[.99,.91,.56,.28,.14],GD]];
  const X=k=>-.8+k*.4,Y=v=>-.7+v*1.4;const at=(r,u)=>{const k=Math.min(3,Math.floor(u*4)),f=u*4-k;return Y(lerp(r[1][k],r[1][k+1],f));};
  return{palace:true,
  p(i,s,o){const c=i%4,k=Math.floor(i/4),u=k/104;if(u>ease(s)+.001){o.a=0;}o.x=-.8+u*1.6;o.y=at(R[c],u);o.c=R[c][2];o.r=.9;},
  draw(s,a,D){NP.forEach((n,k)=>D.text(String(n),X(k),-.82,0,a,{size:.042}));D.text('pairs written',.9,-.95,0,a,{size:.044,align:'right'});D.text('share read back right',-.9,.88,0,a,{size:.044,align:'left'});
    R.forEach((r,j)=>D.text(r[0],.86,Y(r[1][4])+(j-1.5)*.07,0,a*sstep(.8,1,s),{size:.042,align:'left',c:r[2]===FT?-1:r[2]}));}};};

/* 09 · the diffusion formations in a 3-D fog: noise spreads in depth as well. */
function abOf(mode,s){return mode==='forward'?1-.98*ease(s):mode==='reverse'?.02+.98*ease(s):.02;}
const fogDepth=mode=>(i,s,o)=>Math.sqrt(1-abOf(mode,s))*G(i,117)*.38+Math.sqrt(abOf(mode,s))*(((i*.618)%1)-.5)*.12;

/* 09-4 · many short steps against few long strides, from the same noise to the same spiral. */
function spiralPt(i){const u=(i+.5)/N;const th=u*4.4*Math.PI+.4;const r=.1+.86*u;return [r*Math.cos(th)+G(i,101)*.012,r*Math.sin(th)+G(i,103)*.012];}
P.strides=()=>({palace:true,
  p(i,s,o){const [x,y]=spiralPt(i);const ex=G(i,113)*.32,ey=G(i,115)*.32;const ddim=i%2;let t=ease(s),jx=0,jy=0;
    if(ddim){const k=Math.min(9,Math.floor(t*10)),f=t*10-k;t=t>=1?1:(k+sstep(.55,1,f))/10;}
    else{const k=Math.floor(t*50);const w=(1-t)*.03;jx=G(i,731+k)*w;jy=G(i,733+k)*w;}
    const ab=.02+.98*clamp(t);const a=Math.sqrt(ab),b=Math.sqrt(1-ab);o.x=a*x+b*ex+jx;o.y=a*y+b*ey+jy;o.c=ddim?GD:BL;o.r=.9;},
  draw(s,a,D){D.text('blue: many short steps · gold: 10 strides',0,-1.04,0,a,{size:.048});}});

/* 09-5 · guidance bends the clearing toward one labelled cloud. */
P.guide=()=>{const CAT=[-.45,.25],DOG=[.45,-.2];let w=0;return{palace:true,
  prep(s){w=s<.35?0:lerp(1,3,ease((s-.35)/.65));},
  p(i,s,o){const t=s<.35?ease(s/.35):1;const toCat=i%2===0||w>0;const tgt=(i%2===0||(w>0&&H(i,741)<sig(2*w-2.5)))?CAT:DOG;
    const spread=.16/(1+w*.12);const gx=tgt[0]+G(i,743)*spread,gy=tgt[1]+G(i,745)*spread;const nx=G(i,747)*.5,ny=G(i,749)*.5;
    o.x=lerp(nx,gx,t);o.y=lerp(ny,gy,t);o.c=tgt===CAT?GD:BL;o.r=.9;},
  draw(s,a,D){D.text('"a cat"',CAT[0],CAT[1]+.32,0,a,{size:.05,c:GD});D.text('"a dog"',DOG[0],DOG[1]-.34,0,a,{size:.05,c:BL});D.text(w?'guidance w = '+w.toFixed(1)+': prompt "a cat"':'no guidance',0,-1.02,0,a,{size:.05});}};};

/* 33-1 · flow matching: straight strings from noise to data; one step lands on the average. */
P.flow=()=>({palace:true,
  p(i,s,o){const [x,y]=spiralPt(i);const nx=G(i,751)*.42,ny=G(i,753)*.42;if(s<.62){const t=ease(s/.62);o.x=lerp(nx,x,t);o.y=lerp(ny,y,t);o.c=BL;o.r=.9;return;}
    const t=ease(sstep(.66,.9,s));o.x=lerp(nx,G(i,755)*.03,t);o.y=lerp(ny,G(i,757)*.03,t);o.c=GD;o.r=.9;},
  draw(s,a,D){if(s<.62){for(let i=0;i<N;i+=14){const [x,y]=spiralPt(i);D.line(G(i,751)*.42,G(i,753)*.42,0,x,y,0,FT,a*.5);}D.text('many steps: along straight strings',0,-1.02,0,a,{size:.05});}
    else D.text('one step: every point lands on the average',0,-1.02,0,a,{size:.05,c:GD});}});

/* 33-3 · the same line filled in 1, 8 and 128 steps. */
P.dial=()=>{const L=[['128 steps','the cat sat down',''],['8 steps','the cat sat dwwn','..............x.'],['1 step','noe aedwe ttEres','xxxxxxxxxxxxxxxx']];
  const ORD=[...Array(16).keys()].sort((p,q)=>H(p,761)-H(q,761));const RK=new Array(16);ORD.forEach((j,r)=>RK[j]=r);
  const steps=[128,8,1];let t=0;return{palace:true,
  prep(s){t=ease(sstep(.05,.85,s));},
  p(i,s,o){const row=Math.floor(i/140);if(row>2){o.a=0;return;}const k=i%140,j=Math.floor(k/8);if(j>15){o.a=0;return;}const m=k%8;
    o.x=-.72+j*.1+(m%4-1.5)*.016;o.y=.45-row*.55-.12-Math.floor(m/4)*.016;const shownAt=Math.ceil((RK[j]+1)/16*steps[row])/steps[row];const on=t>=shownAt;o.c=on?(L[row][2][j]==='x'?GD:BL):FT;o.r=.7;o.a=on?1:.4;},
  draw(s,a,D){for(let row=0;row<3;row++){D.text(L[row][0],-.78,.45-row*.55+.1,0,a,{size:.044,align:'right'});
      for(let j=0;j<16;j++){const shownAt=Math.ceil((RK[j]+1)/16*steps[row])/steps[row];const on=t>=shownAt;const ch=L[row][1][j];D.text(on?(ch===' '?'·':ch):'_',-.72+j*.1,.45-row*.55,0,a,{size:.06,c:on?(L[row][2][j]==='x'?GD:FG):-1});}}
    D.text('gold: wrong letters',0,-1.02,0,a*sstep(.85,1,s),{size:.046,c:GD});}};};

/* 33-4 · block diffusion: four blocks of four tiles, each unmasked in place, then cached. */
P.blocks=()=>{const TX='the cat sat down';const ORD=[...Array(16).keys()].sort((p,q)=>H(p,771)-H(q,771));const RK=new Array(16);ORD.forEach((j,r)=>RK[j]=r%4);
  let t=0;return{palace:true,
  prep(s){t=sstep(.02,.95,s)*4.99;},
  p(i,s,o){const j=Math.floor(i/26);if(j>15){o.a=0;return;}const m=i%26;const blk=Math.floor(j/4),cur=Math.floor(t);const f=t-cur;
    const done=blk<cur,act=blk===cur,rev=done||(act&&RK[j]<f*4);const drop=done?1:0;
    o.x=-.8+j*.1+Math.floor(j/4)*.04+((m%5)-2)*.012;o.y=(drop?-.45:.2)-Math.floor(m/5)*.012;o.c=done?FT:(rev?BL:FG);o.a=rev||act?1:.35;o.r=.6;},
  draw(s,a,D){const cur=Math.floor(t),f=t-cur;for(let j=0;j<16;j++){const blk=Math.floor(j/4);const done=blk<cur,act=blk===cur,rev=done||(act&&RK[j]<f*4);
      D.text(rev?(TX[j]===' '?'·':TX[j]):'_',-.8+j*.1+blk*.04,(done?-.45:.2)+.12,0,a,{size:.06,c:done?-1:FG});}
    D.text('KV cache: finished blocks',-.85,-.72,0,a,{size:.044,align:'left'});D.text(cur<4?'block '+(cur+1)+': unmask in place':'line done',0,.72,0,a,{size:.05,c:BL});}};};

/* 33-5 · held-out loss against epochs on 100,000 characters (module 33, section 5). */
P.repeat=()=>{const LX=e=>-.85+Math.log(e)/Math.log(200)*1.7,LY=v=>-.75+(v-2.2)/1.65*1.5;
  const ar=e=>e<=64?2.287+1.45*Math.pow(1-Math.log(e)/Math.log(64),1.6):2.287+.251*Math.pow((e-64)/136,1.3);
  const dm=e=>e<=30?3.77-.41*Math.log(e)/Math.log(30):3.36-(3.36-2.498)*Math.pow((Math.log(e)-Math.log(30))/(Math.log(200)-Math.log(30)),.9);
  return{palace:true,
  p(i,s,o){const c=i%2,k=i>>1,e=Math.exp(k/209*Math.log(200));if(k/209>ease(s)+.001)o.a=0;o.x=LX(e);o.y=LY(c?dm(e):ar(e));o.c=c?GD:BL;o.r=.9;},
  draw(s,a,D){[1,10,64,200].forEach(e=>D.text(String(e),LX(e),-.87,0,a,{size:.042}));D.text('epochs (log) →',.85,-1.0,0,a,{size:.044,align:'right'});D.text('held-out loss',-.85,.9,0,a,{size:.044,align:'left'});
    D.text('left to right',LX(20),LY(ar(20))+.12,0,a,{size:.044,c:BL});D.text('diffusion',LX(12),LY(3.36)+.1,0,a,{size:.044,c:GD});
    if(s>.85)D.text('they cross at 200; the stopped twin still leads',0,.62,0,a*sstep(.85,1,s),{size:.044});}};};

/* Placeholder for rooms furnished in a later phase: a slow ring and a post. */
P.marker=()=>({palace:true,placeholder:true,
  p(i,s,o){if(i>=48){o.a=0;return;}const th=i/48*TAU+s*TAU*.25;o.x=.32*Math.cos(th);o.z=.32*Math.sin(th);o.y=-.4;o.c=FT;o.r=.9;o.a=.9;},
  draw(s,a,D){D.line(0,-.9,0,0,-.4,0,FT,a*.6);}});

/* Prologue at the garden gate: the Moves page's first four moves. Epilogue at the lookout: its motifs. */
P.prologue=()=>[F.scatter(),F.grid(),F.clouds('plain'),F.clouds('wrong')];

const OBJ={
  '00a-4':{make:()=>P.sundial(),kind:'3d',base:1.15},
  '00b-2':{make:()=>P.naive()},
  '00b-5':{make:()=>P.kmeans()},
  '00c-1':{make:()=>P.caption(F.clouds('margin'),'three stones hold the plank')},
  '00c-2':{make:()=>P.lift(),kind:'3d',base:1.0},
  '00c-3':{kind:'3d',base:.9,floor:{h:(x,y)=>{const e=1-.95*Math.exp(-((x+.48)**2+(y-.05)**2)/.16)-.85*Math.exp(-((x-.55)**2+(y+.2)**2)/.13);return e*.75-.3;}},note:'basins() laid on the floor and lifted by a two-well energy'},
  '00c-5':{},
  '03-4':{make:()=>[P.caption(F.tokens(0,3),'encode: merge'),P.caption(F.tokens(3,0),'decode: the same 16 bytes',GD)]},
  '03-5':{make:()=>P.caption(P.fertility(),'pangrams: 1.72× a word · five plain sentences: 1.36×')},
  '08-1':{make:()=>P.conv()},
  '08-2':{make:()=>P.filters()},
  '08-4':{make:()=>P.attnmap()},
  '08-5':{make:()=>P.clip(),kind:'3d',base:1.35},
  '05-1':{make:()=>P.params(),kind:'3d',base:1.3},
  '05-2':{kind:'3d',depth:(i,s,o)=>G(i,45)*.16+[.3,-.25,.1,-.35,.4,-.1][i%6]},
  '05-5':{make:()=>P.lens(),kind:'3d',base:1.35},
  '05-6':{make:()=>P.caption(F.tokline(),'the model was trained on this exact wrapper')},
  '07-1':{make:()=>P.manifest()},
  '07-3':{make:()=>P.prefill()},
  '07-4':{make:()=>P.caption(F.bars('temp'),'trim the bag, then draw')},
  '07-5':{make:()=>P.caption(F.quant('hist'),'255 shades per row')},
  '07-6':{make:()=>P.parity()},
  '18-1':{make:()=>P.gru()},
  '18-3':{make:()=>P.scan(false)},
  '18-4':{make:()=>P.scan(true)},
  '18-5':{make:()=>P.memory()},
  '27-1':{make:()=>P.slate()},
  '27-2':{make:()=>P.chalk()},
  '27-4':{make:()=>P.chunks()},
  '27-5':{make:()=>P.recall()},
  '09-1':{kind:'3d',depth:fogDepth('forward')},
  '09-2':{kind:'3d',depth:fogDepth('noise')},
  '09-3':{kind:'3d',depth:fogDepth('reverse')},
  '09-4':{make:()=>P.strides()},
  '09-5':{make:()=>P.guide()},
  '33-1':{make:()=>P.flow()},
  '33-3':{make:()=>P.dial()},
  '33-4':{make:()=>P.blocks()},
  '33-5':{make:()=>P.repeat()}
};

window.PalaceObjects={P,OBJ};
})();
