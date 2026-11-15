/* Palace objects: the formations the Moves page does not have, and how each furnished
   object stands in its room.

   A palace formation has the Moves shape {prep?(s), p(i,s,o), draw?(s,a,D)} with two
   additions: p may set o.z (unit depth, toward the viewer), and draw gets D, which
   draws in three dimensions: D.line(x0,y0,z0,x1,y1,z1,c,a), D.poly(P,c,a) with
   P = [[x,y,z],...], D.text(t,x,y,z,a,{c,size,align}). Units: the Moves unit square,
   about -1..1, y up; one unit is SPEC.scale metres in the room.

   Which formation a locus plays is the 'formation' expression in source/loci.json; OBJ (at the end)
   says only how a locus stands: kind, base height, scale, and caption fixes for its room.
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
  draw(s,a,D){{const c=[at(-.95,-.95),at(.95,-.95),at(.95,.95),at(-.95,.95)];D.poly([...c,c[0]],FG,a*.5);}D.line(Y[0],Y[1],Y[2],FOOT[0],FOOT[1],FOOT[2],BL,a*.8);
    const u=[.07,0,0],v=[e2[0]*.07,e2[1]*.07,e2[2]*.07];const nrm=[Y[0]-FOOT[0],Y[1]-FOOT[1],Y[2]-FOOT[2]];const L=Math.hypot(...nrm);const m=[nrm[0]/L*.07,nrm[1]/L*.07,nrm[2]/L*.07];
    D.line(FOOT[0]+u[0],FOOT[1]+u[1],FOOT[2]+u[2],FOOT[0]+u[0]+m[0],FOOT[1]+u[1]+m[1],FOOT[2]+u[2]+m[2],FG,a*.7);
    D.line(FOOT[0]+m[0],FOOT[1]+m[1],FOOT[2]+m[2],FOOT[0]+u[0]+m[0],FOOT[1]+u[1]+m[1],FOOT[2]+u[2]+m[2],FG,a*.7);
    const k0=Math.min(12,Math.floor(t));const Q=[];for(let k=0;k<=k0;k++){const w=path[k];if(Math.abs(w[0])>1.6||Math.abs(w[1])>1.6)break;Q.push(at(w[0],w[1]));}
    D.poly(Q,GD,a*.75);
    D.text('target y',Y[0]+.08,Y[1]+.1,Y[2],a,{size:.05,align:'left'});
    D.text('fit Xw',FOOT[0]+.1,FOOT[1]-.1,FOOT[2],a,{size:.05,align:'left',c:BL});
    D.text(stable?'step below the bound: settles':'step past the bound: flies off',0,-1.02,0,a,{size:.05,c:stable?FG:GD});}};};

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
/* 17-1 · Bradley-Terry: six items start level; each choice lifts the winner's score and lowers the loser's, and the hidden order comes out. No numbers: the motion is the lesson. */
P.bt=()=>{const ST=[.2,-1,1.1,-.1,.6,-.5],PAIR=[[0,1],[2,3],[4,5],[1,2],[3,4],[5,0],[0,2],[1,3],[2,4],[3,5],[4,0],[5,1]];let t=0,pr=PAIR[0];const X=j=>(j-2.5)*.33,Y=j=>ST[j]*.6*t;return{palace:true,
  prep(s){t=ease(sstep(.08,.85,s));pr=PAIR[Math.min(PAIR.length-1,Math.floor(s*PAIR.length))];},
  p(i,s,o){const j=i%6,k=Math.floor(i/6),a=H(k,1201)*TAU,r=.075*Math.sqrt(H(k,1203));o.x=X(j)+r*Math.cos(a);o.y=Y(j)+r*Math.sin(a);const on=j===pr[0]||j===pr[1];o.c=on?(ST[pr[0]]>ST[pr[1]]===(j===pr[0])?BL:GD):FT;o.r=on?1:.75;},
  draw(s,a,D){D.line(-.95,0,0,.95,0,0,FG,a*.3);const [p,q]=pr,w=ST[p]>ST[q]?p:q,l=w===p?q:p;D.line(X(p),Y(p),0,X(q),Y(q),0,FG,a*.6);
    if(s<.9){D.line(X(w),Y(w)+.1,0,X(w),Y(w)+.24,0,BL,a);D.line(X(w)-.04,Y(w)+.19,0,X(w),Y(w)+.24,0,BL,a);D.line(X(w)+.04,Y(w)+.19,0,X(w),Y(w)+.24,0,BL,a);
      D.line(X(l),Y(l)-.1,0,X(l),Y(l)-.24,0,GD,a);D.line(X(l)-.04,Y(l)-.19,0,X(l),Y(l)-.24,0,GD,a);D.line(X(l)+.04,Y(l)-.19,0,X(l),Y(l)-.24,0,GD,a);}
    D.text('P(i beats j) = σ(sᵢ − sⱼ)',0,.95,0,a,{size:.05});D.text(s<.9?'each choice moves two scores':'the order, from choices alone',0,-.92,0,a,{size:.048});}};};
/* 29-5 · units that go dead and stay dead under Adam; continual backprop swaps the least useful for fresh ones (module 29, section 5: 62.7% against 23.5% dead by the last task). */
P.units=()=>{let t=0;const U=70,DA=.627,DC=.235;return{palace:true,
  prep(s){t=sstep(.05,.8,s);},
  p(i,s,o){const u=i%140,k=Math.floor(i/140),side=u<U?0:1,v=u%U,c=v%7,r=Math.floor(v/7),rk=H(v,1211);o.x=(side?.14:-.86)+c*.12+(k-1)*.022;o.y=.6-r*.125;
    const dies=rk<DA*t,stays=side?rk<DC*t:dies,fresh=side&&dies&&!stays&&rk>DA*t-.07;o.c=stays?FT:(fresh?GD:BL);o.r=stays?.45:(fresh?1.2:.9);o.a=stays?.45:1;},
  draw(s,a,D){D.text('Adam',-.5,-.72,0,a,{size:.05});D.text('continual backprop',.5,-.72,0,a,{size:.05});D.text('62.7% dead',-.5,-.86,0,a*sstep(.75,.9,s),{size:.046,c:FT});D.text('23.5% dead',.5,-.86,0,a*sstep(.75,.9,s),{size:.046,c:BL});
    D.text(s<.8?'task after task, units stop firing':'the least useful units are swapped for fresh ones',0,.9,0,a,{size:.046});}};};

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

/* 08-5 · CLIP: one bean-leaf photo and the module's three symptom prompts on a sphere; the nearest prompt lights (results/clip_zeroshot.json). */
P.clip=()=>{const CAP=['grey angular dead patches','orange-brown rust pustules','green healthy leaf'];const CP=[];for(let j=0;j<3;j++){const th=j/3*TAU+.3,ph=(H(j,641)-.5)*1.2;CP.push([Math.cos(ph)*Math.cos(th),Math.sin(ph),Math.cos(ph)*Math.sin(th)]);}
  const IMG0=[.2,.8,-.56],IMG1=(()=>{const c=CP[0];return [c[0]*.9+.1*IMG0[0],c[1]*.9+.1*IMG0[1],c[2]*.9+.1*IMG0[2]];})();
  const nrm=v=>{const L=Math.hypot(...v);return v.map(x=>x/L);};let im=IMG0,rot=0,best=0;
  return{palace:true,
  prep(s){const t=ease(sstep(0,.6,s));im=nrm([lerp(IMG0[0],IMG1[0],t),lerp(IMG0[1],IMG1[1],t),lerp(IMG0[2],IMG1[2],t)]);rot=s*.8;
    let bd=-2;CP.forEach((c,j)=>{const d=c[0]*im[0]+c[1]*im[1]+c[2]*im[2];if(d>bd){bd=d;best=j;}});},
  R(v){const c=Math.cos(rot),s=Math.sin(rot);return[v[0]*c-v[2]*s,v[1],v[0]*s+v[2]*c].map(x=>x*.82);},
  p(i,s,o){if(i<330){const u=(i+.5)/330,ph=Math.acos(1-2*u),th=i*2.39996;const q=this.R([Math.sin(ph)*Math.cos(th),Math.cos(ph),Math.sin(ph)*Math.sin(th)]);o.x=q[0];o.y=q[1];o.z=q[2];o.c=FT;o.r=.55;o.a=.7;return;}
    if(i<390){const j=Math.floor((i-330)/20),k=(i-330)%20;const q=this.R(CP[j]);o.x=q[0]+.03*Math.cos(k*.315);o.y=q[1]+.03*Math.sin(k*.315);o.z=q[2];o.c=(j===best&&s>.6)?GD:FG;o.r=1.5;return;}
    const k=i-390;const q=this.R(im);o.x=q[0]+.03*Math.cos(k*.21);o.y=q[1]+.03*Math.sin(k*.21);o.z=q[2]+.02*Math.sin(k);o.c=BL;o.r=1.6;},
  draw(s,a,D){const q=this.R(im);CP.forEach((c,j)=>{const p=this.R(c);const lit=j===best&&s>.6;D.line(q[0],q[1],q[2],p[0],p[1],p[2],lit?GD:FT,a*(lit?.9:.45));D.text(CAP[j],p[0]+.06,p[1]+.06,p[2],a,{size:.046,align:'left',c:lit?GD:-1});});
    D.text('photo',q[0]-.06,q[1]-.08,q[2],a,{size:.046,align:'right',c:BL});D.text('nearest caption wins',0,-1.0,0,a,{size:.048});}};};

/* 07-1 · one flat ribbon of numbers splits into named tensors. */
P.manifest=()=>{const parts=[['wte = head',112],['wpe',28],['L0 qkv',45],['L0 out',15],['L0 mlp',90],['L1 (same)',60],['L2 (same)',60],['ln_f',10]];/* module 07: wte, wpe, three blocks, ln_f; the LM head is wte reused. Not to scale. */
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
    if(!selective){const K=[];for(let j=0;j<8;j++)K.push([.55+j*.05,.85-.3*(1-Math.pow(.8,j)),0]);D.poly(K,BL,a*.7);D.text('fading kernel',.72,.95,0,a,{size:.044,c:BL});
      D.text('kernel (blue) = step by step (gold)',0,-1.04,0,a,{size:.048});}
    else{D.text('decay per token',-.9,-.86,0,a,{size:.042,align:'left',c:GD});D.text('no fixed kernel',0,.95,0,a,{size:.05,c:GD});}
    D.text('inputs',-.9,-.65,0,a,{size:.042,align:'left'});}};};

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

/* 09 · the diffusion formations in a 3-D fog: noise spreads in depth as well. */
function abOf(mode,s){return mode==='forward'?1-.98*ease(s):mode==='reverse'?.02+.98*ease(s):.02;}
const fogDepth=mode=>(i,s,o)=>Math.sqrt(1-abOf(mode,s))*G(i,117)*.38+Math.sqrt(abOf(mode,s))*(((i*.618)%1)-.5)*.12;

/* the spiral every diffusion object clears toward */
function spiralPt(i){const u=(i+.5)/N;const th=u*4.4*Math.PI+.4;const r=.1+.86*u;return [r*Math.cos(th)+G(i,101)*.012,r*Math.sin(th)+G(i,103)*.012];}

/* 09-5 · guidance bends the clearing toward one labelled cloud. */
P.guide=()=>{const CAT=[-.45,.25],DOG=[.45,-.2];let w=0;return{palace:true,
  prep(s){w=s<.35?0:lerp(1,3,ease((s-.35)/.65));},
  p(i,s,o){const t=s<.35?ease(s/.35):1;const spread=.16/(1+w*.25);const own=i%2===0?CAT:DOG;const mv=i%2===0?0:sstep(.35,.6,s);/* w = 0: both clouds; w ≥ 1: every point goes to the prompted one (module 09, exercise c) */
    const gx=lerp(own[0],CAT[0],mv)+G(i,743)*spread,gy=lerp(own[1],CAT[1],mv)+G(i,745)*spread;const nx=G(i,747)*.5,ny=G(i,749)*.5;
    o.x=lerp(nx,gx,t);o.y=lerp(ny,gy,t);o.c=mv>.5||i%2===0?GD:BL;o.r=.9;},
  draw(s,a,D){D.text('"a cat"',CAT[0],CAT[1]+.32,0,a,{size:.05,c:GD});D.text('"a dog"',DOG[0],DOG[1]-.34,0,a,{size:.05,c:BL});D.text(w?'guidance w = '+w.toFixed(1)+': prompt "a cat"':'no guidance',0,-1.02,0,a,{size:.05});}};};

/* 33-1 · flow matching: straight strings from noise to data; one step lands on the average. */
P.flow=()=>({palace:true,
  p(i,s,o){const [x,y]=spiralPt(i);const nx=G(i,751)*.42,ny=G(i,753)*.42;if(s<.62){const t=ease(s/.62);o.x=lerp(nx,x,t);o.y=lerp(ny,y,t);o.c=BL;o.r=.9;return;}
    const t=ease(sstep(.66,.9,s));o.x=lerp(nx,G(i,755)*.03,t);o.y=lerp(ny,G(i,757)*.03,t);o.c=GD;o.r=.9;},
  draw(s,a,D){if(s<.62){for(let i=0;i<N;i+=14){const [x,y]=spiralPt(i);D.line(G(i,751)*.42,G(i,753)*.42,0,x,y,0,FT,a*.5);}D.text('many steps: along straight strings',0,-1.02,0,a,{size:.05});}
    else D.text('one step: every point lands on the average',0,-1.02,0,a,{size:.05,c:GD});}});

/* 33-3 · the same line filled in 1, 8 and 128 steps. */
P.dial=()=>{const L=[['128 steps',' tit obody waide',''],['8 steps','ounds kng in he ',''],['1 step','noe aedwe ttEres','']];/* 16-character cuts of module 33's samples (section 3) */
  const ORD=[...Array(16).keys()].sort((p,q)=>H(p,761)-H(q,761));const RK=new Array(16);ORD.forEach((j,r)=>RK[j]=r);
  const steps=[128,8,1];let t=0;return{palace:true,
  prep(s){t=ease(sstep(.05,.85,s));},
  p(i,s,o){const row=Math.floor(i/140);if(row>2){o.a=0;return;}const k=i%140,j=Math.floor(k/8);if(j>15){o.a=0;return;}const m=k%8;
    o.x=-.72+j*.1+(m%4-1.5)*.016;o.y=.45-row*.55-.12-Math.floor(m/4)*.016;const shownAt=Math.ceil((RK[j]+1)/16*steps[row])/steps[row];const on=t>=shownAt;o.c=on?(L[row][2][j]==='x'?GD:BL):FT;o.r=.7;o.a=on?1:.4;},
  draw(s,a,D){for(let row=0;row<3;row++){D.text(L[row][0],-.78,.45-row*.55+.1,0,a,{size:.044,align:'right'});
      for(let j=0;j<16;j++){const shownAt=Math.ceil((RK[j]+1)/16*steps[row])/steps[row];const on=t>=shownAt;const ch=L[row][1][j];D.text(on?(ch===' '?'·':ch):'_',-.72+j*.1,.45-row*.55,0,a,{size:.06,c:on?(L[row][2][j]==='x'?GD:FG):-1});}}
    D.text('the same model, three step counts',0,-1.02,0,a*sstep(.85,1,s),{size:.046});}};};

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

/* Prologue at the garden gate: the Moves page's first four moves. Epilogue at the lookout: its motifs. */
P.prologue=()=>[F.scatter(),F.grid(),F.clouds('plain'),F.clouds('wrong')];

/* 01-4 · two ways to the same slope, side by side: labels from the room (module 01: backward() and a finite difference). */
P.parity=(L={a:'engine',b:'PyTorch',done:'every value within tolerance'})=>{let t=0;return{palace:true,
  prep(s){t=ease(sstep(0,.7,s));},
  p(i,s,o){const k=i>>1,ref=i&1;const u=k/210,x=-.9+u*1.8,y=.45*Math.sin(u*9)*Math.exp(-u*.6)+G(k,671)*.05;
    o.x=ref?x+(1-t)*.9:x;o.y=y;o.c=ref?FG:BL;o.r=ref?.6:1.2;o.a=ref?.9:.75;},
  draw(s,a,D){D.text(L.a,-.9,.75,0,a,{size:.048,c:BL,align:'left'});D.text(L.b,.9,.75,0,a,{size:.048,align:'right'});
    if(s>.75)D.text(L.done,0,-.8,0,a*sstep(.75,.9,s),{size:.05,c:GD});}};};

/* 02-2 · softmax then cross-entropy: the slope on each score is p − y, down on the true class, up on the rest. No numbers: the shape is the lesson. */
P.softgrad=()=>{const PR=[.1,.18,.5,.14,.08],TRUE=2;let g=0;return{palace:true,
  prep(s){g=ease(sstep(.35,.85,s));},
  p(i,s,o){const b=i%5,k=Math.floor(i/5);if(k>=80){o.a=0;return;}const x=(b-2)*.38,h=PR[b]*1.6*ease(sstep(0,.3,s));const c=k%4,r=Math.floor(k/4);
    o.x=x+(c-1.5)*.035;o.y=-.55+(r/19)*h;o.c=b===TRUE?GD:BL;o.r=.85;},
  draw(s,a,D){for(let b=0;b<5;b++){const x=(b-2)*.38,top=-.55+PR[b]*1.6,d=(PR[b]-(b===TRUE?1:0))*g*.9;if(Math.abs(d)<.005)continue;
      const y1=top+.06,y2=y1-d;D.line(x,y1,0,x,y2,0,b===TRUE?GD:FG,a);const s2=Math.sign(-d)*.05;D.line(x,y2,0,x-.04,y2-s2,0,b===TRUE?GD:FG,a);D.line(x,y2,0,x+.04,y2-s2,0,b===TRUE?GD:FG,a);}
    D.text('true class',.0,-.68,0,a,{size:.046,c:GD});D.text(g>.5?'gradient on each score = p − y':'softmax odds p',0,.95,0,a,{size:.05});}};};

/* 05-6 · a chat template: special tokens wrap each turn (module 05: <|im_start|>, <|im_end|>; role names are ordinary tokens). */
P.chat=()=>{const T=[['<|im_start|>',1],['user',0],['your words',2],['<|im_end|>',1],['<|im_start|>',1],['assistant',0],['the reply …',2]];
  const W=T.map(t=>.1+t[0].length*.034);const pos=[];{let x=-.95,y=.35;T.forEach((t,j)=>{if(j===4){x=-.95;y=-.25;}pos.push([x+W[j]/2,y]);x+=W[j]+.05;});}
  let k=0;return{palace:true,
  prep(s){k=ease(sstep(0,.85,s))*T.length;},
  p(i,s,o){const j=Math.floor(i/60),m=i%60;if(j>=T.length){o.a=0;return;}const [cx,cy]=pos[j],w=W[j],h=.18;const u=m/60*2*(w+h);let x,y;
    if(u<w){x=cx-w/2+u;y=cy+h/2;}else if(u<w+h){x=cx+w/2;y=cy+h/2-(u-w);}else if(u<2*w+h){x=cx+w/2-(u-w-h);y=cy-h/2;}else{x=cx-w/2;y=cy-h/2+(u-2*w-h);}
    o.x=x;o.y=y;o.c=T[j][1]===1?GD:(T[j][1]===0?BL:FG);o.r=.7;o.a=clamp(k-j);},
  draw(s,a,D){T.forEach((t,j)=>D.text(t[0],pos[j][0],pos[j][1],0,a*clamp(k-j),{size:.042,c:t[1]===1?GD:(t[1]===0?BL:-1)}));
    D.text('special tokens mark each turn',0,-.75,0,a*sstep(.85,1,s),{size:.05,c:GD});}};};

/* 19 · the six runs of module 19 (results/train_d*.json, held-out loss against tokens seen); N without embeddings (results/fit.json). */
const C19=[[[32768,4.937],[61440,4.902],[126976,4.751],[249856,4.503],[376832,4.149],[499712,3.799],[749568,3.357],[999424,3.132],[1499136,2.847],[1998848,2.683],[2998272,2.547],[4001792,2.492]],[[32768,4.917],[61440,4.839],[126976,4.627],[249856,4.302],[376832,3.856],[499712,3.471],[749568,3.064],[999424,2.857],[1499136,2.647],[1998848,2.553],[2998272,2.469],[4001792,2.43]],[[32768,4.905],[61440,4.79],[126976,4.566],[249856,4.172],[376832,3.642],[499712,3.244],[749568,2.857],[999424,2.693],[1499136,2.551],[1998848,2.485],[2998272,2.424],[4001792,2.38]],[[32768,4.844],[61440,4.631],[126976,4.373],[249856,3.807],[376832,3.229],[499712,2.889],[749568,2.635],[999424,2.554],[1499136,2.469],[1998848,2.429],[2998272,2.34],[4001792,2.239]],[[32768,4.619],[61440,4.281],[126976,3.958],[249856,3.231],[376832,2.769],[499712,2.595],[749568,2.497],[999424,2.451],[1499136,2.404],[1998848,2.302],[2998272,2.147],[4001792,2.031]],[[32768,4.351],[61440,3.904],[126976,3.395],[249856,2.763],[376832,2.556],[499712,2.49],[749568,2.436],[999424,2.387],[1499136,2.261],[1998848,2.122],[2998272,1.977],[4001792,1.855]]];
const N19=[50880,113184,200064,447552,1237440,3159552],AL19=.0737734;/* α from results/fit.json; the line passes through the points' mean in log space, as least squares does */
const A19=Math.exp(C19.reduce((s,c,j)=>s+Math.log(c[11][1])+AL19*Math.log(N19[j]),0)/6);
const L10=Math.log10,COL19=j=>j<2?FT:(j<4?FG:BL);
function along19(P,u){/* P: [[x,y]...] in plot units; u in [0,1] by index */const f=u*(P.length-1),k=Math.min(P.length-2,Math.floor(f)),t=f-k;return[lerp(P[k][0],P[k+1][0],t),lerp(P[k][1],P[k+1][1],t)];}
P.power19=(mode)=>{const LY=v=>-.62+(v-1.8)/(5-1.8)*1.4;let curves,env=null,head,xl;
  if(mode==='curves'){const X=t=>-.88+(L10(t)-4.5)/(6.65-4.5)*1.76;curves=C19.map(c=>c.map(([t,l])=>[X(t),LY(l)]));xl='tokens seen (log) →';head='six sizes, held-out loss: none has stopped falling';}
  else if(mode==='envelope'){const X=c=>-.88+(L10(c)-9.9)/(13.15-9.9)*1.76;curves=C19.map((c,j)=>c.map(([t,l])=>[X(6*N19[j]*t),LY(l)]));xl='compute 6·N·D (log) →';head='best size at each budget: a staircase where runs end';
    env=[];for(let x=-.88;x<=.88;x+=.01){let m=null;curves.forEach(P=>{if(x<P[0][0]||x>P[P.length-1][0])return;for(let k=0;k<P.length-1;k++)if(x>=P[k][0]&&x<=P[k+1][0]){const y=lerp(P[k][1],P[k+1][1],(x-P[k][0])/(P[k+1][0]-P[k][0]));if(m===null||y<m)m=y;}});if(m!==null)env.push([x,m]);}}
  const NX=n=>-.75+(L10(n)-4.6)/(6.6-4.6)*1.5,NY=l=>-.55+(L10(l)-L10(1.8))/(L10(2.6)-L10(1.8))*1.3;
  return{palace:true,
  p(i,s,o){const j=i%6,k=Math.floor(i/6),u=k/69;o.r=.85;o.c=COL19(j);
    if(mode==='line'){const t=ease(s);const L=C19[j][11][1];const a=k/69*TAU;const tx=NX(N19[j])+.03*Math.cos(a*3),ty=NY(L)+.03*Math.sin(a*3);
      const P=curves19(j);const q=along19(P,u);o.x=lerp(q[0],tx,t);o.y=lerp(q[1],ty,t);if(t>.5)o.r=1.1;return;}
    const P=curves[j];const rv=mode==='curves'?ease(s):1;const q=along19(P,Math.min(u,rv));o.x=q[0];o.y=q[1];if(u>rv)o.a=0;
    if(env){const e=env.find(p=>Math.abs(p[0]-q[0])<.006);o.c=(e&&q[1]-e[1]<.01)?GD:FT;}},
  draw(s,a,D){const ax=(x,y)=>{D.line(-.9,-.66,0,.92,-.66,0,FG,a*.4);D.line(-.9,-.66,0,-.9,.82,0,FG,a*.4);D.text(x,.92,-.88,0,a,{size:.044,align:'right'});D.text(y,-.88,.88,0,a,{size:.044,align:'left'});};
    if(mode==='line'){ax('size N (log) →','loss at 4M tokens (log)');const t=ease(s);const y=n=>NY(A19*Math.pow(n,-AL19));D.line(NX(40000),y(40000),0,NX(4e6),y(4e6),0,GD,a*t);
      N19.forEach((n,j)=>D.text(j===0?'51k':j===5?'3.2M':'',NX(n),-.74,0,a,{size:.042}));D.text('slope α = 0.074, R² 0.962: the points bend about the line',0,.98,0,a*sstep(.6,.9,s),{size:.044,c:GD});return;}
    ax(xl,'held-out loss');curves.forEach((P,j)=>{const rv=mode==='curves'?ease(s):1;const Q=[];for(let k=0;k<=40*rv;k++){const q=along19(P,k/40);Q.push([q[0],q[1],0]);}D.poly(Q,COL19(j),a*.6);});
    if(env)D.poly(env.map(p=>[p[0],p[1],0]),GD,a*.9);D.text(head,0,.98,0,a*sstep(.5,.9,s),{size:.044});}};
  function curves19(j){const X=t=>-.88+(L10(t)-4.5)/(6.65-4.5)*1.76;return C19[j].map(([t,l])=>[X(t),LY(l)]);}};

/* 13-1 · one unit among its neighbours: 21 lenses, channels 45 to 65 of InceptionV1's layer 4e, hang from one rail. Each drops by
   its gap (median on drawn Vs minus the larger of the cat and car medians, over the pooled spread): every gap is below zero,
   and 4e:55, the unit read as a V detector, hangs lowest of all. Gaps: module 13, results/v_neuron.json, gap_by_channel. */
P.lenses=()=>{const GAP=[-1.5761,-3.2873,-1.6241,-.4816,-3.4195,-.4974,-1.7052,-.2089,-1.7011,-3.1674,-3.482,-1.8311,-.5496,-3.2291,-1.4336,-2.0971,-.7487,-2.7397,-.6552,-1.2193,-.3272];
  const RAIL=.62,X=i=>-.9+i*1.8/20,Y=i=>RAIL-.16+GAP[i]*.35;let t=0;return{palace:true,
  prep(s){t=ease(sstep(.05,.7,s));},
  p(i,s,o){const j=Math.floor(i/20),k=i%20;if(j>20){o.a=0;return;}const a=k/20*TAU,r=.038;o.x=X(j)+r*Math.cos(a);o.y=lerp(RAIL-.16,Y(j),t)+r*1.25*Math.sin(a);o.c=j===10?GD:FT;o.r=j===10?1.15:.8;o.a=j===10?1:.8;},
  draw(s,a,D){D.line(-.97,RAIL,0,.97,RAIL,0,FG,a*.7);D.text('gap 0',-.97,RAIL+.07,0,a,{size:.042,align:'left'});
    for(let j=0;j<21;j++)D.line(X(j),RAIL,0,X(j),lerp(RAIL-.16,Y(j),t)+.048,0,j===10?GD:FT,a*(j===10?.9:.5));
    D.text('4e:55',X(10)+.06,Y(10)-.02,0,a*sstep(.6,.85,s),{size:.046,c:GD,align:'left'});
    D.text('InceptionV1 4e · channels 45 to 65',0,.9,0,a,{size:.046});D.text('lower: photos beat drawn Vs by more',0,-1.05,0,a*sstep(.5,.8,s),{size:.044});}};};

/* 13-5 · bars that may go below zero: the zero line sits at spec.base (module 13: two R² below zero). */
P.vbarsz=(spec)=>{const nb=spec.bars.length,per=Math.floor(N/nb),cols=spec.cols||6,rows=Math.ceil(per/cols),Y0=spec.base;let hs=[];const SC=.9;return{palace:true,
  prep(s){const t=ease(s);hs=spec.bars.map(b=>b.v*t/spec.max);},
  p(i,s,o){const b=Math.floor(i/per);if(b>=nb){o.a=0;o.x=0;o.y=Y0;return;}const k=i%per,c=k%cols,r=Math.floor(k/cols);const B0=spec.bars[b];
    o.x=B0.x+(c-(cols-1)/2)*.032;o.y=Y0+hs[b]*SC*(r/(rows-1));o.c=B0.c;o.r=.85;},
  draw(s,a,D){D.line(-.95,Y0,0,.95,Y0,0,FG,a*.45);D.text('0',-.97,Y0,0,a,{size:.04,align:'right'});
    spec.bars.forEach((B0,j)=>{const y=Y0+hs[j]*SC;D.text(B0.top,B0.x,y+(B0.v<0?-.08:.08),0,a*sstep(.6,.95,s),{size:.048,c:B0.c===FT?-1:B0.c});if(B0.lab)D.text(B0.lab,B0.x,-.74,0,a,{size:.042});});
    (spec.groups||[]).forEach(g=>D.text(g[1],g[0],-.88,0,a,{size:.046}));if(spec.head)D.text(spec.head,0,.95,0,a,{size:.046});}};};

/* How each locus stands in its room, keyed by its phase 1 id ('from' in source/loci.json). The formation itself is the
   'formation' expression in the data. kind '3d' (the camera looks down at it, it turns slowly) is kept for the two ideas
   that are spatial: least squares as a projection, and the kernel lift. Everything else is a standing panel.
   relabel: [[pattern, replacement or null to drop]] applied to the formation's captions in this room only. */
const OBJ={
  '00a-4':{kind:'3d',base:1.2,scale:1.25},
  '00c-2':{kind:'3d',base:1.05,scale:1.25},
  '08-5':{base:1.4},
  '13-3':{relabel:[[/^layer (\d+) of 32$/,m=>'block '+Math.max(1,Math.round(m[1]*12/32))+' of 12']]},
  '34-4':{relabel:[[/candidates$/,null]]},
  '21-1':{relabel:[[/·/,null]]}
};

window.PalaceObjects={P,OBJ};
})();
