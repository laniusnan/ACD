#!/usr/bin/env python3
"""Render two editable CVPR-style mechanism figures from the read-only probe."""
import ctypes as C, ctypes.util, json
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent; OUT=HERE/'mechanism_figures'; DATA=OUT/'parameterization_force_probe.json'
INK,MUTED,LINE='#172B3A','#627381','#DCE3E8'; ORANGE,BLUE,GREEN,RED,PURPLE='#D88A2D','#357DB2','#23856D','#B94D58','#8B71AB'
class S:
 def __init__(self,title,subtitle,h=690):
  self.h=h;self.a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="{h}" viewBox="0 0 2000 {h}" role="img">',f'<title>{escape(title)}</title><desc>{escape(subtitle)}</desc>','<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10z" fill="#627381"/></marker></defs>','<rect width="2000" height="100%" fill="white"/>'];self.t(45,45,title,29,INK,700);self.t(45,75,subtitle,17,MUTED)
 def t(self,x,y,v,sz=19,c=INK,w=400,a='start'):self.a.append(f'<text x="{x}" y="{y}" font-family="DejaVu Sans, sans-serif" font-size="{sz}" font-weight="{w}" text-anchor="{a}" fill="{c}">{escape(str(v))}</text>')
 def rect(self,x,y,w,h,fill='white',stroke='none',r=8):self.a.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')
 def line(self,x1,y1,x2,y2,c=LINE,w=1.5,dash=False,arrow=False):self.a.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c}" stroke-width="{w}"'+(' stroke-dasharray="6 5"' if dash else '')+(' marker-end="url(#arrow)"' if arrow else '')+'/>')
 def path(self,pts,c=BLUE,w=2,dash=False):self.a.append('<path d="M '+' L '.join(f'{x:.2f} {y:.2f}' for x,y in pts)+f'" fill="none" stroke="{c}" stroke-width="{w}"'+(' stroke-dasharray="6 5"' if dash else '')+'/>')
 def circ(self,x,y,r,c):self.a.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"/>')
 def footer(self,lines):
  y=self.h-45-23*(len(lines)-1);self.line(45,y-20,1955,y-20)
  for i,v in enumerate(lines):self.t(45,y+23*i,v,16,MUTED)
 def save(self,name):
  p=OUT/(name+'.svg');v='\n'.join(self.a+['</svg>'])+'\n';ET.fromstring(v);p.write_text(v);return p
def force(d):
 s=S('Parameterization changes both probability weights and KD force','Read-only checkpoint probe  /  VGG13 → MobileNetV1  /  linear KD seed 2102  /  128 fixed official-test images')
 s.t(45,124,'(a) Exact mechanism',22,INK,700);s.rect(45,148,485,273,'#F5F7F9')
 s.t(70,188,'zₖ = r aₖ cos θₖ + bₖ',29,INK,700);s.t(70,227,'pₖ = softmax(z/T)ₖ',21,INK);s.t(70,260,'∇ₕ LKD = T Σₖ (pₖ − qₖ) aₖuₖ',24,INK,700)
 s.rect(70,286,186,91,'#FFF5E9');s.rect(274,286,220,91,'#F3EEF8')
 s.t(163,319,'Scale / norms',19,ORANGE,700,'middle');s.t(163,350,'r, aₖ change pₖ',17,INK,400,'middle');s.t(384,319,'Bias',19,PURPLE,700,'middle');s.t(384,350,'bₖ changes pₖ',17,INK,400,'middle')
 s.t(70,403,'aₖ also rescales each gradient basis vector uₖ.',17,MUTED)
 s.t(620,124,'(b) Fixed directions → output deformation',22,INK,700);s.t(620,151,'Hold ĥ and ŵₖ fixed; continuously inject r·aₖ and bₖ.',17,MUTED)
 x0,y0,w,h=655,389,430,178
 s.line(x0,y0-h,x0,y0);s.line(x0,y0,x0+w,y0)
 for v in (0,.1,.2):
  y=y0-v/.25*h;s.line(x0-5,y,x0,y);s.t(x0-12,y+6,f'{v:.1f}',16,MUTED,400,'end')
 s.t(x0,y0-h-15,'mean non-target total variation',16,MUTED)
 bars=[('angular → scale',d['distribution_deformation']['angular_to_scale_tv'],ORANGE),('scale → raw bias',d['distribution_deformation']['scale_to_raw_bias_tv'],PURPLE),('angular → raw',d['distribution_deformation']['angular_to_raw_tv'],RED)]
 for i,(lab,v,c) in enumerate(bars):
  x=x0+95+i*140;bh=v/.25*h;s.rect(x-25,y0-bh,50,bh,c,r=3);s.t(x,y0-bh-11,f'{v:.3f}',17,c,700,'middle');s.t(x,y0+27,lab,14,MUTED,400,'middle')
 s.t(1250,124,'(c) Raw force differs in direction and norm',22,INK,700);s.t(1250,151,'Cosine to standardized-logit reference; mean over 128 images.',17,MUTED)
 x0,y0,w=1285,389,360;s.line(x0,y0-178,x0,y0);s.line(x0,y0,x0+w,y0)
 for v in (0,.25,.5,.75):
  y=y0-v/.8*178;s.line(x0-5,y,x0,y);s.t(x0-12,y+6,f'{v:.2f}',16,MUTED,400,'end')
 for i,key in enumerate(['angular','scale_only','bias_only','raw']):
  v=d['gradient_statistics'][key]['mean_cosine_to_standardized_reference'];x=x0+55+i*87;bh=v/.8*178;c={'angular':BLUE,'scale_only':ORANGE,'bias_only':PURPLE,'raw':RED}[key]
  s.rect(x-19,y0-bh,38,bh,c,r=2);s.t(x,y0-bh-10,f'{v:.2f}',16,c,700,'middle');s.t(x,y0+28,{'angular':'angular','scale_only':'scale','bias_only':'bias','raw':'raw'}[key],14,MUTED,400,'middle')
 s.t(1285,444,'Raw force norm / reference = %.1f×'%d['gradient_statistics']['raw']['mean_norm_ratio_to_reference'],17,RED,700)
 s.footer(['Mathematical claim: scale/norms and bias enter pₖ; class norms also enter the gradient basis. No claim that they rotate θₖ.', 'Measured: this checkpoint probe shows substantial scale-driven conditional-distribution deformation; its bias contribution is small here.', 'The standardized-logit gradient is a diagnostic reference, not semantic ground truth or proof of globally conflicting optimization.'])
 return s.save('mechanism_parameterization_force')
def pull(d):
 s=S('Raw KD can pull a student along a parameterization-sensitive force','Local virtual-update view  /  same fixed sample, checkpoint, teacher target, and student feature state  /  arrows are −∇ₕL')
 s.t(45,124,'(a) Measured-gradient construction',21,INK,700)
 s.rect(45,149,570,168,'#F5F7F9');s.t(70,190,'e₁ = standardized-logit reference gradient',21,BLUE,700);s.t(70,226,'e₂ = raw-KL residual orthogonal to e₁',21,RED,700);s.t(70,265,'Every arrow starts from the same student feature hₛ.',18,INK);s.t(70,292,'Its direction is an SGD force −∇ₕL, projected onto [e₁,e₂].',18,INK)
 s.t(720,124,'(b) Counterfactual forces at fixed ĥ, ŵ, and teacher target',20,INK,700);s.t(720,151,'Only the score parameterization used by the student loss changes.',17,MUTED)
 ox,oy=1120,400;s.line(720,oy,1510,oy,LINE,1.5);s.line(ox,210,ox,555,LINE,1.5);s.t(1505,429,'reference-aligned force',17,MUTED,400,'end');s.t(1105,205,'parameterization residual',17,MUTED,400,'end')
 vals=d['virtual_update_plane']; items=[('angular','angular scores',BLUE),('scale_only','+ scale / norms',ORANGE),('bias_only','+ bias only',PURPLE),('raw','raw logits',RED)]
 # normalize to make local directional differences legible; labels retain exact cosine.
 maxv=max(max(abs(q['reference_axis']),abs(q['artifact_axis'])) for q in vals.values())
 offsets={'angular':(20,37),'scale_only':(20,-29),'bias_only':(20,4),'raw':(20,-4)}
 for key,lab,c in items:
  q=vals[key];x=ox-q['reference_axis']/maxv*300;y=oy+q['artifact_axis']/maxv*145;s.line(ox,oy,x,y,c,3,arrow=True);s.circ(x,y,5,c);dx,dy=offsets[key];s.t(x+dx,y+dy,lab,18,c,700);s.t(x+dx,y+dy+22,'cos=%.2f'%q['cosine_to_reference'],15,MUTED)
 s.line(ox,oy,ox+310,oy,BLUE,2,dash=True);s.t(1280,382,'idealized reference axis',16,BLUE,400)
 s.rect(720,570,700,48,'#FFF6F6');s.t(740,592,'Measured local result: raw logits are ≈14.7× stronger',17,RED,600);s.t(740,612,'than the standardized reference on average.',17,RED,600)
 s.footer(['This is not a full-training trajectory: it is a one-state virtual-gradient diagnostic with no parameter update committed.', 'The selected sample is the median raw/reference cosine on the fixed probe—not chosen by accuracy or visual effect.', 'A future multi-checkpoint trajectory figure must separately show whether ACD changes these forces over training.'])
 return s.save('mechanism_virtual_update_plane')
def export(svg):
 r=C.CDLL(ctypes.util.find_library('rsvg-2'));c=C.CDLL(ctypes.util.find_library('cairo'));g=C.CDLL(ctypes.util.find_library('gobject-2.0'));P=C.c_void_p;I=C.c_int;D=C.c_double
 def f(lib,n,args,ret):v=getattr(lib,n);v.argtypes=args;v.restype=ret;return v
 load=f(r,'rsvg_handle_new_from_file',[C.c_char_p,C.POINTER(P)],P);render=f(r,'rsvg_handle_render_cairo',[P,P],I);png=f(c,'cairo_image_surface_create',[I,I,I],P);pdf=f(c,'cairo_pdf_surface_create',[C.c_char_p,D,D],P);ctx=f(c,'cairo_create',[P],P);scale=f(c,'cairo_scale',[P,D,D],None);write=f(c,'cairo_surface_write_to_png',[P,C.c_char_p],I);destroy=f(c,'cairo_destroy',[P],None);finish=f(c,'cairo_surface_finish',[P],None);free=f(c,'cairo_surface_destroy',[P],None);status=f(c,'cairo_surface_status',[P],I);unref=f(g,'g_object_unref',[P],None)
 err=P();h=load(str(svg).encode(),C.byref(err));height=int(ET.parse(svg).getroot().attrib['height'])
 for ext,k in [('png',2.),('pdf',.315)]:
  width=int(ET.parse(svg).getroot().attrib['width']);out=svg.with_suffix('.'+ext);surf=png(0,width*2,height*2) if ext=='png' else pdf(str(out).encode(),width*k,height*k);cr=ctx(surf);scale(cr,k,k);assert render(h,cr);
  if ext=='png':assert not write(surf,str(out).encode())
  destroy(cr);finish(surf);assert not status(surf);free(surf)
 unref(h)
if __name__=='__main__':
 d=json.loads(DATA.read_text());assert d['status']=='completed'
 for fn in (force,pull):
  p=fn(d);export(p);print(p)
