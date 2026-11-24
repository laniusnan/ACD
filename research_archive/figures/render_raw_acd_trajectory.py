#!/usr/bin/env python3
"""Render one editable phase-plane trajectory for matched raw KD and ACD."""
import ctypes as C, ctypes.util, json
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent; OUT=HERE/'mechanism_figures';D=json.loads((OUT/'raw_acd_checkpoint_trajectory_probe.json').read_text())
W,H=1040,760;L,T,PW,PH=145,55,820,575;INK,MUTED,GRID='#172B3A','#52616B','#DCE3E8'; COLORS={'raw_kd':'#B94D58','acd':'#23856D'}
def main():
 rows=[r for v in D['results'].values() for r in v['rows']]; xmax=max(r['conditional_tv_standardized'] for r in rows)*1.12;ymax=max(r['raw_kl'] for r in rows)*1.12
 def xy(x,y):return L+x/xmax*PW,T+PH-y/ymax*PH
 a=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">','<title>Matched raw KD and ACD checkpoint trajectories</title><desc>Phase-plane trajectories compare standardized non-target relation discrepancy and raw-logit KL at four matched checkpoints.</desc>',f'<rect width="{W}" height="{H}" fill="white"/>']
 def line(x1,y1,x2,y2,c=GRID,w=1):a.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{c}" stroke-width="{w}"/>')
 def text(x,y,v,s=15,c=INK,anc='start',weight=400):a.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{s}" font-weight="{weight}" text-anchor="{anc}" fill="{c}">{escape(str(v))}</text>')
 line(L,T+PH,L+PW,T+PH,'#9EABB5',1.1);line(L,T,L,T+PH,'#9EABB5',1.1)
 for i in range(6):
  x=i*xmax/5;px,py=xy(x,0);line(px,T,px,T+PH);text(px,T+PH+23,f'{x:.3f}',12,MUTED,'middle')
  y=i*ymax/5;px,py=xy(0,y);line(L,py,L+PW,py);text(L-12,py+4,f'{y:.2f}',12,MUTED,'end')
 text(L+PW/2,T+PH+65,'Standardized non-target relation discrepancy',16,INK,'middle');a.append(f'<text x="35" y="{T+PH/2:.2f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="16" text-anchor="middle" fill="{INK}" transform="rotate(-90 35 {T+PH/2:.2f})">Raw-logit KL discrepancy</text>')
 for key,value in D['results'].items():
  c=COLORS[key];pts=[xy(r['conditional_tv_standardized'],r['raw_kl']) for r in value['rows']];a.append('<path d="M '+' L '.join(f'{x:.2f} {y:.2f}' for x,y in pts)+f'" fill="none" stroke="{c}" stroke-width="3"/>')
  for (x,y),r in zip(pts,value['rows']):a.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{c}" stroke="white" stroke-width="1.5"/>');text(x+10,y-9,f"e{r['epoch']}",12,c,weight=700)
  x,y=pts[-1];text(x+12,y+20,'Raw KD' if key=='raw_kd' else 'ACD',15,c,weight=700)
 for i,(key,label) in enumerate((('raw_kd','raw KD'),('acd','ACD'))):
  x=175+i*150;a.append(f'<circle cx="{x}" cy="28" r="6" fill="{COLORS[key]}"/>');text(x+11,33,label,14,MUTED)
 p=OUT/'raw_acd_checkpoint_trajectory.svg';p.write_text('\n'.join(a+['</svg>'])+'\n');ET.parse(p);export(p)
def export(svg):
 r=C.CDLL(ctypes.util.find_library('rsvg-2'));c=C.CDLL(ctypes.util.find_library('cairo'));g=C.CDLL(ctypes.util.find_library('gobject-2.0'));P,I,Dbl=C.c_void_p,C.c_int,C.c_double
 def f(lib,n,args,ret):v=getattr(lib,n);v.argtypes=args;v.restype=ret;return v
 load=f(r,'rsvg_handle_new_from_file',[C.c_char_p,C.POINTER(P)],P);render=f(r,'rsvg_handle_render_cairo',[P,P],I);png=f(c,'cairo_image_surface_create',[I,I,I],P);pdf=f(c,'cairo_pdf_surface_create',[C.c_char_p,Dbl,Dbl],P);ctx=f(c,'cairo_create',[P],P);scl=f(c,'cairo_scale',[P,Dbl,Dbl],None);write=f(c,'cairo_surface_write_to_png',[P,C.c_char_p],I);destroy=f(c,'cairo_destroy',[P],None);finish=f(c,'cairo_surface_finish',[P],None);free=f(c,'cairo_surface_destroy',[P],None);status=f(c,'cairo_surface_status',[P],I);unref=f(g,'g_object_unref',[P],None)
 err=P();h=load(str(svg).encode(),C.byref(err))
 for ext,k in (('png',2.),('pdf',.42)):
  o=svg.with_suffix('.'+ext);surf=png(0,W*2,H*2) if ext=='png' else pdf(str(o).encode(),W*k,H*k);cr=ctx(surf);scl(cr,k,k);assert render(h,cr)
  if ext=='png':assert not write(surf,str(o).encode())
  destroy(cr);finish(surf);assert not status(surf);free(surf)
 unref(h)
if __name__=='__main__':main()
