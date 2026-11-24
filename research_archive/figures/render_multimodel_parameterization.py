#!/usr/bin/env python3
"""Render two compact, editable SVG ACD mechanism figures without new deps."""
import ctypes as C
import ctypes.util
import json
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
OUT = HERE / "mechanism_figures"
DATA = json.loads((OUT / "multimodel_parameterization_probe.json").read_text())
COLORS = {"vgg13_mv1": "#2878B5", "res50_mv1": "#D88A2D", "res32x4_sv1": "#23856D", "wrn402_sv1": "#8B71AB", "res32x4_sv2": "#B94D58"}
INK, MUTED, GRID = "#172B3A", "#52616B", "#DCE3E8"
W, H = 1080, 760
PLOT = (125, 55, 890, 570)

class Svg:
    def __init__(self, title, description):
        self.a = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">', f'<title>{escape(title)}</title><desc>{escape(description)}</desc>', f'<rect width="{W}" height="{H}" fill="white"/>']
    def line(self, x1, y1, x2, y2, color=GRID, width=1): self.a.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{color}" stroke-width="{width}"/>')
    def circle(self, x, y, r, fill, opacity=1, stroke="none", sw=0): self.a.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>')
    def rect(self, x, y, w, h, fill, stroke="none", sw=0): self.a.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    def text(self, x, y, value, size=15, color=INK, anchor="start", weight=400, rotate=None):
        extra = f' transform="rotate({rotate} {x} {y})"' if rotate else ''
        self.a.append(f'<text x="{x:.2f}" y="{y:.2f}" font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}"{extra}>{escape(value)}</text>')
    def path(self, points, color, width=2): self.a.append('<path d="M ' + ' L '.join(f'{x:.2f} {y:.2f}' for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="{width}"/>')
    def save(self, stem):
        if stem == 'multimodel_deformation_force_association':
            # Keep labels and points; trim the unused outer canvas only.
            self.a[0] = '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="640" viewBox="0 60 1000 640" role="img">'
        path = OUT / f'{stem}.svg'; path.write_text('\n'.join(self.a + ['</svg>']) + '\n'); ET.parse(path); return path

def scale(points, xlim, ylim):
    left, top, width, height = PLOT
    return [(left + (x-xlim[0])/(xlim[1]-xlim[0])*width, top+height-(y-ylim[0])/(ylim[1]-ylim[0])*height) for x, y in points]

def axes(s, xlim, ylim, xlabel, ylabel, x_ticks=5, y_ticks=5):
    left, top, width, height = PLOT
    s.line(left, top+height, left+width, top+height, '#9EABB5', 1.1); s.line(left, top, left, top+height, '#9EABB5', 1.1)
    for i in range(x_ticks+1):
        val=xlim[0]+i*(xlim[1]-xlim[0])/x_ticks; x=left+i*width/x_ticks; s.line(x, top, x, top+height, GRID, .8); s.text(x, top+height+23, f'{val:.2f}', 12, MUTED, 'middle')
    for i in range(y_ticks+1):
        val=ylim[0]+i*(ylim[1]-ylim[0])/y_ticks; y=top+height-i*height/y_ticks; s.line(left, y, left+width, y, GRID, .8); s.text(left-12, y+4, f'{val:.2f}', 12, MUTED, 'end')
    s.text(left+width/2, top+height+64, xlabel, 16, INK, 'middle'); s.text(31, top+height/2, ylabel, 16, INK, 'middle', rotate=-90)

def quantile(values, q):
    data=sorted(values); pos=(len(data)-1)*q; lo=int(pos); hi=min(lo+1,len(data)-1); return data[lo]+(data[hi]-data[lo])*(pos-lo)

def association():
    # Central-window view: the dense sample cloud is the paper-facing object;
    # sparse extremes remain documented in the companion probe record.
    left,top,width,height=105,75,875,545; xlim=(.10,.45); ylim=(0,1.60)
    def xy(x,y): return left+(x-xlim[0])/(xlim[1]-xlim[0])*width,top+height-(y-ylim[0])/(ylim[1]-ylim[0])*height
    s=Svg('Parameterization deformation and KD-force deviation', 'Central-window view of the five-pair probe. IQR ellipses describe image-level central spread, not seed uncertainty.')
    s.line(left,top+height,left+width,top+height,'#9EABB5',1.1);s.line(left,top,left,top+height,'#9EABB5',1.1)
    for i in range(6):
        v=xlim[0]+i*(xlim[1]-xlim[0])/5;x=left+i*width/5;s.line(x,top,x,top+height,GRID,.8);s.text(x,top+height+23,f'{v:.2f}',12,MUTED,'middle')
        v=ylim[0]+i*(ylim[1]-ylim[0])/5;y=top+height-i*height/5;s.line(left,y,left+width,y,GRID,.8);s.text(left-12,y+4,f'{v:.2f}',12,MUTED,'end')
    s.text(left+width/2,top+height+64,'Non-target conditional-distribution deformation',15,INK,'middle');s.text(31,top+height/2,'KD-force directional deviation',15,INK,'middle',rotate=-90)
    # Tight conventional in-plot legend, placed in the sparse upper-right area.
    lx,ly=755,99;s.rect(lx-12,ly-23,207,166,'white','#DCE3E8',.8);s.text(lx,ly,'Heterogeneous pairs',13,INK,weight=700)
    for row,(pair,value) in enumerate(DATA['results'].items()):
        color=COLORS[pair];xs=value['distribution_tv_angular_to_raw'];ys=value['gradient_direction_deviation_angular_to_raw'];q1x,q3x=quantile(xs,.25),quantile(xs,.75);q1y,q3y=quantile(ys,.25),quantile(ys,.75);medx,medy=quantile(xs,.5),quantile(ys,.5)
        # Axis-aligned IQR ellipse: transparent descriptive central range.
        ex,ey=xy(medx,medy);rx=(q3x-q1x)/2/(xlim[1]-xlim[0])*width;ry=(q3y-q1y)/2/(ylim[1]-ylim[0])*height
        s.a.append(f'<ellipse cx="{ex:.2f}" cy="{ey:.2f}" rx="{rx:.2f}" ry="{ry:.2f}" fill="{color}" fill-opacity=".09" stroke="{color}" stroke-opacity=".58" stroke-width="1.2"/>')
        for x0,y0 in zip(xs,ys):
            if xlim[0] <= x0 <= xlim[1] and ylim[0] <= y0 <= ylim[1]:
                px,py=xy(x0,y0);s.circle(px,py,3,color,.20)
        mx,my=value['summary']['mean_tv_angular_to_raw'],value['summary']['mean_gradient_direction_deviation'];px,py=xy(mx,my);s.circle(px,py,7,color,1,'white',2)
        offsets={'vgg13_mv1':(10,18),'res50_mv1':(10,17),'res32x4_sv1':(-103,4),'wrn402_sv1':(10,-10),'res32x4_sv2':(-103,-14)}
        dx,dy=offsets[pair];s.text(px+dx,py+dy,f'μ ({mx:.3f}, {my:.3f})',11,color,weight=700)
        yy=ly+27+row*26;s.circle(lx+6,yy-5,5.5,color);s.text(lx+17,yy,value['label'],11,INK,weight=700)
    return s.save('multimodel_deformation_force_association')

def trajectories():
    # Direction-only fan plot: unit-normalize forces to make angular rotation
    # visible instead of allowing large norm changes to dominate the layout.
    import math
    s=Svg('Parameterization rotates local KD-force directions', 'Each row unit-normalizes the angular, scale/norm, and raw mean KD forces in its pair-local plane. Angles state raw-force rotation from the angular baseline.')
    s.text(80,62,'Pair',14,MUTED,weight=700);s.text(283,62,'Unit-norm KD-force direction',14,MUTED,weight=700)
    s.line(275,76,880,76,GRID,1)
    shapes={'angular':'circle','scale_norm':'square','raw':'diamond'}
    origin_x,length=315,65
    for row,(pair,value) in enumerate(DATA['results'].items()):
        color=COLORS[pair]; origin_y=180+row*105
        s.line(80,origin_y+55,950,origin_y+55,GRID,.8)
        s.text(80,origin_y+5,value['label'],16,color,weight=700)
        vectors={stage:value['trajectory'][stage] for stage in ('angular','scale_norm','raw')}
        def endpoint(vector):
            vx,vy=vector; norm=math.hypot(vx,vy); return origin_x+length*vx/norm,origin_y-length*vy/norm
        # Angular reference is exactly horizontal by construction.
        for stage,opacity,width in (('angular',.9,2),('scale_norm',.55,2.4),('raw',1,3.2)):
            tx,ty=endpoint(vectors[stage]); stroke=MUTED if stage=='angular' else color
            s.line(origin_x,origin_y,tx,ty,stroke,width)
            if stage=='angular':s.circle(tx,ty,6,MUTED,opacity,'white',1.2)
            elif stage=='scale_norm':s.rect(tx-5,ty-5,10,10,color,'white',1.2)
            else:s.a.append(f'<path d="M {tx:.2f} {ty-7:.2f} L {tx+7:.2f} {ty:.2f} L {tx:.2f} {ty+7:.2f} L {tx-7:.2f} {ty:.2f} Z" fill="{color}" stroke="white" stroke-width="1.2"/>')
        raw=vectors['raw']; angle=math.degrees(math.atan2(raw[1],raw[0])); tx,ty=endpoint(raw);s.text(tx+11,ty-5,f'Δθ {angle:.0f}°',13,color,weight=700)
        s.circle(origin_x,origin_y,3.5,INK)
    for i,(label,stage) in enumerate((('angular','angular'),('+ scale/norm','scale_norm'),('raw','raw'))):
        x=560+i*150;y=38
        if shapes[stage]=='circle':s.circle(x,y,5,MUTED)
        elif shapes[stage]=='square':s.rect(x-5,y-5,10,10,MUTED)
        else:s.a.append(f'<path d="M {x:.2f} {y-6:.2f} L {x+6:.2f} {y:.2f} L {x:.2f} {y+6:.2f} L {x-6:.2f} {y:.2f} Z" fill="{MUTED}"/>')
        s.text(x+11,y+5,label,13,MUTED)
    return s.save('multimodel_parameterization_force_trajectories')

def export(svg):
    root=ET.parse(svg).getroot();canvas_w=int(root.attrib['width']);canvas_h=int(root.attrib['height'])
    rsvg=C.CDLL(ctypes.util.find_library('rsvg-2'));cairo=C.CDLL(ctypes.util.find_library('cairo'));gobj=C.CDLL(ctypes.util.find_library('gobject-2.0'));P,I,D=C.c_void_p,C.c_int,C.c_double
    def bind(lib,name,args,ret):fn=getattr(lib,name);fn.argtypes=args;fn.restype=ret;return fn
    load=bind(rsvg,'rsvg_handle_new_from_file',[C.c_char_p,C.POINTER(P)],P);render=bind(rsvg,'rsvg_handle_render_cairo',[P,P],I);png=bind(cairo,'cairo_image_surface_create',[I,I,I],P);pdf=bind(cairo,'cairo_pdf_surface_create',[C.c_char_p,D,D],P);ctx=bind(cairo,'cairo_create',[P],P);scl=bind(cairo,'cairo_scale',[P,D,D],None);write=bind(cairo,'cairo_surface_write_to_png',[P,C.c_char_p],I);destroy=bind(cairo,'cairo_destroy',[P],None);finish=bind(cairo,'cairo_surface_finish',[P],None);free=bind(cairo,'cairo_surface_destroy',[P],None);status=bind(cairo,'cairo_surface_status',[P],I);unref=bind(gobj,'g_object_unref',[P],None)
    err=P();handle=load(str(svg).encode(),C.byref(err))
    for ext,factor in (('png',2.0),('pdf',.42)):
        out=svg.with_suffix('.'+ext);surface=png(0,canvas_w*2,canvas_h*2) if ext=='png' else pdf(str(out).encode(),canvas_w*factor,canvas_h*factor);context=ctx(surface);scl(context,factor,factor);assert render(handle,context)
        if ext=='png':assert not write(surface,str(out).encode())
        destroy(context);finish(surface);assert not status(surface);free(surface)
    unref(handle)

if __name__=='__main__':
    for fn in (association,trajectories):export(fn())
