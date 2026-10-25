"""
PDB (Point-Based Deformation Blending for Facial Animation Retargeting) method video,
3Blue1Brown style, no equations. Run in the Manim venv (_tmp/manim_env) after
render/method_video_assets.py has written vis_CBD/method_video/assets/:

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 _tmp/manim_env/bin/manim \\
        -r 1920,1080 --fps 30 --media_dir vis_CBD/method_video/manim \\
        -o pdb_method.mp4 render/method_video.py PDBMethod

(LD_PRELOAD: the conda python's libffi clashes with the system GLib used by Pango.)
Scenes: network (overview.png layout) / points v / weights w of 6 points v / blend
(no title / end cards; it follows the render_weight_vis.py overview in the supplementary
video). w and the points v use the render_weight_vis.py nipy_spectral colors. Target:
mf_ROM ID_012 in every scene.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from manim import (
    Scene, Text, ImageMobject, Dot, Arrow, Line, Circle, Polygon, RoundedRectangle, VGroup, Group,
    ValueTracker, FadeIn, FadeOut, GrowArrow, Create, DrawBorderThenFill, LaggedStart, Indicate,
    UP, DOWN, LEFT, RIGHT, BLACK, WHITE, GREY, linear,
)

ASSETS = Path(__file__).resolve().parent.parent / "vis_CBD" / "method_video" / "assets"
YELLOW, GREEN, ORANGE = "#FFD54A", "#8BC34A", "#FF9F43"
FONT = "Times New Roman"
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def label(text, size=36, color=WHITE):
    return Text(text, font=FONT, font_size=size, color=color)


def img(path, height):
    return ImageMobject(str(path)).set_height(height)


def rgb_hex(c):
    return "#%02x%02x%02x" % tuple(int(255 * x) for x in c)


def cloud_pos(xy, center, half):
    """normalized [-1, 1] point coords -> scene coords in a box of half-size `half`"""
    xy = np.atleast_2d(xy)
    return np.stack([center[0] + xy[:, 0] * half, center[1] + xy[:, 1] * half, np.zeros(len(xy))], 1)


def encoder(name, color, center):
    """trapezoid block (wide input side -> narrow output side) with its name"""
    poly = Polygon([-0.55, 0.85, 0], [0.55, 0.45, 0], [0.55, -0.45, 0], [-0.55, -0.85, 0],
                   color=color, fill_color=color, fill_opacity=0.85, stroke_width=0).move_to(center)
    return VGroup(poly, label(name, 44, BLACK).move_to(center))


class PDBMethod(Scene):
    def construct(self):
        self.camera.background_color = BLACK
        P = np.load(ASSETS / "points.npz")
        picks, used = P["picks"], P["used"]
        pick_rows = [int(np.where(used == p)[0][0]) for p in picks]
        pick_colors = [rgb_hex(c) for c in P["pick_colors"]]
        point_colors = [rgb_hex(c) for c in P["colors"]]
        n, mid = len(P["frames"]), int(P["still"])
        src_seq = [np.array(Image.open(ASSETS / "face" / "source_seq" / f"{k:04d}.png").convert("RGBA")) for k in range(n)]
        nrm_seq = [np.array(Image.open(ASSETS / "face" / "source_normal_seq" / f"{k:04d}.png").convert("RGBA")) for k in range(n)]
        tgt_seq = [np.array(Image.open(ASSETS / "face" / "target_seq" / f"{k:04d}.png").convert("RGBA")) for k in range(n)]

        # 3. network (overview.png layout): phi -> Reshape -> v; w x v -> output
        yt, yb = 1.9, -1.9
        x_in = img(ASSETS / "face" / "target_normal.png", 2.4).move_to([-5.4, yt, 0])
        l_in = label("Target neutral", 22).next_to(x_in, DOWN, buff=0.05)
        psi = encoder("ψ", GREEN, [-2.9, yt, 0])
        w_img = img(ASSETS / "face" / "target_w.png", 2.4).move_to([-0.4, yt, 0])
        l_w = label("w", 30).next_to(w_img, DOWN, buff=0.05)
        s_n = img(ASSETS / "face" / "source_normal.png", 2.1).move_to([-5.7, yb, 0])
        s_e = img(ASSETS / "face" / "source_expression_normal.png", 2.1).move_to([-4.4, yb, 0])
        l_s = label("Source neutral, expression", 22).next_to(Group(s_n, s_e), DOWN, buff=0.05)
        phi = encoder("φ", YELLOW, [-2.9, yb, 0])
        reshape = VGroup(RoundedRectangle(width=1.7, height=0.6, corner_radius=0.12, color=GREY,
                                          fill_color=GREY, fill_opacity=0.35),
                         label("Reshape", 24)).move_to([0.15, yb, 0])
        v_box_c, v_half = np.array([3.2, yb, 0]), 0.95
        v_cloud = VGroup(*[Dot(p, radius=0.022, color=c)
                           for p, c in zip(cloud_pos(P["seq"][mid], v_box_c, v_half), point_colors)])
        l_v = label("v", 30, YELLOW).next_to(v_cloud, DOWN, buff=0.12)
        mult = VGroup(Circle(radius=0.3, color=WHITE, stroke_width=4),
                      label("×", 36).move_to([0, 0, 0])).move_to([3.2, yt, 0])
        l_mult = label("Matrix multiplication", 22).next_to(mult, UP, buff=0.15)
        out = img(ASSETS / "face" / "target_output.png", 2.4).move_to([5.7, yt, 0])
        l_out = label("Target expression", 22).next_to(out, DOWN, buff=0.05)
        a1 = Arrow(x_in.get_right(), psi.get_left(), buff=0.1, color=WHITE)
        a2 = Arrow(psi.get_right(), w_img.get_left(), buff=0.1, color=WHITE)
        a3 = Arrow(w_img.get_right(), mult.get_left(), buff=0.1, color=WHITE)
        b1 = Arrow(s_e.get_right(), phi.get_left(), buff=0.1, color=WHITE)
        b2 = Arrow(phi.get_right(), reshape.get_left(), buff=0.1, color=WHITE)
        b3 = Arrow(reshape.get_right(), [v_box_c[0] - v_half, yb, 0], buff=0.1, color=WHITE)
        b4 = Arrow([v_box_c[0], yb + v_half, 0], mult.get_bottom(), buff=0.1, color=WHITE)
        a4 = Arrow(mult.get_right(), out.get_left(), buff=0.1, color=WHITE)
        self.play(FadeIn(x_in), FadeIn(l_in))
        self.play(GrowArrow(a1), DrawBorderThenFill(psi))
        self.play(GrowArrow(a2), FadeIn(w_img), FadeIn(l_w))
        self.wait(0.5)
        self.play(FadeIn(s_n), FadeIn(s_e), FadeIn(l_s))
        self.play(GrowArrow(b1), DrawBorderThenFill(phi))
        self.play(GrowArrow(b2), FadeIn(reshape))
        self.play(GrowArrow(b3), LaggedStart(*[FadeIn(d) for d in v_cloud], lag_ratio=0.003), FadeIn(l_v))
        self.remove(*v_cloud)   # the dots were added one by one: put the group itself in the
        self.add(v_cloud)       # scene, otherwise its updater below never runs
        self.wait(0.5)
        self.play(GrowArrow(b4), GrowArrow(a3), FadeIn(mult), FadeIn(l_mult))
        self.play(GrowArrow(a4), FadeIn(out), FadeIn(l_out))
        self.wait(1)
        # the source expression plays: v follows it, the output follows v
        tk3 = ValueTracker(mid)
        f3 = lambda: int(np.clip(round(tk3.get_value()), 0, n - 1))           # noqa: E731
        seq3 = [cloud_pos(P["seq"][k], v_box_c, v_half) for k in range(n)]
        s_e.add_updater(lambda m: setattr(m, "pixel_array", nrm_seq[f3()]))
        v_cloud.add_updater(lambda g: [d.move_to(seq3[f3()][i]) for i, d in enumerate(g)])
        out.add_updater(lambda m: setattr(m, "pixel_array", tgt_seq[f3()]))
        self.play(tk3.animate.set_value(0), run_time=mid / 30, rate_func=linear)
        self.play(tk3.animate.set_value(n - 1), run_time=n / 30 * 1.5, rate_func=linear)
        self.play(tk3.animate.set_value(mid), run_time=(n - 1 - mid) / 30, rate_func=linear)
        for m in (s_e, v_cloud, out):
            m.clear_updaters()
        self.wait(1)
        net = Group(x_in, l_in, psi, w_img, l_w, s_n, s_e, l_s, phi, reshape, v_cloud,
                    l_v, mult, l_mult, out, l_out, a1, a2, a3, a4, b1, b2, b3, b4)
        self.play(FadeOut(net))

        # 4. points v (no face overlay) --------------------------------------------------
        tk = ValueTracker(0)

        def frame():
            return int(np.clip(round(tk.get_value()), 0, n - 1))

        # zoom into the dense center (90th percentile radius fills the box) so the motion
        # is visible; points that would leave the box are hidden
        c4, h4 = np.array([-2.6, -0.3, 0]), 2.9
        zoom = 1.0 / np.percentile(np.linalg.norm(P["seq"][0], axis=1), 90)
        seq4 = [cloud_pos(P["seq"][k] * zoom, c4, h4) for k in range(n)]
        inside = (np.abs(np.stack([P["seq"][k] * zoom for k in range(n)])) <= 1.0).all(axis=(0, 2))
        dots4 = VGroup(*[Dot(p, radius=0.035, color=c, fill_opacity=1.0 if ok else 0.0)
                         for p, ok, c in zip(seq4[0], inside, point_colors)])
        src4 = img(ASSETS / "face" / "source_seq" / "0000.png", 5.0).move_to([3.7, -0.3, 0])
        head4 = label("Points v", 40, YELLOW).to_edge(UP)
        sub4 = label("move with the source expression", 26, GREY).next_to(head4, DOWN, buff=0.12)
        self.play(FadeIn(head4), FadeIn(sub4), FadeIn(src4),
                  LaggedStart(*[FadeIn(d, scale=0.5) for d in dots4], lag_ratio=0.004, run_time=1.5))
        self.remove(*dots4)     # same as v_cloud: the group must be in the scene for its updater
        self.add(dots4)
        src4.add_updater(lambda m: setattr(m, "pixel_array", src_seq[frame()]))
        dots4.add_updater(lambda g: [d.move_to(seq4[frame()][i]) for i, d in enumerate(g)])
        self.play(tk.animate.set_value(n - 1), run_time=n / 30 * 1.5, rate_func=linear)
        self.play(tk.animate.set_value(0), run_time=n / 30 * 1.5, rate_func=linear)
        src4.clear_updaters()
        dots4.clear_updaters()
        self.play(FadeOut(Group(src4, sub4, head4)))

        # 5. weights w of 6 points v (brow, cheek, jaw / forehead, nose, eye) ------------------
        head5 = label("Weights w", 40).to_edge(UP)
        sub5 = label("which part of the target each point moves", 26, GREY).next_to(head5, DOWN, buff=0.12)
        c5, h5 = np.array([-4.5, -0.7, 0]), 2.0            # smaller point cloud on the left
        pos5 = cloud_pos(P["seq"][0] * zoom, c5, h5)
        self.play(FadeIn(head5), FadeIn(sub5),
                  *[d.animate.move_to(p).scale(0.7).set_opacity(0.3 if ok else 0.0)
                    for d, p, ok in zip(dots4, pos5, inside)])
        grid = [[x, y] for y in (0.55, -2.35) for x in (0.4, 3.1, 5.8)]
        shown = []
        for k, (row, col, pid) in enumerate(zip(pick_rows, pick_colors, picks)):
            d = dots4[row]
            big = Dot(d.get_center(), radius=0.07, color=col)
            name = label(f"v{int(pid)}".translate(SUB), 22, col).next_to(big, UR, buff=0.03)
            face = img(ASSETS / "face" / f"weight_{k}.png", 2.4).move_to([grid[k][0], grid[k][1], 0])
            fname = label(f"v{int(pid)}".translate(SUB), 24, col).next_to(face, DOWN, buff=0.0)
            self.play(FadeIn(big, scale=2), FadeIn(name), run_time=0.5)
            self.play(Indicate(big, color=col, scale_factor=1.6), FadeIn(face), FadeIn(fname), run_time=0.8)
            shown += [big, name, face, fname]
        self.wait(2)
        self.play(FadeOut(Group(dots4, head5, sub5, *shown)))

        # 6. blend ------------------------------------------------------------------------------
        tk.set_value(0)
        src6 = img(ASSETS / "face" / "source_seq" / "0000.png", 4.4).move_to([-4.6, -0.4, 0])
        tgt6 = img(ASSETS / "face" / "target_seq" / "0000.png", 4.4).move_to([4.6, -0.4, 0])
        c6, h6 = np.array([0.0, -0.4, 0]), 1.5
        seq6 = [cloud_pos(P["seq"][k], c6, h6) for k in range(n)]
        dots6 = VGroup(*[Dot(p, radius=0.022, color=c) for p, c in zip(seq6[0], point_colors)])
        ar1 = Arrow([-2.3, -0.4, 0], [-1.6, -0.4, 0], color=WHITE, buff=0)
        ar2 = Arrow([1.6, -0.4, 0], [2.3, -0.4, 0], color=WHITE, buff=0)
        lw = label("× target weights w", 24, GREY).next_to(tgt6, UP, buff=0.1)
        ls6 = label("Source", 28).next_to(src6, DOWN, buff=0.05)
        lv6 = label("Points v", 28, YELLOW).next_to(dots6, DOWN, buff=0.35)
        lt6 = label("Target", 28).next_to(tgt6, DOWN, buff=0.05)
        head6 = label("Blend", 40).to_edge(UP)
        self.play(FadeIn(Group(src6, tgt6, dots6, ar1, ar2, lw, ls6, lv6, lt6, head6)))
        src6.add_updater(lambda m: setattr(m, "pixel_array", src_seq[frame()]))
        tgt6.add_updater(lambda m: setattr(m, "pixel_array", tgt_seq[frame()]))
        dots6.add_updater(lambda g: [d.move_to(seq6[frame()][i]) for i, d in enumerate(g)])
        for _ in range(2):
            self.play(tk.animate.set_value(n - 1), run_time=n / 30 * 1.5, rate_func=linear)
            self.play(tk.animate.set_value(0), run_time=n / 30 * 1.5, rate_func=linear)
        for m in (src6, tgt6, dots6):
            m.clear_updaters()
        self.play(FadeOut(Group(src6, tgt6, dots6, ar1, ar2, lw, ls6, lv6, lt6, head6)))



UR = UP + RIGHT
