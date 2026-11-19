"""
Particle trails with a geometry shader.

Each particle is uploaded as a single point (a seed: phase, radius, speed, hue).
The vertex shader evaluates the particle's head position on a Lissajous curve.
The geometry shader expands every point into a line strip that re-evaluates the
same curve backwards in time, fading the alpha towards the tail.

Runs headless and writes ``particle_trails.png`` next to this file.
"""

from pathlib import Path

import moderngl
import numpy as np
from PIL import Image

WIDTH, HEIGHT = 1024, 1024
PARTICLES = 24
TRAIL_SEGMENTS = 64  # vertices emitted per point by the geometry shader

ctx = moderngl.create_standalone_context()

fbo = ctx.framebuffer([ctx.renderbuffer((WIDTH, HEIGHT), components=4, samples=4)])
output = ctx.framebuffer([ctx.renderbuffer((WIDTH, HEIGHT), components=4)])

prog = ctx.program(
    vertex_shader="""
        #version 330 core

        uniform float u_time;

        in vec4 in_seed;  // phase_x, phase_y, radius, speed
        in float in_hue;

        out float v_hue;
        out vec4 v_seed;

        vec2 orbit(vec4 seed, float t) {
            return seed.z * vec2(
                sin(2.0 * seed.w * t + seed.x),
                sin(3.0 * seed.w * t + seed.y)
            );
        }

        void main() {
            v_seed = in_seed;
            v_hue = in_hue;
            gl_Position = vec4(orbit(in_seed, u_time), 0.0, 1.0);
        }
    """,
    geometry_shader="""
        #version 330 core

        layout(points) in;
        layout(line_strip, max_vertices=%(trail_segments)d) out;

        uniform float u_time;

        in float v_hue[];
        in vec4 v_seed[];

        out vec4 g_color;

        vec3 palette(float hue) {
            return 0.5 + 0.5 * cos(6.2831853 * (hue + vec3(0.0, 0.33, 0.67)));
        }

        vec2 orbit(vec4 seed, float t) {
            return seed.z * vec2(
                sin(2.0 * seed.w * t + seed.x),
                sin(3.0 * seed.w * t + seed.y)
            );
        }

        void main() {
            vec3 color = palette(v_hue[0]);
            for (int i = 0; i < %(trail_segments)d; ++i) {
                float k = float(i) / float(%(trail_segments)d - 1);
                float t = u_time - k * 0.9;  // trail reaches back in time
                g_color = vec4(color * 1.8, pow(1.0 - k, 1.2));
                gl_Position = vec4(orbit(v_seed[0], t), 0.0, 1.0);
                EmitVertex();
            }
            EndPrimitive();
        }
    """ % {"trail_segments": TRAIL_SEGMENTS},
    fragment_shader="""
        #version 330 core

        in vec4 g_color;
        out vec4 f_color;

        void main() {
            f_color = g_color;
        }
    """,
)

assert prog.geometry_input == moderngl.POINTS
assert prog.geometry_output == moderngl.LINES
assert prog.geometry_vertices == TRAIL_SEGMENTS

rng = np.random.default_rng(42)
seeds = np.column_stack(
    [
        rng.uniform(0.0, 2.0 * np.pi, PARTICLES),  # phase_x
        rng.uniform(0.0, 2.0 * np.pi, PARTICLES),  # phase_y
        rng.uniform(0.45, 0.9, PARTICLES),  # radius
        rng.uniform(0.6, 1.4, PARTICLES),  # speed
        rng.uniform(0.0, 1.0, PARTICLES),  # hue
    ]
).astype("f4")

vbo = ctx.buffer(seeds.tobytes())
vao = ctx.vertex_array(prog, [(vbo, "4f 1f", "in_seed", "in_hue")], mode=moderngl.POINTS)

# Glowing sprite for each particle head, reusing the same seed buffer.
head_prog = ctx.program(
    vertex_shader="""
        #version 330 core

        uniform float u_time;

        in vec4 in_seed;
        in float in_hue;

        out vec3 v_color;

        void main() {
            vec2 pos = in_seed.z * vec2(
                sin(2.0 * in_seed.w * u_time + in_seed.x),
                sin(3.0 * in_seed.w * u_time + in_seed.y)
            );
            v_color = 0.5 + 0.5 * cos(6.2831853 * (in_hue + vec3(0.0, 0.33, 0.67)));
            gl_Position = vec4(pos, 0.0, 1.0);
            gl_PointSize = 24.0;
        }
    """,
    fragment_shader="""
        #version 330 core

        in vec3 v_color;
        out vec4 f_color;

        void main() {
            float d = length(gl_PointCoord - 0.5) * 2.0;
            f_color = vec4(v_color, pow(max(1.0 - d, 0.0), 2.0));
        }
    """,
)
head_vao = ctx.vertex_array(head_prog, [(vbo, "4f 1f", "in_seed", "in_hue")], mode=moderngl.POINTS)

prog["u_time"] = 3.0
head_prog["u_time"] = 3.0

fbo.use()
ctx.clear(0.02, 0.02, 0.05)
ctx.enable(moderngl.BLEND)
ctx.blend_func = moderngl.PREMULTIPLIED_ALPHA  # (SRC_ALPHA, ONE): additive glow
vao.render()
ctx.enable(moderngl.PROGRAM_POINT_SIZE)
head_vao.render()

ctx.copy_framebuffer(output, fbo)

image = Image.frombuffer("RGBA", output.size, output.read(components=4))
image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
path = Path(__file__).with_name("particle_trails.png")
image.save(path)
print(f"saved {path}")
