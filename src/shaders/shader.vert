// shader.vert
#version 450

layout(location=1) out vec4 v_color;

const vec2 positions[3] = vec2[3](
vec2(0.0, -0.5),
vec2(-0.5, 0.5),
vec2(0.5, 0.5)
);

void main() {
    v_color = vec4(0.3, 0.2, 0.1, 1.0);
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
}
