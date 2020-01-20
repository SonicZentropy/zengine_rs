// shader.frag
#version 450

//this means that the value of f_color will be saved to whatever buffer is at location zero in our application.
//In most cases, location=0 is the current texture from the swapchain aka. the screen.
layout(location=0) out vec4 f_color;

// shader.frag
layout(location=1) in vec3 v_color;

void main() {
    f_color = vec4(v_color, 1.0);
}
