#![allow(non_snake_case)]

extern crate log;

#[allow(unused_imports)]
use log::{error, info, warn};
use dotenv;

mod util;
#[allow(unused_imports)]
use util::string_utils::*;
#[allow(unused_imports)]
use util::file_utils::*;

use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{WindowBuilder, Window},
};

fn main() {

    init();
    info!("Initialized");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    let mut state = State::new(&window);

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => if state.input(event) {
                *control_flow = ControlFlow::Wait;
            } else {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::KeyboardInput {
                        input,
                        ..
                    } => {
                        match input {
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            } => *control_flow = ControlFlow::Exit,
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Space),
                                ..
                            } => {
                                info!("Toggling");
                                state.use_triangle_pipeline = !state.use_triangle_pipeline;
                                *control_flow = ControlFlow::Wait;
                            },
                            _ => *control_flow = ControlFlow::Wait,
                        }
                    }
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                        *control_flow = ControlFlow::Wait;
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                        *control_flow = ControlFlow::Wait;
                    }
                    WindowEvent::CursorMoved {position, ..} => {
                        let window_size = window.inner_size();

                        let pos_x = position.x as f64;
                        let pos_y = position.y as f64;
                        let width = window_size.width as f64;
                        let height = window_size.height as f64;

                        let r: f64 = pos_x / width;
                        let g: f64 = pos_y / height;
                        let b: f64 = ((pos_x + pos_y)/2.0)  / ((width + height) /2.0) ;

                        state.clear_color = wgpu::Color {r, g, b, a: 1.0};
                    }
                    _ => *control_flow = ControlFlow::Wait,
                }
            }
            Event::MainEventsCleared => {
                state.update();
                state.render();
                *control_flow = ControlFlow::Wait;
            }
            _ => *control_flow = ControlFlow::Wait,
        }
    });
}


struct State {
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sc_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
    clear_color: wgpu::Color,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    diffuse_texture: wgpu::Texture,
    diffuse_texture_view: wgpu::TextureView,
    diffuse_sampler: wgpu::Sampler,
    diffuse_bind_group: wgpu::BindGroup,
    size: winit::dpi::PhysicalSize<u32>,
    use_triangle_pipeline: bool,
    num_indices: u32,
    camera: Camera,
    camera_controller: CameraController,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl State {
    fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let surface = wgpu::Surface::create(window);

        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
            ..Default::default()
        }).unwrap();

        let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: Default::default(),
        });

        let sc_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Vsync,
        };
        let swap_chain = device.create_swap_chain(&surface, &sc_desc);

        let clear_color = wgpu::Color {r: 0.1, g: 0.3, b: 0.5, a: 1.0};

        //load texture
        let diffuse_bytes = include_bytes!("../assets/textures/happy-tree.png");
        let diffuse_image = image::load_from_memory(diffuse_bytes).unwrap();
        let diffuse_rgba = diffuse_image.as_rgba8().unwrap();
        use image::GenericImageView;
        let dimensions = diffuse_image.dimensions();

        let size3d = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth: 1,
        };

        let diffuse_texture = device.create_texture(&wgpu::TextureDescriptor {
           size: size3d,
            array_layer_count: 1,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            //sampled means we want to use this texture in shaders
            // copy_dst means we want to copy data to the texture
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        let diffuse_buffer = device
            .create_buffer_mapped(diffuse_rgba.len(), wgpu::BufferUsage::COPY_SRC)
            .fill_from_slice(&diffuse_rgba);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            todo: 0,
        });

        encoder.copy_buffer_to_texture(
            wgpu::BufferCopyView {
                buffer: &diffuse_buffer,
                offset: 0,
                row_pitch: 4 * dimensions.0,
                image_height: dimensions.1,
            },
            wgpu::TextureCopyView {
                texture: &diffuse_texture,
                mip_level: 0,
                array_layer: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            size3d,
        );

        queue.submit(&[encoder.finish()]);

        let diffuse_texture_view = diffuse_texture.create_default_view();

        let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            lod_min_clamp: -100.0,
            lod_max_clamp: 100.0,
            compare_function: wgpu::CompareFunction::Always,
        });

        //bindgroup describes resources and how they're accessed by a shader
        let texture_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                },
                wgpu::BindGroupLayoutBinding {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler,
                },
            ],
        });

        let diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &texture_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
                }
            ],
        });

        let camera = Camera {
            // position the camera one unit up and 2 units back
            eye: (0.0, 1.0, -2.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect: sc_desc.width as f32 / sc_desc.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let camera_controller = CameraController::new(0.2);

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniform_buffer = device
            // The COPY_DST part will be important later
            .create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST)
            .fill_from_slice(&[uniforms]);

        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            bindings: &[
                wgpu::BindGroupLayoutBinding {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX, // 1.
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false, // 2.
                    },
                }
            ]
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer {
                        buffer: &uniform_buffer,
                        // FYI: you can share a single buffer between bindings.
                        range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress,
                    }
                }
            ],
        });

        let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&texture_bind_group_layout, &uniform_bind_group_layout],
        });

        //NORMAL PIPELINE
        let vs_src = include_str!("shaders/shader.vert");
        let fs_src = include_str!("shaders/shader.frag");

        let vs_spirv = glsl_to_spirv::compile(vs_src, glsl_to_spirv::ShaderType::Vertex).unwrap();
        let fs_spirv = glsl_to_spirv::compile(fs_src, glsl_to_spirv::ShaderType::Fragment).unwrap();

        let vs_data = wgpu::read_spirv(vs_spirv).unwrap();
        let fs_data = wgpu::read_spirv(fs_spirv).unwrap();

        let vs_module = device.create_shader_module(&vs_data);
        let fs_module = device.create_shader_module(&fs_data);



        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: &render_pipeline_layout,
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
            }),
            color_states: &[
                wgpu::ColorStateDescriptor {
                    format: sc_desc.format,
                    color_blend: wgpu::BlendDescriptor::REPLACE,
                    alpha_blend: wgpu::BlendDescriptor::REPLACE,
                    write_mask: wgpu::ColorWrite::ALL,
                }
            ],
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            depth_stencil_state: None,
            index_format: wgpu::IndexFormat::Uint16,
            vertex_buffers: &[
                Vertex::desc(),
            ],
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });


        let vertex_buffer = device
            .create_buffer_mapped(VERTICES.len(), wgpu::BufferUsage::VERTEX)
            .fill_from_slice(VERTICES);

        let index_buffer = device
            .create_buffer_mapped(INDICES.len(), wgpu::BufferUsage::INDEX)
            .fill_from_slice(INDICES);
        let num_indices = INDICES.len() as u32;



        Self {
            surface,
            adapter,
            device,
            queue,
            sc_desc,
            swap_chain,
            clear_color,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            size,
            diffuse_texture,
            diffuse_texture_view,
            diffuse_sampler,
            diffuse_bind_group,
            use_triangle_pipeline: false,
            num_indices,
            camera,
            camera_controller,
            uniform_buffer,
            uniform_bind_group,
            uniforms,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.uniforms.update_view_proj(&self.camera);

        // Copy operation's are performed on the gpu, so we'll need
        // a CommandEncoder for that
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            todo: 0,
        });

        let staging_buffer = self.device
                                 .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
                                 .fill_from_slice(&[self.uniforms]);

        encoder.copy_buffer_to_buffer(&staging_buffer, 0, &self.uniform_buffer, 0, std::mem::size_of::<Uniforms>() as wgpu::BufferAddress);

        // We need to remember to submit our CommandEncoder's output
        // otherwise we won't see any change.
        self.queue.submit(&[encoder.finish()]);
    }

    fn render(&mut self) {
        // This is actual swap chain texture we render to
        let frame = self.swap_chain.get_next_texture();

        //Command Encoder is just fancy name for vulkan command buffer we're going to submit to gpu through
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            todo: 0,
        });


            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view, //Draw to swapchain screen texture
                        resolve_target: None,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color: self.clear_color,
                    }
                ],
                depth_stencil_attachment: None,
            });


            render_pass.set_pipeline(&self.render_pipeline);

            render_pass.set_vertex_buffers(0, &[(&self.vertex_buffer, 0)]);
            render_pass.set_index_buffer(&self.index_buffer, 0);
            render_pass.set_bind_group(0, &self.diffuse_bind_group, &[]);
            render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        render_pass.set_vertex_buffers(0, &[(&self.vertex_buffer, 0)]);
        render_pass.set_index_buffer(&self.index_buffer,0);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);

        drop(render_pass);

        self.queue.submit(&[
            encoder.finish()
        ]);
    }
}

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        // 1.
        let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);
        // 2.
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        // 3. This is the ONLY place you have to convert vulkan to opengl coordinate system
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

#[repr(C)] // We need this for Rust to store our data correctly for the shaders
#[derive(Debug, Copy, Clone)] // This is so we can store this in a buffer
struct Uniforms {
    view_proj: cgmath::Matrix4<f32>,
}

impl Uniforms {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix();
    }
}


#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    position: [f32; 3],
    tex_coords: [f32; 2],
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        use std::mem;
        wgpu::VertexBufferDescriptor {
            stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3,
                },
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float2,
                },
            ]
        }
    }
}

const VERTICES: &[Vertex] = &[
    Vertex { position: [-0.0868241, -0.49240386, 0.0], tex_coords: [1.0 - 0.4131759, 1.0 - 0.00759614], }, // A
    Vertex { position: [-0.49513406, -0.06958647, 0.0], tex_coords: [1.0 - 0.0048659444, 1.0 - 0.43041354], }, // B
    Vertex { position: [-0.21918549, 0.44939706, 0.0], tex_coords: [1.0 - 0.28081453, 1.0 - 0.949397057], }, // C
    Vertex { position: [0.35966998, 0.3473291, 0.0], tex_coords: [1.0 - 0.85967, 1.0 - 0.84732911], }, // D
    Vertex { position: [0.44147372, -0.2347359, 0.0], tex_coords: [1.0 - 0.9414737, 1.0 - 0.2652641], }, // E
];

const INDICES: &[u16] = &[
    0, 1, 4,
    1, 2, 4,
    2, 3, 4,
];

struct CameraController {
    speed: f32,
    is_up_pressed: bool,
    is_down_pressed: bool,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_up_pressed: false,
            is_down_pressed: false,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                input: KeyboardInput {
                    state,
                    virtual_keycode: Some(keycode),
                    ..
                },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    VirtualKeyCode::Space => {
                        self.is_up_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::LShift => {
                        self.is_down_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::W | VirtualKeyCode::Up => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::A | VirtualKeyCode::Left => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::S | VirtualKeyCode::Down => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    VirtualKeyCode::D | VirtualKeyCode::Right => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = (camera.target - camera.eye).normalize();

        if self.is_forward_pressed {
            camera.eye += forward * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward * self.speed;
        }

        let right = forward.cross(camera.up);

        if self.is_right_pressed {
            camera.eye += right * self.speed;
        }
        if self.is_left_pressed {
            camera.eye -= right * self.speed;
        }
    }
}

#[cfg_attr(rustfmt, rustfmt_skip)]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
);


fn init() {
    dotenv::dotenv().ok();
    let log_filepath = dotenv::var("LOGGER_FILEPATH").unwrap();
    fern::Dispatch::new()
        // Perform allocation-free log formatting
        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}", chrono::Local::now().format("[%Y-%m-%d][%H:%M]"),
                record.target(),
                record.level(),
                message))
        })
        // Add blanket level filter -
        .level(log::LevelFilter::Info)
        // Output to stdout, files, and other Dispatch configurations
        .chain(std::io::stdout())
        .chain(fern::log_file(&log_filepath).unwrap())
        // Apply globally
        .apply()
        .unwrap();
    info!("Initialization Complete");
}



//////////////////////////TESTING

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adds_properly() {
        assert_eq!(2+2, 4);
    }

    #[test]
    fn adds_improperly() {
        assert_ne!(2 + 2, 5);
    }
}
