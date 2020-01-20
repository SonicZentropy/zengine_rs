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
    size: winit::dpi::PhysicalSize<u32>,
}

impl State {
    fn new(window: &Window) -> Self {
        let size = window.inner_size();

        let surface = wgpu::Surface::create(window);

        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions {
            ..Default::default()
        }).unwrap();

        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
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

        Self {
            surface,
            adapter,
            device,
            queue,
            sc_desc,
            swap_chain,
            clear_color,
            size,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.sc_desc.width = new_size.width;
        self.sc_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.sc_desc);
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        false
    }

    fn update(&mut self) {

    }

    fn render(&mut self) {
        // This is actual swap chain texture we render to
        let frame = self.swap_chain.get_next_texture();

        //Command Encoder is just fancy name for vulkan command buffer we're going to submit to gpu through
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            todo: 0,
        });

        //scope the mut self-borrow from encoder.begin_render_pass so it gets dropped before we try to reborrow for finish
        {
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
        }

        self.queue.submit(&[
            encoder.finish()
        ]);
    }
}


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
