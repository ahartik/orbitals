use std::borrow::Cow;
use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};

use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalPosition,
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use log::{
    info,
    debug
};

extern crate nalgebra as na;

mod build_shader;
use build_shader::{ShaderBuilder, ShaderParams};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    position: [f32; 2],
}

const fn vertex(x: f32, y: f32) -> Vertex {
    return Vertex {
        position: [x as f32, y as f32],
    };
}

impl Vertex {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            }],
        }
    }
}

const QUAD_VERTICES: [Vertex; 6] = [
    vertex(-1.0, -1.0),
    vertex(1.0, -1.0),
    vertex(1.0, 1.0),
    vertex(1.0, 1.0),
    vertex(-1.0, 1.0),
    vertex(-1.0, -1.0),
];

// This must have the same bit layout as the Uniforms struct in the shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    // Extra size for alignment
    camera_pos: [f32; 4],
    // Same here
    look_matrix: [[f32; 4]; 3],

    surf_limit: f32,
    padding: [u32; 3],
}

type Vec3f = na::Vector3<f32>;
type Vec3d = na::Vector3<f64>;
type Mat3f = na::Matrix3<f32>;

// type Vec4f = na::Vector4<f32>;
// type Mat4f = na::Matrix4<f32>;

struct UniformBuffer {
    buffer: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl UniformBuffer {
    fn create(device: &wgpu::Device) -> Self {
        let zero_uniforms = Uniforms::zeroed();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[zero_uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&Self::layout());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("my uniforms"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        return Self {
            buffer,
            layout,
            bind_group,
        };
    }

    fn layout<'a>() -> wgpu::BindGroupLayoutDescriptor<'a> {
        wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("uniform_bind_group"),
        }
    }

    fn queue_update(&self, queue: &wgpu::Queue, uniforms: &Uniforms) {
        queue.write_buffer(
            &self.buffer,
            0,
            bytemuck::cast_slice(std::array::from_ref(uniforms)),
        );
    }

    /*
     fn desc<'a>(layout: &'a wgpu::BindGroupLayout,
         buffer: &'a wgpu::Buffer)-> wgpu::BindGroupDescriptor<'a> {
         return
             wgpu::BindGroupDescriptor {
                 label: Some("my uniforms"),
                 layout,
                 entries: &[
             wgpu::BindGroupEntry {
             binding: 0,
             resource:
                 wgpu::BindingResource::Buffer(
                     wgpu::BufferBinding {
                         buffer,
                         offset: 0,
                         size: wgpu::BufferSize::new(std::mem::size_of::<[f32; 3]>() as u64),
                     })
         }]
             };
     }
    */
}

struct CameraController {
    is_mouse_pressed: bool,
    r: f64,
    theta: f64,
    phi: f64,
}

impl CameraController {
    const SENS_X: f64 = 0.004;
    const SENS_Y: f64 = 0.004;
    const PI: f64 = std::f64::consts::PI;

    fn new() -> Self {
        Self {
            is_mouse_pressed: false,
            r: 45.0,
            theta: Self::PI / 2.0,
            phi: (3.0 * Self::PI / 2.0),
        }
    }

    fn pan(&mut self, dx: f64, dy: f64) {
        self.phi -= 1.5 * Self::SENS_X * dx;
        self.phi %= 2.0 * Self::PI;
        self.theta -= 1.5 * Self::SENS_Y * dy;
        self.theta = self.theta.clamp(0.0, Self::PI);
    }

    fn process_mouse(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                if self.is_mouse_pressed {
                    self.pan(*dx, *dy);

                    return true;
                } else {
                    return false;
                }
            }
            DeviceEvent::Button { state, .. } => {
                self.is_mouse_pressed = *state == winit::event::ElementState::Pressed;
                return false;
            }
            DeviceEvent::MouseWheel {
                delta
            } => {
                let y : f64 = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_x, y) => *y as f64,
                    winit::event::MouseScrollDelta::PixelDelta(
                        PhysicalPosition{y, ..}) => -*y,
                };
                debug!("scroll {}", y);
                if y > 0.0 {
                    self.r *= 1.125;
                } else {
                    self.r /= 1.125;
                }
                return true;
            }
            _ => false,
        }
    }

    fn camera_pos(&self) -> Vec3f {
        return (self.r as f32)
            * Vec3d::new(
                self.theta.sin() * self.phi.cos(),
                self.theta.sin() * self.phi.sin(),
                self.theta.cos(),
            )
            .cast::<f32>();
    }

    // This matrix maps from ((-1, 1), (-1, 1), 1) to rays
    // In "math" coordinates Z-axel goes from bottom to top, while X and Y
    // are horizontal.
    fn look_matrix(&self, aspect_ratio: f32) -> Mat3f {
        // Target the camera is pointing at
        let look_at = Vec3f::zeros();
        let pos = self.camera_pos();
        // This is multiplied by screen-Z (i.e. 1)
        let look_z: Vec3f = (look_at - pos).normalize();
        // This is "up"
        let mut look_y = Vec3f::new(
            -(self.phi.cos() * self.theta.cos()) as f32,
            -(self.phi.sin() * self.theta.cos()) as f32,
            self.theta.sin() as f32,
        );

        let mut look_x = Vec3f::new(-self.phi.sin() as f32, self.phi.cos() as f32, 0.0);
        debug!("look_x: {}", look_x);
        debug!("look_y: {}", look_y);
        debug!("look_z: {}", look_z);

        // Correct for FOV and aspect ratio
        let fov = (75.0 / 180.0) * std::f32::consts::PI;
        look_x *= (fov / 2.0).sin();
        look_y *= (fov / 2.0).sin() / aspect_ratio;

        return Mat3f::from_rows(&[look_x.transpose(), look_y.transpose(), look_z.transpose()]);
    }
}

// Largest supported value of N
const MAX_N: usize = 8;

#[derive(Copy, Clone, Debug)]
pub struct OrbitalParams {
    n: i32,
    l: i32,
    m: i32,
    surf_limit: f64,
    enable_cuts: bool,
    real_orbital: bool,
}

impl OrbitalParams {
    fn new() -> Self {
        Self {
            n: 4,
            l: 2,
            m: 1,
            // Matches 0.25/nm^3 from Griffiths page 153
            surf_limit: 0.25,
            enable_cuts: false,
            real_orbital: true,
        }
    }
    fn sanitize(&mut self) {
        if self.n < 1 {
            self.n = 1;
        }
        if self.n > MAX_N as i32 {
            self.n = 1;
        }
        if self.l >= self.n {
            self.l = 0;
        }
        if self.m > self.l {
            self.m = 0;
        }
    }
    fn surf_limit_bohr(&self) -> f64 {
        // Input is in probability per nanometer, convert to units
        // where bohr radius is 1.
        let conv = 3.7e-05/0.25;
        return conv * self.surf_limit;
    }
}

#[derive(Copy, Clone, Debug)]
struct TouchState {
    last_id: u64,
    last_location: PhysicalPosition<f64>
}

//
struct AppState {
    camera: CameraController,
    aspect_ratio: f32,
    params: OrbitalParams,
    current_touch: Option<TouchState>
}

impl AppState {
    fn new() -> Self {
        Self {
            camera: CameraController::new(),
            aspect_ratio: 1.0,
            params: OrbitalParams::new(),
            current_touch: None,
        }
    }

    fn update_size(&mut self, w: u32, h: u32) {
        self.aspect_ratio = (w as f32) / (h as f32);
    }

    fn shader_params(&self) -> ShaderParams {
        return ShaderParams {
            n: self.params.n,
            l: self.params.l,
            m: self.params.m,
            cut_half: self.params.enable_cuts,
            real_orbital: self.params.real_orbital,
        };
    }

    fn uniforms(&self) -> Uniforms {
        let look_mat = self.camera.look_matrix(self.aspect_ratio);
        let mut look_mat_array: [[f32; 4]; 3] = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                look_mat_array[i][j] = look_mat[(i, j)];
            }
        }
        return Uniforms {
            camera_pos: *self.camera.camera_pos().push(0.0).as_ref(),
            look_matrix: look_mat_array,
            surf_limit: self.params.surf_limit_bohr() as f32,
            padding: [0; 3],
        };
    }

    fn change_params(&mut self, params: &OrbitalParams) {
        self.params = *params;
        info!("params: {:?}", self.params);
    }

    fn process_touch(&mut self, touch: &winit::event::Touch) -> bool {
        info!("Touch: {:?}", touch);
        match touch.phase {
            winit::event::TouchPhase::Started => {
                // TODO: handle multi touch for zoom.
                if self.current_touch.is_none() {
                    self.current_touch = Some(TouchState {
                        last_id: touch.id,
                        last_location: touch.location,
                    });
                }
                return false;
            },
            winit::event::TouchPhase::Cancelled |
            winit::event::TouchPhase::Ended => {
                if let Some(cur) = self.current_touch.clone() {
                    if cur.last_id == touch.id {
                        self.current_touch = None;
                    }
                }
                return false;
            },
            winit::event::TouchPhase::Moved => {
                if let Some(cur) = self.current_touch {
                    if cur.last_id == touch.id {
                        let dx : f64= 
                            touch.location.x - 
                            cur.last_location.x;
                        let dy :f64 = 
                            touch.location.y - 
                            cur.last_location.y;
                        const TOUCH_SENS: f64 = 1.0;
                        self.camera.pan(TOUCH_SENS * dx, TOUCH_SENS * dy);
                        self.current_touch = Some(TouchState {
                            last_id: touch.id,
                            last_location: touch.location,
                        });
                        return true;
                    }
                }
                return false;
            }
        }
    }

    fn process_event(&mut self, event: &DeviceEvent) -> bool {
        if self.camera.process_mouse(event) {
            return true;
        }

        let mut changed = false;
        match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(code),
                state: ElementState::Pressed,
                ..
            }) => {
                type K = VirtualKeyCode;
                match code {
                    K::N => {
                        self.params.n += 1;
                        changed = true;
                    }
                    K::L => {
                        self.params.l += 1;
                        changed = true;
                    }
                    K::M => {
                        self.params.m += 1;
                        changed = true;
                    }
                    K::Equals | K::Plus => {
                        self.params.surf_limit /= 1.125;
                        changed = true;
                    }
                    K::Minus => {
                        self.params.surf_limit *= 1.125;
                        changed = true;
                    }
                    K::C => {
                        self.params.enable_cuts = !self.params.enable_cuts;
                        changed = true;
                    }
                    K::R => {
                        self.params.real_orbital = !self.params.real_orbital;
                        changed = true;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        if changed {
            self.params.sanitize();
            info!("{:?}", self.params);
            return true;
        }
        return false;
    }

}

// Web controls are implemented using custom events that contain new settings
// for the orbital. EventLoop is handy for this.
#[derive(Copy, Clone, Debug)]
pub enum WebUIEvent {
    ChangeParams(OrbitalParams),
    ChangeSize(i32, i32)
}

// Main rendering function.
pub async fn run(event_loop: EventLoop<WebUIEvent>, window: Window) {
    let size = window.inner_size();
    let instance = wgpu::Instance::new(wgpu::Backends::all());
    let surface = unsafe { instance.create_surface(&window) };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            // Request an adapter which can render to our surface
            compatible_surface: Some(&surface),
        })
        .await
        .expect("Failed to find an appropriate adapter");

    // Create the logical device and command queue
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("the_device"),
                features: wgpu::Features::empty(),
                // WebGL doesn't support all of wgpu's features, so if
                // we're building for the web we'll have to disable some.
                limits: wgpu::Limits::downlevel_webgl2_defaults(),
            },
            None,
        )
        .await
        .expect("Failed to create device");

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&QUAD_VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    // UNIFORM STUFF
    let uniform_buffer = UniformBuffer::create(&device);

    // PIPELINE STUFF
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("MyPipelineLayout"),
        bind_group_layouts: &[&uniform_buffer.layout],
        push_constant_ranges: &[],
    });

    // RENDER TO TEXTURE
    

    let swapchain_format = surface.get_supported_formats(&adapter)[0];
    let mut pipelines = HashMap::<ShaderParams, wgpu::RenderPipeline>::new();
    let builder = ShaderBuilder::new(MAX_N);

    let build_pipeline =
        move |device: &wgpu::Device, params: &ShaderParams| -> wgpu::RenderPipeline {
            let shader_src: String = builder.build(&params);
            // Load the shaders from disk
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("OrbitalShader"),
                source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(shader_src.as_str())),
            });
            return device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[Vertex::desc()],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(swapchain_format.into())],
                }),
                // primitive: wgpu::PrimitiveState::default(),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                    // or Features::POLYGON_MODE_POINT
                    polygon_mode: wgpu::PolygonMode::Fill,
                    // Requires Features::DEPTH_CLIP_CONTROL
                    unclipped_depth: false,
                    // Requires Features::CONSERVATIVE_RASTERIZATION
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });
        };

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface.get_supported_alpha_modes(&adapter)[0],
    };

    let mut app = AppState::new();
    app.update_size(size.width, size.height);
    surface.configure(&device, &config);

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        // let _ = (&instance, &adapter, &pipeline_layout);
        let _ = (&instance, &adapter);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Reconfigure the surface with the new size
                config.width = size.width;
                config.height = size.height;

                app.update_size(size.width, size.height);
                surface.configure(&device, &config);
                // On macos the window needs to be redrawn manually after resizing
                window.request_redraw();
            }
            Event::DeviceEvent {
                event,
                device_id: _,
            } => {
                if app.process_event(&event) {
                    window.request_redraw();
                }
            }
            Event::UserEvent(ev) => {
                match ev {
                    WebUIEvent::ChangeParams(params)
                        => {
                        app.change_params(&params);
                    },
                    WebUIEvent::ChangeSize(w,h)
                        => {
                        window.set_inner_size(winit::dpi::LogicalSize::new(w, h));

                    },
                }
                window.request_redraw();
            },
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());

                use std::collections::hash_map::Entry;
                let sp = app.shader_params();
                let pipeline_entry = pipelines.entry(sp);
                let pipeline: &wgpu::RenderPipeline = match pipeline_entry {
                    Entry::Occupied(e) => e.into_mut(),
                    Entry::Vacant(e) => e.insert(build_pipeline(&device, &sp)),
                };

                let uniforms = app.uniforms();
                uniform_buffer.queue_update(&queue, &uniforms);

                let mut encoder =
                    device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: None,
                    });
                    rpass.set_pipeline(pipeline);
                    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    rpass.set_bind_group(0, &uniform_buffer.bind_group, &[]);
                    rpass.draw(0..(QUAD_VERTICES.len() as u32), 0..1);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::RedrawEventsCleared => {
                // window.request_redraw();
            }
            Event::WindowEvent {
                ref event,
                window_id: _,
            } => match event {
                #[cfg(not(target_arch = "wasm32"))]
                WindowEvent::CloseRequested
                | WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                #[cfg(target_arch = "wasm32")]
                WindowEvent::MouseInput {
                    button: winit::event::MouseButton::Left,
                    state,
                    ..
                } => {
                    let fake_event = DeviceEvent::Button { button: 1u32, state: state.clone() };
                    if app.process_event(&fake_event) {
                        window.request_redraw();
                    }
                },
                #[cfg(target_arch = "wasm32")]
                WindowEvent::MouseWheel { delta, .. } => {
                    let fake_event = DeviceEvent::MouseWheel{delta: delta.clone()};
                    if app.process_event(&fake_event) {
                        window.request_redraw();
                    }
                },
                WindowEvent::Touch(touch) => {
                    if app.process_touch(&touch) {
                        window.request_redraw();
                    }
                },
                _ => {
                    // info!("Unhandled event: {:?}", e);
                }
            },
            _ => {}
        }
    });
}

#[cfg(target_arch = "wasm32")]
mod web;
