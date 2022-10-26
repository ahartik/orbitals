use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};

use wgpu::util::DeviceExt;
use winit::{
    event::{Event, WindowEvent, DeviceEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

extern crate nalgebra as na;

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

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    // Extra size for alignment
    camera_pos: [f32; 4],
    // Same here
    look_matrix: [[f32; 4]; 3],

    quantum_nums:  [f32; 4],
    // 
    rfun_coeff: [f32; 4],
}

type Vec3f = na::Vector3<f32>;
type Vec3d = na::Vector3<f64>;
type Mat3f = na::Matrix3<f32>;

type Vec4f = na::Vector4<f32>;
type Mat4f = na::Matrix4<f32>;

impl Uniforms {
    // fn new(pos: &Vec3f, look_at: &Vec3f, up: 
    fn new(pos: Vec3f, look_at: Vec3f, aspect_ratio: f32) -> Self {
        // We're converting from "screen" coordinates into our physics coordinates.
        // In screen coordinates X and Y go from -1 to 1 from bottom left to bottom right.
        // Z is constant 1.
        // 
        // In "physics coordinates" Z is up/down, X is horizontal and Y is depth.
        
        // This is multiplied by screen-Z (i.e. 1)
        let look_z : Vec3f = (look_at - pos).normalize();
        // This is "up"
        let look_y = Vec3f::new(0.0,0.0,1.0);

        let mut look_x : Vec3f = look_z.cross(&look_y).normalize();
        let mut look_y : Vec3f = look_x.cross(&look_z).normalize();

        // Correct for FOV and aspect ratio
        let fov = (75.0 / 180.0)*std::f32::consts::PI;
        look_x *= (fov/2.0).sin();
        look_y *= (fov/2.0).sin()/aspect_ratio;

        // println!("look_x: {}", look_x);
        // println!("look_y: {}", look_y);
        // println!("look_z: {}", look_z);

        let mut look_mat_array : [[f32; 4]; 3] = [[0.0; 4]; 3];

        for i in 0..3 {
            look_mat_array[0][i] = look_x[i];
            look_mat_array[1][i] = look_y[i];
            look_mat_array[2][i] = look_z[i];
        }
        return Uniforms {
            camera_pos: *pos.push(0.0).as_ref(),
            look_matrix: look_mat_array,
            quantum_nums: [1.0, 0.0, 0.0, 0.0],
            rfun_coeff: [1.0, 0.0, 0.0, 0.0],
        }
    }

}

struct UniformBuffer{
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
            entries: &[
                wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
                    }],
        });
        return Self {
            buffer, layout, bind_group
        };
    }

    fn layout<'a>() -> wgpu::BindGroupLayoutDescriptor<'a> {
        wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("uniform_bind_group"),
        }
    }

    fn queue_update(
        &self, queue: &wgpu::Queue, uniforms: &Uniforms) { 
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(
                std::array::from_ref(uniforms)));
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
    const PI : f64 = std::f64::consts::PI;

    fn new() -> Self {
        Self{
            is_mouse_pressed: false,
            r: 45.0,
            theta: Self::PI/2.0,
            phi: (3.0 * Self::PI/2.0),
        }

    }

    fn process_mouse(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::MouseMotion {
                delta: (dx, dy)
            } => {
                if self.is_mouse_pressed {
                    println!("move {}, {}", dx, dy);
                    self.phi -= Self::SENS_X * dx;
                    self.phi %= 2.0 * Self::PI;
                    self.theta -= Self::SENS_Y * dy;
                    self.theta = self.theta.clamp(0.0, Self::PI);
                
                    return true;
                } else {
                    return false;
                }
            }
            DeviceEvent::Button { button, state } => {
                println!("button: {}", button);
                self.is_mouse_pressed = *state == winit::event::ElementState::Pressed;
                return false;

            }
            _ => false
        }
    }

    fn camera_pos(&self) -> Vec3f {
        return (self.r as f32) * Vec3d::new(
            self.theta.sin() * self.phi.cos(),
            self.theta.sin() * self.phi.sin(),
            self.theta.cos()).cast::<f32>();
    }
}

#[derive(Clone, Debug)]
struct Polynomial {
    coeffs: Vec<f64>,
}

impl Polynomial {
    fn from_coeffs(mut coeffs: Vec<f64>) -> Self {
        while let Some(x) = coeffs.last() {
            if *x == 0.0 {
                coeffs.pop();
            } else {
                break;
            }
        }
        if coeffs.is_empty() {
            coeffs.push(0.0);
        }
        Self {
            coeffs
        }
    }

    fn mul(&self, o: &Polynomial) -> Self {
        let mut coeffs = vec![];
        coeffs.resize(1 + (self.coeffs.len()-1)+(o.coeffs.len()-1), 0.0);
        for i in 0..self.coeffs.len() {
            for j in 0..o.coeffs.len() {
                coeffs[i+j] += self.coeffs[i] * o.coeffs[j];
            }
        }
        return Self {
            coeffs
        };
    }

    fn scale_inplace(&mut self, a: f64) {
        for i in 0..self.coeffs.len() {
            self.coeffs[i] *= a;
        }
    }

    fn diff_inplace(&mut self) {
        for i in 1..self.coeffs.len() {
            self.coeffs[i-1] = (i as f64) * self.coeffs[i];
        }
        self.coeffs.pop();
        if self.coeffs.is_empty() {
            self.coeffs.push(0.0);
        }
    }
}


fn gen_legendre(maxn: usize) -> Vec<Polynomial> {
    // (x^2-1)^n
    let mut base : Vec<Polynomial> = vec![];
    let mut last = Polynomial::from_coeffs(vec![1.0]);
    let x2 = Polynomial::from_coeffs(vec![-1.0, 0.0, 1.0]);

    base.push(last.clone());

    for _ in 1..(maxn+1) {
        last = last.mul(&x2);
        base.push(last.clone());
    }

    let mut res : Vec<Polynomial> = vec![];
    res.push(Polynomial::from_coeffs(vec![1.0]));
    for n in 1..(maxn+1) {
        // P_n = (1/(2^n * n!))(d/dx)^n (x^2-1)^n
        let mut p = base[n].clone(); // (x^2-1)^n
        for k in 1..(n+1) {
            p.diff_inplace();
            p.scale_inplace(1.0 / (2.0 * (k as f64)));
        }
        res.push(p);
    }
    return res;
}


// 
struct AppState {
    camera: CameraController,
    aspect_ratio: f32,
    n: i32,
    l: i32,
    m: i32,
}

async fn run(event_loop: EventLoop<()>, window: Window) {
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
                limits: if cfg!(target_arch = "wasm32") {
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
            },
            None,
        )
        .await
        .expect("Failed to create device");

    // Load the shaders from disk
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

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

    let swapchain_format = surface.get_supported_formats(&adapter)[0];
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

    let mut config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: swapchain_format,
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Fifo,
        alpha_mode: surface.get_supported_alpha_modes(&adapter)[0],
    };

    let mut camera = CameraController::new();
    let mut aspect_ratio = (size.width as f32) / (size.height as f32);
    surface.configure(&device, &config);

    event_loop.run(move |event, _, control_flow| {
        // Have the closure take ownership of the resources.
        // `event_loop.run` never returns, therefore we must do this to ensure
        // the resources are properly cleaned up.
        let _ = (&instance, &adapter, &shader, &pipeline_layout);

        *control_flow = ControlFlow::Wait;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                // Reconfigure the surface with the new size
                config.width = size.width;
                config.height = size.height;

                aspect_ratio = (size.width as f32) / (size.height as f32);
                surface.configure(&device, &config);
                // On macos the window needs to be redrawn manually after resizing
                window.request_redraw();
            },
            Event::DeviceEvent{ event, device_id:_} => {
                if camera.process_mouse(&event) {
                    window.request_redraw();
                }
            },
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let uniforms = Uniforms::new(camera.camera_pos(), Vec3f::zeros(), aspect_ratio);
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
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    rpass.set_bind_group(0, &uniform_buffer.bind_group, &[]);
                    rpass.draw(0..(QUAD_VERTICES.len() as u32), 0..1);
                }


                queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });
}

fn main() {
    let event_loop = EventLoop::new();
    let window = winit::window::Window::new(&event_loop).unwrap();


    let legendre = gen_legendre(5);
    for x in legendre {
        println!("{:?}", x);
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        // Temporarily avoid srgb formats for the swapchain on the web
        pollster::block_on(run(event_loop, window));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        use winit::platform::web::WindowExtWebSys;
        // On wasm, append the canvas to the document body
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| doc.body())
            .and_then(|body| {
                body.append_child(&web_sys::Element::from(window.canvas()))
                    .ok()
            })
            .expect("couldn't append canvas to document body");
        wasm_bindgen_futures::spawn_local(run(event_loop, window));
    }
}
