use std::borrow::Cow;

use bytemuck::{Pod, Zeroable};

use wgpu::util::DeviceExt;
use winit::{
    event::{DeviceEvent, Event, WindowEvent},
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


// This must have the same bit layout as the Uniforms struct in the shader.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    // Extra size for alignment
    camera_pos: [f32; 4],
    // Same here
    look_matrix: [[f32; 4]; 3],

    quantum_nums: [f32; 4],
    //
    rfun_coeffs: [f32; 4],
    yfun_coeffs: [f32; 4],
    rand: u32,
    surf_limit: f32,
    max_phi: f32,
    end_padding: [u32; 1],
}

type Vec3f = na::Vector3<f32>;
type Vec3d = na::Vector3<f64>;
type Mat3f = na::Matrix3<f32>;

type Vec4f = na::Vector4<f32>;
type Mat4f = na::Matrix4<f32>;

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

    fn process_mouse(&mut self, event: &DeviceEvent) -> bool {
        // println!("DeviceEvent: {:?}", event);
        match event {
            DeviceEvent::MouseMotion { delta: (dx, dy) } => {
                if self.is_mouse_pressed {
                    // println!("move {}, {}", dx, dy);
                    self.phi -= Self::SENS_X * dx;
                    self.phi %= 2.0 * Self::PI;
                    self.theta -= Self::SENS_Y * dy;
                    self.theta = self.theta.clamp(0.0, Self::PI);

                    return true;
                } else {
                    return false;
                }
            },
            DeviceEvent::Button { button, state } => {
                // println!("button: {}", button);
                self.is_mouse_pressed = *state == winit::event::ElementState::Pressed;
                return false;
            },
            DeviceEvent::MouseWheel{
                delta: winit::event::MouseScrollDelta::LineDelta(_x, y)
            } => {
                if *y > 0.0 {
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

    fn look_matrix(&self, aspect_ratio: f32) -> Mat3f {
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
        // let mut look_x = look_z.cross(&look_y).normalize();
        // println!("look_x: {}", look_x);
        // println!("look_y: {}", look_y);
        // println!("look_z: {}", look_z);

        // Correct for FOV and aspect ratio
        let fov = (75.0 / 180.0) * std::f32::consts::PI;
        look_x *= (fov / 2.0).sin();
        look_y *= (fov / 2.0).sin() / aspect_ratio;

        return Mat3f::from_rows(&[look_x.transpose(), look_y.transpose(), look_z.transpose()]);
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
        Self { coeffs }
    }

    fn single_term(scale: f64, n: usize) -> Self {
        let mut c = vec![];
        c.resize(n + 1, 0.0);
        c[n] = scale;
        return Self::from_coeffs(c);
    }

    fn mul(&self, o: &Polynomial) -> Self {
        let mut coeffs = vec![];
        coeffs.resize(1 + (self.coeffs.len() - 1) + (o.coeffs.len() - 1), 0.0);
        for i in 0..self.coeffs.len() {
            for j in 0..o.coeffs.len() {
                coeffs[i + j] += self.coeffs[i] * o.coeffs[j];
            }
        }
        return Self { coeffs };
    }

    fn add(&self, o: &Polynomial) -> Self {
        let mut coeffs = vec![];
        coeffs.resize(self.coeffs.len().max(o.coeffs.len()), 0.0);
        for i in 0..self.coeffs.len() {
            coeffs[i] += self.coeffs[i];
        }
        for i in 0..o.coeffs.len() {
            coeffs[i] += o.coeffs[i];
        }
        return Self { coeffs };
    }

    fn sub(&self, o: &Polynomial) -> Self {
        let mut coeffs = vec![];
        coeffs.resize(self.coeffs.len().max(o.coeffs.len()), 0.0);
        for i in 0..self.coeffs.len() {
            coeffs[i] += self.coeffs[i];
        }
        for i in 0..o.coeffs.len() {
            coeffs[i] -= o.coeffs[i];
        }
        return Self { coeffs };
    }

    fn scale_inplace(&mut self, a: f64) {
        for i in 0..self.coeffs.len() {
            self.coeffs[i] *= a;
        }
    }

    fn diff_inplace(&mut self) {
        for i in 1..self.coeffs.len() {
            self.coeffs[i - 1] = (i as f64) * self.coeffs[i];
        }
        self.coeffs.pop();
        if self.coeffs.is_empty() {
            self.coeffs.push(0.0);
        }
    }
}

fn gen_legendre(maxn: usize) -> Vec<Polynomial> {
    // (x^2-1)^n
    let mut base: Vec<Polynomial> = vec![];
    let mut last = Polynomial::from_coeffs(vec![1.0]);
    let x2 = Polynomial::from_coeffs(vec![-1.0, 0.0, 1.0]);

    base.push(last.clone());

    for _ in 1..(maxn + 1) {
        last = last.mul(&x2);
        base.push(last.clone());
    }

    let mut res: Vec<Polynomial> = vec![];
    res.push(Polynomial::from_coeffs(vec![1.0]));
    for n in 1..(maxn + 1) {
        // P_n = (1/(2^n * n!))(d/dx)^n (x^2-1)^n
        let mut p = base[n].clone(); // (x^2-1)^n
        for k in 1..(n + 1) {
            p.diff_inplace();
            p.scale_inplace(1.0 / (2.0 * (k as f64)));
        }
        res.push(p);
    }
    return res;
}

fn factorial(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    return (n as f64) * factorial(n - 1);
}

fn gen_laguerre(maxn: usize) -> Vec<Polynomial> {
    let mut res: Vec<Polynomial> = vec![];
    let mut b = Polynomial::from_coeffs(vec![1.0]);
    for n in 0..(maxn + 1) {
        let mut z = b.clone();
        //println!("z: {:?}", z);
        for _ in 0..n {
            let mut w = z.clone();
            w.diff_inplace();
            //println!("w: {:?}", w);
            z = w.sub(&z);
            //println!("z: {:?}", z);
        }
        //println!("z2: {:?}", z);
        z.scale_inplace(1.0 / factorial(n));
        res.push(z);
        // b = x**(n+1) for next iter
        *b.coeffs.last_mut().unwrap() = 0.0;
        b.coeffs.push(1.0);
    }
    return res;
}

// Largest supported value of N
const MAX_N: i32 = 4;

//
struct AppState {
    camera: CameraController,
    aspect_ratio: f32,
    n: i32,
    l: i32,
    m: i32,
    surf_limit: f64,
    enable_cuts: bool,
    legendre: Vec<Polynomial>,
    laguerre: Vec<Polynomial>,
}

impl AppState {
    fn new() -> Self {
        Self {
            camera: CameraController::new(),
            aspect_ratio: 1.0,
            n: 3,
            l: 0,
            m: 0,
            surf_limit: 3.7e-05,
            enable_cuts: false,
            legendre: gen_legendre(MAX_N as usize),
            laguerre: gen_laguerre((2 * MAX_N) as usize),
        }
    }

    fn update_size(&mut self, w: u32, h: u32) {
        self.aspect_ratio = (w as f32) / (h as f32);
    }

    fn assoc_laguerre(&self, q: i32, p: i32) -> Polynomial {
        let mut poly = self.laguerre[(q + p) as usize].clone();
        for _ in 0..p {
            poly.diff_inplace();
        }
        poly.scale_inplace((-1.0 as f64).powi(p));
        return poly;
    }

    fn assoc_legendre(&self, l: i32, m: i32) -> Polynomial {
        assert!(m >= 0);
        let mut poly = self.legendre[l as usize].clone();
        for _ in 0..m {
            poly.diff_inplace();
        }
        poly.scale_inplace((-1.0 as f64).powi(m));
        return poly;
    }

    fn uniforms(&self) -> Uniforms {
        let look_mat = self.camera.look_matrix(self.aspect_ratio);
        let mut look_mat_array: [[f32; 4]; 3] = [[0.0; 4]; 3];
        for i in 0..3 {
            for j in 0..3 {
                look_mat_array[i][j] = look_mat[(i, j)];
            }
        }

        // Coefficients for R(r):
        let rmul = f64::sqrt(
            (2.0 / (self.n as f64)).powi(3) * factorial((self.n - self.l - 1) as usize)
                / (2.0 * (self.n as f64) * factorial((self.n + self.l) as usize)),
        );
        let p = 2 * self.l + 1;
        let q = self.n - self.l - 1;
        let mut rfun = self.assoc_laguerre(q, p);
        for i in 0..rfun.coeffs.len() {
            rfun.coeffs[i] *= (2.0 / (self.n as f64)).powi(i as i32);
        }
        // println!("L_{}^{} {:?}", q, p, rfun);
        rfun.scale_inplace(rmul);

        rfun = rfun.mul(&Polynomial::single_term(
            (2.0 / (self.n as f64)).powi(self.l),
            self.l as usize,
        ));

        let mut rfun_coeffs: [f32; 4] = [0.0; 4];
        assert!(rfun.coeffs.len() <= 4);
        for i in 0..rfun.coeffs.len() {
            rfun_coeffs[i] = rfun.coeffs[i] as f32;
        }

        // Coefficients for Y(theta, phi):
        let mut yfun = self.assoc_legendre(self.l, self.m.abs());
        yfun.scale_inplace(f64::sqrt(
            ((2 * self.l + 1) as f64 / (4.0 * std::f64::consts::PI))
                * (factorial((self.l - self.m) as usize) / factorial((self.l + self.m) as usize)),
        ));
        let mut yfun_coeffs: [f32; 4] = [0.0; 4];
        assert!(yfun.coeffs.len() <= 4);
        for i in 0..yfun.coeffs.len() {
            yfun_coeffs[i] = yfun.coeffs[i] as f32;
        }
return Uniforms {
            camera_pos: *self.camera.camera_pos().push(0.0).as_ref(),
            look_matrix: look_mat_array,
            quantum_nums: [self.n as f32, self.l as f32, self.m as f32, 0.0],
            rfun_coeffs,
            yfun_coeffs,
            rand: rand::random::<u32>(),
            surf_limit: self.surf_limit as f32,
            max_phi: 
                if self.enable_cuts {
                    0.5 * std::f32::consts::PI
                } else {
                    2.0 * std::f32::consts::PI
                }
            ,
            end_padding: [0],
        };
    }

    fn process_event(&mut self, event: &DeviceEvent) -> bool {
        if self.camera.process_mouse(event) {
            return true;
        }

        let mut changed = false;
        match event {
            DeviceEvent::Key(
                winit::event::KeyboardInput {
                    virtual_keycode: Some(code),
                    state: winit::event::ElementState::Pressed,
                    ..
                }) => {
                type K = winit::event::VirtualKeyCode;
                match code {
                    K::N => {
                        self.n += 1;
                        changed = true;
                    },
                    K::L => {
                        self.l += 1;
                        changed = true;
                    },
                    K::M => {
                        self.m += 1;
                        changed = true;
                    },
                    K::Equals | K::Plus => {
                        self.surf_limit /= 1.25;
                        changed = true;
                    },
                    K::Minus => {
                        self.surf_limit *= 1.25;
                        changed = true;
                    },
                    K::C => {
                        self.enable_cuts = !self.enable_cuts;
                        changed = true;
                    },
                    _ => {}
                }
            },
            _ => {}
        }
        if changed {
            if self.n > MAX_N {
                self.n = 1;
            }
            if self.l >= self.n {
                self.l = 0;
            }
            if self.m > self.l {
                self.m = 0;
            }
            println!("N={} L={} M={}", self.n, self.l, self.m);
            println!("surf_limit={} enable_cuts={}", self.surf_limit, self.enable_cuts);
            return true;
        }
        return false;
    }
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

    let mut app = AppState::new();
    app.aspect_ratio = (size.width as f32) / (size.height as f32);
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

                app.aspect_ratio = (size.width as f32) / (size.height as f32);
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
            Event::RedrawRequested(_) => {
                let frame = surface
                    .get_current_texture()
                    .expect("Failed to acquire next swap chain texture");
                let view = frame
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
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
                    rpass.set_pipeline(&render_pipeline);
                    rpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                    rpass.set_bind_group(0, &uniform_buffer.bind_group, &[]);
                    rpass.draw(0..(QUAD_VERTICES.len() as u32), 0..1);
                }

                queue.submit(Some(encoder.finish()));
                frame.present();
                // XXX: Not needed for regular rendering
                // window.request_redraw();
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

    // window.set_cursor_grab(winit::window::CursorGrabMode::Confined).unwrap();

    let legendre = gen_legendre(5);
    for x in legendre {
        println!("Legendre: {:?}", x);
    }
    let lag = gen_laguerre(5);
    for x in lag {
        println!("Laguerre: {:?}", x);
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
