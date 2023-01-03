use super::{run, OrbitalParams, WebUIEvent};
use log::info;
use winit::event_loop::{EventLoop, EventLoopBuilder, EventLoopProxy};

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WebApp {
    loop_proxy: EventLoopProxy<WebUIEvent>,
    params: OrbitalParams,
}

#[wasm_bindgen]
impl WebApp {
    fn new(eloop: &EventLoop<WebUIEvent>) -> Self {
        Self {
            loop_proxy: eloop.create_proxy(),
            params: OrbitalParams::new(),
        }
    }
    pub fn get_n(&self) -> i32 {
        return self.params.n;
    }
    pub fn set_n(&mut self, n: i32) {
        self.params.n = n;
        self.params.sanitize();
        self.loop_proxy
            .send_event(WebUIEvent::ChangeParams(self.params))
            .unwrap();
    }
    pub fn get_l(&self) -> i32 {
        return self.params.l;
    }
    pub fn set_l(&mut self, l: i32) {
        self.params.l = l;
        self.params.sanitize();
        self.loop_proxy
            .send_event(WebUIEvent::ChangeParams(self.params))
            .unwrap();
    }
    pub fn get_m(&self) -> i32 {
        return self.params.m;
    }
    pub fn set_m(&mut self, m: i32) {
        self.params.m = m;
        self.params.sanitize();
        self.loop_proxy
            .send_event(WebUIEvent::ChangeParams(self.params))
            .unwrap();
    }

    pub fn get_surf_limit(&self) -> f64 {
        return self.params.surf_limit;
    }
    pub fn set_surf_limit(&mut self, s: f64) {
        self.params.surf_limit = s;
        self.loop_proxy
            .send_event(WebUIEvent::ChangeParams(self.params))
            .unwrap();
    }
    pub fn get_cut(&self) -> bool {
        return self.params.enable_cuts;
    }
    pub fn set_cut(&mut self, c: bool) {
        self.params.enable_cuts = c;
        self.loop_proxy
            .send_event(WebUIEvent::ChangeParams(self.params))
            .unwrap();
    }
    pub fn get_real(&self) -> bool {
        return self.params.real_orbital;
    }
    pub fn set_real(&mut self, r: bool) {
        self.params.real_orbital = r;
        self.loop_proxy
            .send_event(WebUIEvent::ChangeParams(self.params))
            .unwrap();
    }

    pub fn set_size(&mut self, w: i32, h: i32) {
        self.loop_proxy.send_event(WebUIEvent::ChangeSize(w, h))
            .unwrap();
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub fn web_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    console_log::init().expect("could not initialize logger");
    info!("logging initialized");
}

#[wasm_bindgen]
pub fn web_start_app() -> WebApp {
    let event_loop = EventLoopBuilder::<WebUIEvent>::with_user_event().build();
    let window = winit::window::Window::new(&event_loop).unwrap();
    window.set_inner_size(winit::dpi::PhysicalSize::new(400, 400));
    // window.set_outer_size(winit::dpi::LogicalSize(800, 800));
    use winit::platform::web::WindowExtWebSys;

    let app = WebApp::new(&event_loop);

    // Add window to canvas in document body.
    let win_canvas = window.canvas();
    win_canvas.set_id("wasm-canvas");
    web_sys::window()
        .and_then(|win| win.document())
        .and_then(|doc| {
            let dst = doc.get_element_by_id("wasm-canvas-div")?;
            let canvas = web_sys::Element::from(win_canvas);
            dst.append_child(&canvas).ok()?;
            Some(())
        })
        .expect("couldn't append canvas to document body");
    wasm_bindgen_futures::spawn_local(run(event_loop, window));
    return app;
}
