use env_logger::Env;
use orbitals::WebUIEvent;
use winit::event_loop::EventLoopBuilder;

fn main() {
    let event_loop = EventLoopBuilder::<WebUIEvent>::with_user_event().build();
    let window = winit::window::Window::new(&event_loop).unwrap();

    let env = Env::default().filter_or("RUST_LOG", "info");
    env_logger::init_from_env(env);
    // Temporarily avoid srgb formats for the swapchain on the web
    pollster::block_on(orbitals::run(event_loop, window));
}
