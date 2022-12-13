import {default as init, web_start_app} from "./pkg/orbitals.js";
init().then(() => {
    console.log("WASM Loaded");
    let app = web_start_app();
    console.log(app);

    let qn = document.getElementById("qn");
    let ql = document.getElementById("ql");
    let qm = document.getElementById("qm");

    let surf_range = document.getElementById("surf_range");
    let surf_number = document.getElementById("surf_number");

    qn.value = app.get_n()
    ql.value = app.get_l()
    qm.value = app.get_m()

    console.log(app.get_n());
    console.log(app.get_l());
    console.log(app.get_m());

    // Logarithmic / exponential 
    let surf_min = 0.0001;
    let surf_default = 0.25;
    let surf_range_default = 0.8;
    // min * exp(B * 1.0) = center
    // B = ln(center / min)
    let surf_B = Math.log(surf_default / surf_min);

    function sanitize() {
      qn.value = app.get_n()
      ql.value = app.get_l()
      qm.value = app.get_m()

      let s = app.get_surf_limit();
      surf_number.value = s;
      // min * exp(B * x) = s 
      // B * x = log(s / min)
      // x = log(s/min) / B
      surf_range.value = (Math.log(s/ surf_min) / surf_B) * surf_range_default;
    }

    qn.addEventListener('change', (event) => {
      app.set_n(event.target.value);
      sanitize();
    });
    ql.addEventListener('change', (event) => {
      app.set_l(event.target.value);
      sanitize();
    });
    qm.addEventListener('change', (event) => {
      app.set_m(event.target.value);
      sanitize();
    });

    surf_range.addEventListener('input', (event) => {
      let x = event.target.value;
      let val = surf_min * Math.exp(surf_B * x / surf_range_default);
      app.set_surf_limit(val);
      sanitize();
    });
    surf_number.addEventListener('change', (event) => {
      let val = event.target.value;
      app.set_surf_limit(val);
      sanitize();
    });

    let cut = document.getElementById("cut");
    cut.checked = app.get_cut();
    cut.addEventListener('change', (event) => {
      let val = event.target.checked;
      app.set_cut(val);
      sanitize();
    });

    let complex = document.getElementById("complex");
    let real = document.getElementById("real");
    if (app.get_real()) {
      real.checked = true;
    } else {
      complex.checked = true;
    }
    real.addEventListener('change', (event) => {
      let val = event.target.checked;
      app.set_real(val);
    });
    complex.addEventListener('change', (event) => {
      let val = event.target.checked;
      app.set_real(!val);
    });


    // Fix canvas:
    let canvas = document.getElementById("wasm-canvas");

    // XXX
    let lowres = document.getElementById("lowres");

    function resizeCanvas() {
      console.log("resize:");
      let w = window.innerWidth;
      let h = window.innerHeight;
      console.log(w);
      console.log(h);
      if (lowres.checked) {
        app.set_size(w/2, h/2);
      } else {
        app.set_size(w, h);
      }
      canvas.removeAttribute("style");
    }

    lowres.addEventListener('change', (event) => {
      resizeCanvas();
    });

    canvas.removeAttribute("style");
    window.addEventListener('resize', resizeCanvas, false);
    // Draw canvas border for the first time.
    resizeCanvas();
});

