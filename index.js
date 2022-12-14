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

    // Show/hide controls
    let showControls = true;

    let showButton = document.getElementById("show");
    let hideButton = document.getElementById("hide");
    let controlDiv = document.getElementById("inputs");
    showButton.hidden = true;

    showButton.addEventListener('click', (event) => {
      showControls = true;
      showButton.hidden = true;
      hideButton.hidden = false;
      controlDiv.hidden = false;
    });
    hideButton.addEventListener('click', (event) => {
      showControls = false;
      showButton.hidden = false;
      hideButton.hidden = true;
      controlDiv.hidden = true;
    });

    // Canvas size stuff
    const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
    let canvas = document.getElementById("wasm-canvas");

    let scale = 1;

    function resizeCanvas() {
      let w = window.innerWidth;
      let h = window.innerHeight;

      app.set_size(w/scale, h/scale);
    }

    let scale1 = document.getElementById("scale1");
    let scale2 = document.getElementById("scale2");
    let scale3 = document.getElementById("scale3");
    function changeScale(event) {
      if (scale1.checked) {
        scale = 1;
      } else if (scale2.checked) {
        scale = 2;
      } else if (scale3.checked) {
        scale = 3;
      }
      resizeCanvas();
    }
    scale1.addEventListener('change', changeScale);
    scale2.addEventListener('change', changeScale);
    scale3.addEventListener('change', changeScale);

    if (isMobile) {
      scale3.checked = true;
      scale = 3;
    } else {
      scale2.checked = true;
      scale = 2;
    }

    window.addEventListener('resize', resizeCanvas, false);
    // Draw canvas border for the first time.
    resizeCanvas();
});

