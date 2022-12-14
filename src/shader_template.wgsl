
struct VertexInput {
  @location(0) position: vec2<f32>,
};
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  // This doesn't get transformed
  @location(0) pos: vec2<f32>,
};


@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var pos = vec4<f32>(1.0*input.position, 0.0, 1.0);
    var result: VertexOutput;
    result.position = pos;
    result.pos = input.position;
    return result;
}

struct Uniforms {
  camera_pos: vec3<f32>,
  look_matrix: mat3x3<f32>,

  // All non-vector stuff should be in the end:
  surf_limit: f32,
}

@group(0)
@binding(0)
var<uniform> uniforms: Uniforms;

let PI : f32 = 3.14159265;
let N : i32 = VAL_N;
let L : i32 = VAL_L;
let M : i32 = VAL_M;

// Defines:
// TODO: Modify minipre lib so that the existance of these is verified:
// - REAL_ORBITALS
// - CUT_HALF
// - YFUNC
// - RFUNC
// - PHI_SYMMETRIC

// This is called from generated code.
fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn hydrogen_wave(pos: vec3<f32>) -> f32 {
  let r : f32 = length(pos);

  let xylen : f32 = length(vec2(pos.x, pos.y));
  let costheta : f32 = pos.z / r;
  let sintheta : f32 = xylen / r;

  let cosphi : f32 = pos.x / xylen;
  let sinphi : f32 = pos.y / xylen;

  // Put generated code here 
  WAVE_FUNC
}

fn hydrogen(pos: vec3<f32>) -> f32 {
  return pow(hydrogen_wave(pos), 2.0);
}

fn normal(pos: vec3<f32>, dpos: vec3<f32>, cval: f32) -> vec3<f32> {
  // Calculate in tangent space.
  let H = 0.5 * length(dpos); // 0.05;

  var r = length(pos);
  // let theta = atan2(r, pos.z);
  let costheta = pos.z / r;
  let xylen = length(vec2(pos.x, pos.y));
  let sintheta = xylen / r;

  // let phi = atan2(pos.y, pos.x);

  let cosphi = pos.x / xylen;
  let sinphi = pos.y / xylen;

  let unit_r = normalize(pos) ;
  let unit_phi = vec3(-sinphi, cosphi, 0.0);

  let unit_theta =
    -sintheta * vec3(0.0, 0.0, 1.0)
    + costheta * vec3(cosphi, sinphi, 0.0);

  let ddr = (hydrogen( pos + H* unit_r)-cval) / H;
  let ddtheta = (hydrogen( pos + H * unit_theta)-cval) / H;

#if PHI_SYMMETRIC
  // "Complex"
  let ddphi = 0.0;
#else
  // "Real"
  let ddphi = (hydrogen(pos + H * unit_phi)-cval) / H;
#endif

   return -normalize(
       unit_r * ddr +
       unit_theta * ddtheta +
       unit_phi * ddphi
       );
}

fn pos_color(pos: vec3<f32>) -> vec3<f32> {
  let phi = atan2(pos.y, pos.x);

  // "Complex"
#if PHI_SYMMETRIC
  // TODO: Make this generic input to the shader template.
  var arg = f32(M) * (phi + 2.0 * PI);
#else
  // "Real"
  var arg = 0.0;
#endif

  let psi = hydrogen_wave(pos);
  if (psi < 0.0) {
    arg += PI;
  }
  arg %= (2.0 * PI);

  let h = arg;
  let s = 0.8;
  let l = 0.5;

  // HSL to RGB
  let hp = arg / (PI / 3.0);

  let c = (1.0 - abs(2.0 * l - 1.0))*s;
  let x = c * (1.0 - abs(hp % 2.0 - 1.0));
  if (hp < 1.0) {
    return vec3(c, x, 0.0);
  }
  if (hp < 2.0) {
    return vec3(x, c, 0.0);
  }
  if (hp < 3.0) {
    return vec3(0.0, c, x);
  }
  if (hp < 4.0) {
    return vec3(0.0, x, c);
  }
  if (hp < 5.0) {
    return vec3(x, 0.0, c);
  }
  if (hp < 6.0) {
    return vec3(c, 0.0, x);
  }
  return vec3(1.0, 1.0, 1.0);
}

fn lighting(pos: vec3<f32>, normal: vec3<f32>, light_pos: vec3<f32>) -> vec4<f32>{
  let AMBIENT_COLOR = vec3(1.0, 1.0, 1.0);

  let base_color = pos_color(pos);

  var color = vec3(0.0, 0.0, 0.0);
  color += 0.1 * base_color;

  let to_light = normalize(light_pos - pos);
  // Diffuse lighting
  color += max(0.0, 0.9 * dot(normal, to_light)) * base_color;

  return vec4(color, 1.0);
}


@fragment
fn fs_main(vertex : VertexOutput)->@location(0) vec4<f32> {
  var pos = uniforms.camera_pos;

  // var light_pos = vec3(normalize(vec2(pos.x, pos.y)) * 100.0 , 100.0);
  // var light_pos = vec3(0.0, -100.0, 100.0);
  // var light_pos = pos + uniforms.look_matrix * vec3(0.0, 100.0, 0.0);
  var light_pos = uniforms.look_matrix * vec3(0.0, 1000.0, -500.0);

  var v = vec3(vertex.pos.x, vertex.pos.y, 1.0);
  let ray = normalize(uniforms.look_matrix * v);

  // Limit following to sphere of certain radius.
  // This prevents rendering glitches when zoomed too far out.
  let MAX_RADIUS : f32 = 150.0;
  if (length(pos) > MAX_RADIUS) {
    // We're outside this radius, let's et 
    let c = dot(pos, pos) - MAX_RADIUS * MAX_RADIUS;
    let b = 2.0 * dot(ray, pos);
    let a = dot(ray, ray);
    if (b * b < 4.0 * a * c) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    }
    // // Solve ax^2 + bx + c = 0
    let t1 = (-b - sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
    // let t2 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a);
    pos = pos + t1 * ray;
  }

  let LIMIT = uniforms.surf_limit;

  let EVAL_N = 256;
  let MAX_DIST = 2.0 * length(pos);
  let DX = MAX_DIST / f32(EVAL_N);
  let dpos = DX * ray;
  var dist = 0.0;
  var prob : f32 = 0.0;
  var last = 0.0;
  var last_pos = pos;



#if CUT_HALF
  if (pos.y < 0.0) {
    // Possibly need to render the flat "cut" face
    if (dpos.y < 0.0) {
      return vec4(0.0, 0.0, 0.0, 0.0);
    }
    pos += dpos * (-pos.y) / dpos.y;
    var h = hydrogen(pos);
    if h > LIMIT {
      let n = vec3(0.0, -1.0, 0.0);
      return lighting(pos, n, light_pos);
    }
  }
#endif



  for (var i : i32 = 0; i < EVAL_N; i++) {
#if CUT_HALF
    if (pos.y < -0.0001) {

      // Change pos back so that pos.y = 0
      pos -= dpos * (pos.y) / dpos.y;
      var h = hydrogen(pos);
      // return vec4(0.2*pos, 1.0);
      if h > LIMIT {
        let n = normal(pos, dpos, h);
        let col = lighting(pos, n, light_pos);
        // return vec4(col.x, 0.0, 0.0, col.w);
        return col;
      }
      break;
    }
#endif

    var h = hydrogen(pos);
    if h > LIMIT {
      // Secant method for a few iterations:
      var a = 0.0;
      var b = 1.0;
      var ah = last;
      var bh = h;
      for (var j: i32 = 0; j < 2; j++) {
        // let c = (a + b) * (bh - LIMIT) / (bh - ah);
        let c = a + (b-a)*(LIMIT - ah) / (bh - ah);
        pos = last_pos + dpos * c;
        h = hydrogen(pos);
        if (h < LIMIT) {
          a = c;
          ah = h;
        } else {
          b = c;
          bh = h;
        }
      }

      let n = normal(pos, 0.5 * dpos, h);
      return lighting(pos, n, light_pos);
    }

    last = h;
    prob += h * DX;

    last_pos = pos;
    pos += dpos;
  }
  prob *= 200.0;
  return vec4(prob, prob, prob, 1.0);
}
