
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
  quantum_nums: vec3<f32>,
  rfun_coeffs: vec4<f32>,
  yfun_coeffs: vec4<f32>,

  // All non-vector stuff should be in the end:
  rand: u32,
  surf_limit: f32,
  cut_quarter: u32,
  padding: f32
}

@group(0)
@binding(0)
var<uniform> uniforms: Uniforms;
// @group(0)
// @binding(1)
// var<uniform> look_at: vec3<f32>;

let PI : f32 = 3.14159265;


fn hydrogen(pos: vec3<f32>) -> f32 {
  var uni = uniforms;
  var rfun_coeffs = uni.rfun_coeffs;
  var n = uni.quantum_nums.x;
  var l = uni.quantum_nums.y;
  var m = uni.quantum_nums.z;


  var r = length(pos);
  // let theta = atan2(r, pos.z);
  let costheta = pos.z / r;
  let sintheta = length(vec2(pos.x,pos.y)) / r;
  // let sintheta = sin(theta);

  if (uniforms.cut_quarter!=0u && pos.x < 0.0 && pos.y > 0.0) {
    return 0.0;
  }


  // Y00:
  // var yfun = sqrt(1.0/(4.0*PI));
  // Y10:
  // var yfun = sqrt(3.0/(4.0*PI)) * costheta;
  // Y11:
  // var yfun = sqrt(3.0/(8.0*PI)) * sintheta;
  
  let thetapows =
    vec4(1.0, costheta, costheta * costheta, costheta * costheta * costheta);
  // "Real" orbitals
  // let phi = atan2(pos.y, pos.x);
  // var yfun = dot(uni.yfun_coeffs, thetapows) * pow(sintheta, m) * cos(m * phi);
  // "Complex" orbitals
  var yfun = dot(uni.yfun_coeffs, thetapows) * pow(sintheta, m);

  // R10
  // var rfun = 2.0 * exp(-r);
  // R21
  // var rfun = (1.0/(2.0*sqrt(6.0))) *r* exp(-0.5 * r);
  // R31
  // var rfun = (8.0/(27.0*sqrt(6.0)))*(1.0-r/6.0)*r* exp(-r/3.0);

  var rpows = vec4(1.0, r, r * r, r * r * r);
  var rfun = dot(rfun_coeffs, rpows) * exp(-r / n);

  var wave = yfun * rfun;
  return wave * wave;
}

fn normal(pos: vec3<f32>, dpos: vec3<f32>, cval: f32) -> vec3<f32> {
  // Calculate in tangent space.
  let H  = length(dpos); // 0.05;

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
    +
    costheta * vec3(cosphi, sinphi, 0.0);

  let ddr = (hydrogen( pos + H* unit_r)-cval) / H;
  let ddtheta = (hydrogen( pos + H * unit_theta)-cval) / H;

  // For complex orbitals this is always 0.0
  // let ddphi = (hydrogen( pos + H * unit_phi)-cval) / H;
  let ddphi = 0.0;

  let last_xy = pos.xy - dpos.xy;

  if (uniforms.cut_quarter!=0u && last_xy.x < 0.0 && last_xy.y > 0.0) {
    // Hit the plane.
    if (dot(unit_phi, dpos) < 0.0) {
      return unit_phi;
    } else {
      return -unit_phi;
    }
  }

  // return unit_theta;
   return -normalize(
       unit_r * ddr +
       unit_theta * ddtheta +
       unit_phi * ddphi
       );
}

fn lighting(pos: vec3<f32>, normal: vec3<f32>, light_pos: vec3<f32>) -> vec4<f32>{
  let AMBIENT_COLOR = vec3(1.0, 1.0, 1.0);

  var color = vec3(0.0, 0.0, 0.0);
  color += 0.3 * AMBIENT_COLOR;

  let to_light = normalize(light_pos - pos);
  // Diffuse lighting
  color += max(0.0, 0.7 * dot(normal, to_light)) * vec3(1.0, 1.0, 1.0);

  return vec4(color, 1.0);
}

// var seed: u32 = 0;

fn hash(s: u32) -> u32{
  var seed = s;
  seed ^= 2747636419u;
  seed *= 2654435769u;
  seed ^= seed >> 16u;
  seed *= 2654435769u;
  seed ^= seed >> 16u;
  seed *= 2654435769u;
  return seed;
}

fn get_rand(seed: u32) -> f32{
  return f32(seed) / 4294967295.0;
}

@fragment
fn fs_main(vertex : VertexOutput)->@location(0) vec4<f32> {
  var pos = uniforms.camera_pos;

  // var light_pos = vec3(normalize(vec2(pos.x, pos.y)) * 100.0 , 100.0);
  // var light_pos = vec3(0.0, -100.0, 100.0);
  var light_pos = pos + uniforms.look_matrix * vec3(0.0, 100.0, 0.0);

  var v = vec3(vertex.pos.x, vertex.pos.y, 1.0);
  var ray = normalize(uniforms.look_matrix * v);
  // var ray = v;

  var seed : u32 = uniforms.rand +
    u32((ray.x * ray.x) * 71337231.0 +
                       vertex.position.y * 7171.0 + vertex.position.x);

  // let LIMIT = 0.0015;
  // let LIMIT = 0.0015;

  // Matches 0.25/nm^3 from Griffiths page 153
  // let LIMIT = 3.7e-05;
  let LIMIT = uniforms.surf_limit;

  let N = 512;
  let MAX_DIST = 2.0 * length(pos);
  let DX = MAX_DIST / f32(N);
  let dpos = DX * ray;
  var dist = 0.0;
  var prob : f32 = 0.0;
  for (var i : i32 = 0; i < N; i++) {
    let h = hydrogen(pos);

    if h > LIMIT {
      // dist /= MAX_DIST;
      // return vec4(dist, dist, dist, 1.0);
       let n = normal(pos, dpos, h);
       // return vec4(vec3(0.5, 0.5, 0.5) + 0.5*n, 1.0);
       // return vec4(n, 1.0);
       return lighting(pos, n, light_pos);
    }

    prob += h * DX;
    /*
    seed = hash(seed);
    if h * DX > 0.01 * get_rand(seed) {
      return vec4(1000.0 * h, 1.0, 1.0, 1.0);
    }
    */

    pos += dpos;
    dist += DX;
  }
  // return vec4(vec3(0.5, 0.5, 0.5) - 0.5*ray, 1.0);
  // return vec4(0.5, 0.2, 0.7, 1.0);
  prob *= 100.0;
  return vec4(prob, prob, prob, 1.0);
  // return vec4(0.0, 0.0, 0.0, 0.0);
}
