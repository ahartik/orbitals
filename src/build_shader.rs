use std::fmt::Write;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderParams {
    pub n: i32,
    pub l: i32,
    pub m: i32,
    pub cut_half: bool,
    pub real_orbital: bool,
}

static TEMPLATE: &str = include_str!("shader_template.wgsl");

pub struct ShaderBuilder {
    max_n: usize,
    legendre: Vec<Polynomial>,
    laguerre: Vec<Polynomial>,
}

impl ShaderBuilder {
    pub fn new(max_n: usize) -> Self {
        Self {
            max_n,
            legendre: gen_legendre(max_n),
            laguerre: gen_laguerre(2 * max_n),
        }
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

    pub fn build(&self, params: &ShaderParams) -> String {
        let n = params.n;
        let l = params.l;
        let m = params.m;
        assert!(n <= self.max_n as i32);
        assert!(l < n);
        assert!(m.abs() <= l);

        let mut ctx = minipre::Context::new();
        ctx.define("VAL_N", format!("{:.1}", n));
        ctx.define("VAL_L", format!("{:.1}", l));
        ctx.define("VAL_M", format!("{:.1}", m));
        ctx.define("CUT_HALF", if params.cut_half { "1" } else { "0" });
        ctx.define("PHI_SYMMETRIC", if params.real_orbital {"0"} else {"1"});

        let mut wave = String::new();
        // rfun
        {
            let p = 2 * l + 1;
            let q = n - l - 1;
            let mut rfun = self.assoc_laguerre(q, p);
            for i in 0..rfun.coeffs.len() {
                rfun.coeffs[i] *= (2.0 / (n as f64)).powi(i as i32);
            }
            let rmul = f64::sqrt(
                (2.0 / (n as f64)).powi(3) * factorial((n - l - 1) as usize)
                    / (2.0 * (n as f64) * factorial((n + l) as usize)),
            );
            rfun.scale_inplace(rmul);
            rfun = rfun.mul(&Polynomial::single_term(
                (2.0 / (n as f64)).powi(l),
                l as usize,
            ));

            let ncoeffs = rfun.coeffs.len();
            let nvecs = (ncoeffs + 3) / 4;
            writeln!(wave, "let rpows0 = vec4(1.0, r, r * r, r * r * r);").unwrap();
            for i in 1..nvecs {
                writeln!(wave, "let rpows{} = r *rpows{}.w * rpows0;", i, i-1).unwrap();
            }
            writeln!(wave, "var rfun : f32 = 0.0;").unwrap();
            for i in 0..nvecs {
                let mut coeffs: [f64; 4] = [0.0; 4];
                for j in 0..4 {
                    let p = 4 * i + j;
                    if p < ncoeffs {
                        coeffs[j] = rfun.coeffs[p];
                    }
                }
                writeln!(
                    wave,
                    "let rcoeff{} = vec4({:.8e}, {:.8e}, {:.8e}, {:.8e});",
                    i, coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                ).unwrap();
                writeln!(wave, "rfun += dot(rpows0, pow(r, {:.1})* rcoeff{});", (4*i) as f64, i).unwrap();
            }
        }
        writeln!(wave, "rfun *= exp(-r / {:.1});", n as f64).unwrap();
        // yfun
        {
            // Coefficients for Y(theta, phi):
            let mut yfun = self.assoc_legendre(l, m.abs());
            yfun.scale_inplace(f64::sqrt(
                ((2 * l + 1) as f64 / (4.0 * std::f64::consts::PI))
                    * (factorial((l - m) as usize) / factorial((l + m) as usize)),
            ));
            let ncoeffs = yfun.coeffs.len();
            writeln!(wave, "var yfun : f32 = 0.0;").unwrap();
            /*
            for i in 0..ncoeffs {
                if yfun.coeffs[i] != 0.0 {
                    writeln!(wave, "yfun += pow(costheta, {:.1}) * {:.8};",
                    i as f64, yfun.coeffs[i]).unwrap();
                }
            }
            */
            writeln!(wave,
                "let cospows0 : vec4<f32>= vec4(1.0, costheta, costheta * costheta, costheta * costheta * costheta);").unwrap();
            let nvecs = (ncoeffs + 3) / 4;
            for i in 1..nvecs {
                writeln!(
                    wave,
                    "let cospows{} = costheta * cospows{}.w * cospows0;",
                    i, i-1
                )
                .unwrap();
            }
            for i in 0..nvecs {
                let mut coeffs: [f64; 4] = [0.0; 4];
                for j in 0..4 {
                    let p = 4 * i + j;
                    if p < ncoeffs {
                        coeffs[j] = yfun.coeffs[p];
                    }
                }

                writeln!(
                    wave,
                    "let ycoeff{} : vec4<f32> = vec4({:.8e}, {:.8e}, {:.8e}, {:.8e});",
                    i, coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                )
                .unwrap();
                writeln!(wave, "yfun += dot(cospows{}, ycoeff{});", i, i).unwrap();
            }
            for _ in 0..m {
                writeln!(wave, "yfun *= sintheta;").unwrap();
            }
            if params.real_orbital && m != 0 {
                writeln!(wave, "let phiz = vec2(pos.x, pos.y) / xylen;").unwrap();
                writeln!(wave, "let z0 = phiz;").unwrap();
                let mut i = 1usize;
                while (1 << i) <= m {
                    writeln!(wave, "let z{} = complex_mul(z{}, z{});", i, i-1, i-1).unwrap();
                    i += 1;
                }

                writeln!(wave, "var zf = vec2(1.0, 0.0);").unwrap();
                for j in 0..i {
                    if ((1 << j) & m) != 0 {
                        writeln!(wave, "zf = complex_mul(zf, z{});",j ).unwrap();
                    }
                }
                writeln!(wave, "yfun *= zf.x;").unwrap();
                /*
                writeln!(wave, "let cosphi = pos.x / xylen;").unwrap();
                writeln!(wave, "let sinphi = pos.y / xylen;").unwrap();
                writeln!(wave, "var cosa = cosphi;").unwrap();
                writeln!(wave, "var sina = sinphi;").unwrap();
                writeln!(wave, "var tmp = 0.0;").unwrap();
                for _ in 1..m {
                    writeln!(wave, "tmp = cosa * cosphi - sina * sinphi;").unwrap();
                    writeln!(wave, "sina = sina * cosphi + cosa * sinphi;").unwrap();
                    writeln!(wave, "cosa = tmp;").unwrap();
                }
                writeln!(wave, "yfun *= cosa;").unwrap();
                */
                /*
                writeln!(wave, "let phi = atan2(pos.y, pos.x);").unwrap();
                writeln!(wave, "yfun *= cos({:.1} * phi);", m as f64).unwrap();
                */
            }
        }

        writeln!(wave, "return (yfun * rfun);").unwrap();
        println!("{}", wave);
        // if m == 5 && l == 6 {
        //     println!("{}", wave);
        // }

        ctx.define("WAVE_FUNC", wave);
        return minipre::process_str(TEMPLATE, &mut ctx).unwrap();
    }

    pub fn all_params(&self) -> Vec<ShaderParams> {
        let mut res = vec![];
        for n in 1i32..((self.max_n + 1) as i32) {
            for l in 0i32..n {
                // for m in (-l)..(l + 1) {
                for m in 0..(l + 1) {
                    res.push(ShaderParams {
                        n,
                        l,
                        m,
                        cut_half: false,
                        real_orbital: true,
                    });
                    res.push(ShaderParams {
                        n,
                        l,
                        m,
                        cut_half: true,
                        real_orbital: true,
                    });
                }
            }
        }
        return res;
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
