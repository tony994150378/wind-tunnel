/*
 * LBM.js — 二维格子玻尔兹曼风洞求解器 (D2Q9, BGK)
 * 纯 JS，无依赖。同时支持浏览器 <script> 与 Node (module.exports)。
 * 物理：求解不可压 Navier-Stokes 的低马赫数近似，用于真实绕流仿真。
 */
(function (global) {
  'use strict';

  // D2Q9 离散速度方向与权重
  const CX = [0, 1, 0, -1, 0, 1, -1, -1, 1];
  const CY = [0, 0, 1, 0, -1, 1, 1, -1, -1];
  const W  = [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36];
  const OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6];

  class WindTunnel {
    /**
     * @param {number} w 网格宽（格子数）
     * @param {number} h 网格高（格子数）
     * @param {object} opt { u0, re, rho0 }
     */
    constructor(w, h, opt = {}) {
      this.w = w;
      this.h = h;
      this.N = w * h;
      this.u0 = opt.u0 != null ? opt.u0 : 0.1;     // 入口速度（格子单位，建议<=0.12 保证稳定）
      this.rho0 = opt.rho0 != null ? opt.rho0 : 1.0;
      this.re = opt.re != null ? opt.re : 100;      // 目标雷诺数
      this.omega = 1.0; // 由 setRe 计算

      // 分布函数（9 个方向，按 [dir*N + y*w + x] 或 [y*w+x]*9 存储）
      // 采用 SoA：f[dir] 为 Float64Array(N)
      this.f = [];
      this.ftmp = [];
      for (let i = 0; i < 9; i++) {
        this.f.push(new Float64Array(this.N));
        this.ftmp.push(new Float64Array(this.N));
      }
      this.rho = new Float64Array(this.N);
      this.ux = new Float64Array(this.N);
      this.uy = new Float64Array(this.N);
      this.curl = new Float64Array(this.N); // 涡量（用于可视化）
      this.mask = new Uint8Array(this.N);   // 1=固体(障碍物)

      this.stepCount = 0;
      this.forceX = 0;
      this.forceY = 0;
      this.forceXsmooth = 0;
      this.forceYsmooth = 0;
      this.cd = 0;
      this.cl = 0;
      this.charLength = 1; // 特征长度（投影高度），算力系数用
      this.les = false;    // Smagorinsky 大涡模拟（LES）开关
      this.cs = 0.11;      // Smagorinsky 常数（典型 0.1~0.2）

      this._initFields();
      this.setRe(this.re);
    }

    _initFields() {
      const { w, h, u0, rho0 } = this;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = y * w + x;
          const u = u0, v = 0;
          const u2 = u * u + v * v;
          for (let i = 0; i < 9; i++) {
            const cu = CX[i] * u + CY[i] * v;
            this.f[i][idx] = W[i] * rho0 * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u2);
          }
          this.rho[idx] = rho0;
          this.ux[idx] = u;
          this.uy[idx] = 0;
        }
      }
    }

    /** 根据目标雷诺数设置松弛频率（charLength 默认取当前障碍物投影高度） */
    setRe(re) {
      this.re = re;
      const nu = (this.u0 * this.charLength) / re; // 运动粘度
      let tau = 3 * nu + 0.5;
      if (tau < 0.51) tau = 0.51; // 稳定性下限
      this.tau = tau;
      this.nu = (tau - 0.5) / 3;
      this.omega = 1 / tau;
    }

    setInflow(u0) {
      this.u0 = u0;
      this.setRe(this.re);
    }

    /** 开关 Smagorinsky 大涡模拟（LES）：强剪切区自动加湍流粘性，高 Re 尾流更真实 */
    setTurb(on){ this.les = !!on; }

    /** 用 Uint8Array 掩膜设置障碍物（1=固体）。
     *  特征长度默认取沿来流方向(x)的投影长度：圆柱=直径、方块=边长、翼型=弦长。
     *  传入 charLength 时可锁定参考长度（如旋转攻角时保持弦长不变，得到标准气动系数）。 */
    setObstacle(mask, charLength) {
      this.mask = mask;
      if (charLength != null) {
        this.charLength = Math.max(1, charLength);
      } else {
        let minX = this.w, maxX = -1;
        for (let x = 0; x < this.w; x++) {
          for (let y = 0; y < this.h; y++) {
            if (mask[y * this.w + x]) { if (x < minX) minX = x; if (x > maxX) maxX = x; break; }
          }
        }
        this.charLength = Math.max(1, maxX - minX + 1);
      }
      this.setRe(this.re);
    }

    /** 生成内置示例形状掩膜 */
    static shapeMask(w, h, kind, opts = {}) {
      const mask = new Uint8Array(w * h);
      const cx = w * 0.3, cyc = h / 2;
      const inPoly = (px, py, pts) => {
        let ins = false;
        for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
          const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
          if (((yi > py) !== (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) ins = !ins;
        }
        return ins;
      };
      const fillPoly = (pts) => {
        let mnx = Infinity, mxx = -Infinity, mny = Infinity, mxy = -Infinity;
        for (const [a, b] of pts) { if (a < mnx) mnx = a; if (a > mxx) mxx = a; if (b < mny) mny = b; if (b > mxy) mxy = b; }
        const X0 = Math.max(0, Math.floor(mnx)), X1 = Math.min(w - 1, Math.ceil(mxx));
        const Y0 = Math.max(0, Math.floor(mny)), Y1 = Math.min(h - 1, Math.ceil(mxy));
        for (let y = Y0; y <= Y1; y++)
          for (let x = X0; x <= X1; x++)
            if (inPoly(x + 0.5, y + 0.5, pts)) mask[y * w + x] = 1;
      };
      if (kind === 'cylinder') {
        const r = opts.r || Math.min(w, h) * 0.12;
        for (let y = 0; y < h; y++)
          for (let x = 0; x < w; x++) {
            const dx = x - cx, dy = y - cyc;
            if (dx * dx + dy * dy <= r * r) mask[y * w + x] = 1;
          }
      } else if (kind === 'square') {
        const s = (opts.s || Math.min(w, h) * 0.2);
        for (let y = 0; y < h; y++)
          for (let x = 0; x < w; x++) {
            if (Math.abs(x - cx) <= s / 2 && Math.abs(y - cyc) <= s / 2) mask[y * w + x] = 1;
          }
      } else if (kind === 'airfoil') {
        // 简化对称翼型（NACA 00xx 近似），弦长沿 x
        const chord = opts.chord || Math.min(w, h) * 0.5;
        const th = opts.thick || 0.16;
        const x0 = w * 0.18;
        for (let y = 0; y < h; y++)
          for (let x = 0; x < w; x++) {
            const t = (x - x0) / chord;
            if (t < 0 || t > 1) continue;
            const yt = 5 * th * chord * (0.2969 * Math.sqrt(t) - 0.1260 * t - 0.3516 * t * t + 0.2843 * t * t * t - 0.1015 * t * t * t * t);
            if (Math.abs(y - cyc) <= yt) mask[y * w + x] = 1;
          }
      } else if (kind === 'sphere') {
        // 球体正交投影 = 圆，与 cylinder 同形（2D 截面下等价）
        const r = opts.r || Math.min(w, h) * 0.18;
        for (let y = 0; y < h; y++)
          for (let x = 0; x < w; x++) {
            const dx = x - cx, dy = y - cyc;
            if (dx * dx + dy * dy <= r * r) mask[y * w + x] = 1;
          }
      } else if (kind === 'car') {
        // F1 赛车侧视轮廓：车头（尖鼻 + 前翼，左）迎风，车尾（高尾翼，右）。
        // 来流固定左→右，车头朝左，直观对应真实风洞“从前方吹”。
        const yc = h / 2, Hh = Math.min(w, h) * 0.18;
        const x0 = w * 0.15, L = w * 0.54, x1 = x0 + L;
        const floor = yc + Hh * 0.5;
        // 主体楔形：低尖鼻(左) → 座舱/进气(中高) → 低车尾(右)
        fillPoly([
          [x0, floor],
          [x0, yc - Hh * 0.10],
          [x0 + 0.10 * L, yc - Hh * 0.40],
          [x0 + 0.28 * L, yc - Hh * 0.82],
          [x0 + 0.40 * L, yc - Hh * 0.92],
          [x0 + 0.56 * L, yc - Hh * 0.52],
          [x0 + 0.82 * L, yc - Hh * 0.22],
          [x1, yc - Hh * 0.08],
          [x1, floor],
        ]);
        // 前翼（车头，向左下突出）
        fillPoly([
          [x0 - 0.05 * L, floor],
          [x0 + 0.09 * L, floor],
          [x0 + 0.09 * L, floor + Hh * 0.30],
          [x0 - 0.05 * L, floor + Hh * 0.30],
        ]);
        // 尾翼（车尾，右侧高耸，底部与车体相连）
        fillPoly([
          [x1 - 0.05 * L, yc],
          [x1 + 0.02 * L, yc],
          [x1 + 0.02 * L, yc - Hh * 1.45],
          [x1 - 0.05 * L, yc - Hh * 1.45],
        ]);
        // 尾翼端板
        fillPoly([
          [x1 - 0.07 * L, yc - Hh * 1.55],
          [x1 + 0.04 * L, yc - Hh * 1.55],
          [x1 + 0.04 * L, yc - Hh * 1.40],
          [x1 - 0.07 * L, yc - Hh * 1.40],
        ]);
      } else if (kind === 'car_top') {
        // 俯视：来流仍从左→右（车头朝左），截面为车顶平面（前圆后尖的梯形）。
        // 与侧视共用同一“车头迎风”约定，只是观察的是车顶这个面。
        const L = opts.L || Math.min(w, h) * 0.52;
        const bw = opts.bw || Math.min(w, h) * 0.22;  // 半宽（车宽方向）
        const x0 = w * 0.20, yc = h / 2;
        const slantLen = L * 0.34, tailStart = x0 + L - slantLen;
        for (let x = 0; x < w; x++) {
          for (let y = 0; y < h; y++) {
            if (x < x0 || x > x0 + L) continue;
            let half = bw;
            if (x < x0 + L * 0.08) { const t = (x - x0) / (L * 0.08); half = bw * Math.sqrt(Math.max(0, t)); }
            if (x > tailStart) { const tt = (x - tailStart) / slantLen; half = bw * (1 - 0.5 * tt); }
            if (x > x0 + L * 0.90) half = bw * 1.5;   // 车尾尾翼在俯视更宽
            if (Math.abs(y - yc) <= half) mask[y * w + x] = 1;
          }
        }
      } else if (kind === 'car_front') {
        // 正视：来流从左→右直撞车头迎风面（竖向矩形，高=车宽方向）。
        // 这是真实风洞里“从正前方吹”的视角，注意来流方向仍固定左→右。
        const T = opts.T || Math.min(w, h) * 0.16;    // 车头厚度（x 方向）
        const bw = opts.bw || Math.min(w, h) * 0.26;  // 半高（车宽方向）
        const x0 = w * 0.40, yc = h / 2;
        for (let x = 0; x < w; x++) {
          for (let y = 0; y < h; y++) {
            if (x < x0 || x > x0 + T) continue;
            let half = bw;
            if (x < x0 + T * 0.3) { const t = (x - x0) / (T * 0.3); half = bw * Math.min(1, Math.sqrt(Math.max(0, t))); }
            if (Math.abs(y - yc) <= half) mask[y * w + x] = 1;
          }
        }
      }
      return mask;
    }

    /** 宏观量 + 碰撞（BGK），并返回碰撞后的分布用于动量交换 */
    _collide() {
      const { w, h, N, omega, u0, rho0 } = this;
      const f = this.f;
      // 先算宏观量
      for (let idx = 0; idx < N; idx++) {
        if (this.mask[idx]) continue;
        let r = 0, mx = 0, my = 0;
        for (let i = 0; i < 9; i++) { const fi = f[i][idx]; r += fi; mx += CX[i] * fi; my += CY[i] * fi; }
        this.rho[idx] = r;
        const ux = mx / r, uy = my / r;
        this.ux[idx] = ux; this.uy[idx] = uy;
      }
      // 碰撞（可选 Smagorinsky 大涡模拟：强剪切区加湍流粘性，高 Re 尾流更丰富/更稳）
      const cs2 = 1/3, tau0 = this.tau, omega0 = this.omega, les = this.les, Cs2 = this.cs * this.cs;
      for (let idx = 0; idx < N; idx++) {
        if (this.mask[idx]) continue;
        const r = this.rho[idx], ux = this.ux[idx], uy = this.uy[idx];
        const u2 = ux * ux + uy * uy;
        let omega = omega0;
        if (les) {
          // 应变率张量 S_ab = -(1/(2·rho·cs²·tau)) · Pi_ab，Pi_ab = Σ e_ia·e_ib·f_i^neq
          let Pxx = 0, Pxy = 0, Pyy = 0;
          for (let i = 0; i < 9; i++) {
            const cu = CX[i]*ux + CY[i]*uy;
            const feq = W[i] * r * (1 + 3*cu + 4.5*cu*cu - 1.5*u2);
            const fn = f[i][idx] - feq;
            Pxx += CX[i]*CX[i]*fn; Pxy += CX[i]*CY[i]*fn; Pyy += CY[i]*CY[i]*fn;
          }
          const k = -1/(2*r*cs2*tau0);
          const Sxx = k*Pxx, Sxy = k*Pxy, Syy = k*Pyy;
          const S2 = 2*(Sxx*Sxx + 2*Sxy*Sxy + Syy*Syy);
          const nut = Cs2 * Math.sqrt(Math.max(0, S2));   // 格子尺寸 Δ=1
          let tauE = 3*(this.nu + nut) + 0.5;
          if (tauE < 0.51) tauE = 0.51;                   // 稳定性下限
          omega = 1/tauE;
        }
        for (let i = 0; i < 9; i++) {
          const cu = CX[i] * ux + CY[i] * uy;
          const feq = W[i] * r * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u2);
          f[i][idx] = f[i][idx] * (1 - omega) + omega * feq;
        }
      }
    }

    /** 流 + 反弹边界（半步反弹）。固体节点不装配分布，但写入反弹量供下一步算力 */
    _stream() {
      const { w, h, N, mask, f, ftmp } = this;
      for (let i = 0; i < 9; i++) ftmp[i].set(f[i]);
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = y * w + x;
          if (mask[idx]) continue; // 固体节点不装配分布
          for (let i = 0; i < 9; i++) {
            const xs = x - CX[i];
            const ys = y - CY[i];
            let val;
            if (xs < 0 || xs >= w || ys < 0 || ys >= h) {
              // 域外：留给 _applyBC，这里先填自由来流平衡态
              const cu = CX[i] * this.u0;
              val = W[i] * this.rho0 * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * (this.u0 * this.u0));
            } else {
              const sidx = ys * w + xs;
              if (mask[sidx]) {
                // 反弹：来自固体的入射分布 = 当前节点相反方向碰撞后分布
                val = ftmp[OPP[i]][idx];
              } else {
                val = ftmp[i][sidx];
              }
            }
            f[i][idx] = val;
          }
        }
      }
    }

    /**
     * 动量交换法 (Momentum Exchange Method) 算力 —— 局部、对边界处理不敏感。
     * 用非平衡分布 f_i^{neq} = f_i^{post} - f_i^{eq}（均衡部分在净动量中自相消）。
     * F = Σ_{流体节点x, 方向i指向固体} e_i · f_i^{neq}(x)
     */
    _computeForce() {
      const { w, h, mask, f, rho, ux, uy } = this;
      let Fx = 0, Fy = 0;
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const idx = y * w + x;
          if (mask[idx]) continue; // 仅流体节点
          const r = rho[idx], vx = ux[idx], vy = uy[idx];
          const u2 = vx * vx + vy * vy;
          for (let i = 0; i < 9; i++) {
            const xs = x + CX[i];
            const ys = y + CY[i];
            if (xs < 0 || xs >= w || ys < 0 || ys >= h) continue;
            const sidx = ys * w + xs;
            if (mask[sidx]) { // 邻居是固体 → 边界链
              const cu = CX[i] * vx + CY[i] * vy;
              const feq = W[i] * r * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u2);
              const fneq = f[i][idx] - feq; // 非平衡分布
              // 反弹使净动量交换 = 2·e_i·f_i^{neq}
              Fx += 2 * CX[i] * fneq;
              Fy += 2 * CY[i] * fneq;
            }
          }
        }
      }
      this.forceX = Fx;
      this.forceY = Fy;
    }

    /** 入口/出口/上下边界条件 */
    _applyBC() {
      const { w, h, u0, rho0 } = this;
      const f = this.f;
      const setEq = (idx, u, v) => {
        const r = rho0, u2 = u * u + v * v;
        for (let i = 0; i < 9; i++) {
          const cu = CX[i] * u + CY[i] * v;
          f[i][idx] = W[i] * r * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u2);
        }
      };
      // 入口列 x=0
      for (let y = 0; y < h; y++) setEq(y * w + 0, u0, 0);
      // 出口列 x=w-1：零梯度（复制前一列）
      for (let y = 0; y < h; y++) {
        const dst = y * w + (w - 1);
        const src = y * w + (w - 2);
        for (let i = 0; i < 9; i++) f[i][dst] = f[i][src];
      }
      // 上下边界 y=0 / y=h-1：自由来流平衡态（近似远场）
      for (let x = 0; x < w; x++) {
        setEq(0 * w + x, u0, 0);
        setEq((h - 1) * w + x, u0, 0);
      }
    }

    /** 出口缓冲（sponge zone）：最右 15% 网格把流场平滑松弛回自由来流，
     *  吸收下游扰动，消除开放边界的数值反射（避免流场右端出现“气流反弹”伪影）。 */
    _sponge() {
      const { w, h, rho0, u0, mask, f } = this;
      const x0 = Math.floor(w * 0.85);
      if (x0 >= w - 2) return;
      for (let x = x0; x < w; x++) {
        const k = 0.06 * (x - x0) / (w - 1 - x0); // 越靠右吸收越强
        for (let y = 0; y < h; y++) {
          const idx = y * w + x;
          if (mask[idx]) continue;
          let ux = this.ux[idx], uy = this.uy[idx], r = this.rho[idx];
          ux += k * (u0 - ux);
          uy += k * (0 - uy);
          r  += k * (rho0 - r);
          this.ux[idx] = ux; this.uy[idx] = uy; this.rho[idx] = r;
          const u2 = ux * ux + uy * uy;
          for (let i = 0; i < 9; i++) {
            const cu = CX[i] * ux + CY[i] * uy;
            f[i][idx] = W[i] * r * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * u2);
          }
        }
      }
    }

    /** 计算涡量（可视化用） */
    _computeCurl() {
      const { w, h, ux, uy, curl } = this;
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const idx = y * w + x;
          const dvx = (ux[idx + 1] - ux[idx - 1]) * 0.5;
          const duy = (uy[idx + w] - uy[idx - w]) * 0.5;
          curl[idx] = duy - dvx;
        }
      }
    }

    /** 推进一个时间步 */
    step() {
      this._collide();
      this._computeForce();   // 读碰撞后流体分布 + 上一步固体节点存的反弹量
      this._stream();         // 更新流体入射分布，并把反弹量写入固体节点供下一步
      this._applyBC();
      this._sponge();        // 出口缓冲：吸收下游反射，避免流场右端“气流反弹”伪影
      this._computeCurl();
      this.stepCount++;

      // 平滑力（指数滑动平均）用于更稳定的系数显示
      const a = 0.05;
      this.forceXsmooth = this.forceXsmooth * (1 - a) + this.forceX * a;
      this.forceYsmooth = this.forceYsmooth * (1 - a) + this.forceY * a;

      const denom = 0.5 * this.rho0 * this.u0 * this.u0 * this.charLength;
      // 阻力 = 流体对物体的下游推力（此处 forceX<0 → 取负得正阻力系数）
      this.cd = denom > 0 ? (-this.forceXsmooth) / denom : 0;
      this.cl = denom > 0 ? this.forceYsmooth / denom : 0;
    }

    /** 连续推进 n 步 */
    run(n) {
      for (let k = 0; k < n; k++) this.step();
    }
  }

  const api = { WindTunnel, CX, CY, W, OPP };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  global.LBM = api;
})(typeof window !== 'undefined' ? window : globalThis);
