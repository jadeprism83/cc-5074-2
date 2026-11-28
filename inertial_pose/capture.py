"""角色捕捉视窗（图形入口）。

这是惯性姿态模块的**主要使用入口**：用标准库 ``http.server`` 提供一个
浏览器视窗（无需 Tk/显示器/第三方前端依赖），把完整的
``观测 -> 标定 -> 层级求解 -> 角色姿态`` 链路跑在真实计算上，并在同一
页面展示：

- 可切换的**观测轴向**（仅允许行列式 +1 的正当轴置换；非法反射由后端
  显式报错并在视窗异常区呈现，而不是静默退化）；
- 合成动作的**回放**（播放/暂停、拖动、循环、速度、视角）；
- **结果区**：3D 骨架姿态与每个骨段的**惯性偏差**（求解朝向相对惯性
  观测的真实角度残差）、足滑、航向仲裁事件与汇总指标。

命令行（:mod:`inertial_pose.cli`）只做批处理辅助，不承担该交互入口。

启动::

    python3 -m inertial_pose.capture            # 默认 127.0.0.1:8000
    python3 -m inertial_pose.capture --port 8123
"""

from __future__ import annotations

import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from . import quat
from .calibration import Calibration
from .sensors import sequence_from_orientations
from .skeleton import DEFAULT_SENSOR_PLACEMENT, FOOT_SEGMENTS
from .solver import PoseSolver, SolverConfig
from .synthetic import apply_online_yaw_drift, generate_motion, make_calibration


# 传感器换绑只允许正当旋转：(0,1,2) 的行列式 +1 置换（恒等 + 两个循环置换）。
# 行列式 -1 的镜像反射物理上不构成旋转，后端会显式拒绝，这里也不提供。
LEGAL_REMAPS: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (1, 2, 0),
    (2, 0, 1),
)

ACTIONS = ("walk", "squat", "arm_raise")


# --------------------------------------------------------------------------- #
# 计算后端：一次请求解完整段序列，返回可直接驱动视窗的 JSON 数据
# --------------------------------------------------------------------------- #
def build_capture_payload(params: dict) -> dict:
    """按视窗参数生成数据、标定并逐帧求解，返回可 JSON 序列化的负载。

    任何参数/链路错误都以普通异常向上抛，由 HTTP 层捕获后转成
    ``{"ok": false, "error": ...}``，在视窗异常区显式呈现。

    求解器的唯一动作输入始终是合成传感器观测 ``q_WS``；真值骨段旋转
    只进入标定（已知标定姿态）与最终指标，绝不绕过标定驱动角色。
    """
    action = str(params.get("action", "walk"))
    if action not in ACTIONS:
        raise ValueError(f"未知动作 {action!r}，可选 {list(ACTIONS)}")
    duration = _bounded_float(params.get("duration", 6.0), 1.0, 30.0, "时长")
    dt = _bounded_float(params.get("dt", 0.05), 0.01, 0.2, "采样周期")
    seed = int(params.get("seed", 0))
    noise = _bounded_float(params.get("noise_deg", 0.0), 0.0, 20.0, "观测噪声")
    drift = _bounded_float(params.get("drift_rate", 0.0), 0.0, 1.0, "航向漂移率")
    mount_tilt = _bounded_float(
        params.get("mount_tilt_deg", 15.0), 0.0, 45.0, "安装偏航幅度")

    # 1) 已知动作 + 随机安装偏差 + 世界航向 -> 传感器观测
    rng = np.random.default_rng(seed)
    mounts = _random_mounts(rng, mount_tilt)
    motion = generate_motion(
        action=action,
        duration=duration,
        dt=dt,
        mount_q_BS=mounts,
        world_yaw={sid: 0.25 for sid in mounts},
        noise_deg=noise,
        seed=seed,
    )

    # 2) 显式标定（安装旋转与世界参考分开估计），使用无漂移观测
    cal = make_calibration(motion)

    # 3) 运行期可选航向漂移（用于触发足锁-航向冲突仲裁）
    observed = (
        apply_online_yaw_drift(motion, drift) if drift > 0.0
        else motion.sensor_q_WS
    )

    # 4) 观测轴向开关：逐传感器正当轴置换（字符串键 -> 整数传感器号）
    remap = _parse_remap(params.get("axis_remap"), cal)

    config = SolverConfig(
        slide_weight=_bounded_float(
            params.get("slide_weight", SolverConfig.slide_weight),
            0.0, 10.0, "足滑权重"),
        jump_weight=_bounded_float(
            params.get("jump_weight", SolverConfig.jump_weight),
            0.0, 10.0, "航向跃变权重"),
    )

    # 5) 逐帧求解；非法轴向等错误在此抛出，由视窗异常区呈现
    solver = PoseSolver(
        cal, skeleton=motion.skeleton,
        config=config, observed_axis_remap=remap,
    )
    results = solver.solve_sequence(
        sequence_from_orientations(observed, motion.dt))

    skel = motion.skeleton
    frames = [_encode_frame(skel.order, r) for r in results]

    # 6) 汇总指标（真实计算，非占位）
    residual_matrix = np.array([
        [r.inertial_residual_deg.get(seg, 0.0) for seg in cal.segments]
        for r in results
    ])
    slide_total = {seg: 0.0 for seg in FOOT_SEGMENTS}
    for r in results:
        for seg, d in r.foot_slide_m.items():
            slide_total[seg] = slide_total.get(seg, 0.0) + d
    # 冲突记录带实际帧号，便于视窗在回放时间轴上定位
    conflicts = [
        {"frame": k, "t": r.t, **r.conflict}
        for k, r in enumerate(results) if r.conflict is not None
    ]

    # 真值对照（仅指标用）：关节角误差与根位置误差
    joint_err = {}
    for joint in skel.order:
        errs = np.degrees([
            quat.angular_distance(r.local_rot[joint],
                                  motion.truth_local_rot[joint][k])
            for k, r in enumerate(results)
        ])
        joint_err[joint] = {
            "mean": float(np.mean(errs)),
            "p95": float(np.percentile(errs, 95)),
        }
    root_err = np.linalg.norm(
        np.stack([r.root_pos for r in results])
        - motion.truth_root_pos[: len(results)], axis=1)

    return {
        "ok": True,
        "meta": {
            "action": action,
            "dt": motion.dt,
            "num_frames": len(results),
            "duration_s": float(motion.times[-1]) if len(motion.times) else 0.0,
            "seed": seed,
            "noise_deg": noise,
            "drift_rate": drift,
            "convention": "Z 向上，角色面向 +X，左侧为 +Y；矢状铰链绕局部 y",
        },
        "bones": skel.order,
        "segments": list(cal.segments),
        "placement": {seg: sid for seg, sid in motion.placement.items()},
        "frames": frames,
        "residual": {
            "segments": list(cal.segments),
            "mean_deg": [float(x) for x in residual_matrix.mean(axis=0)],
            "max_deg": [float(x) for x in residual_matrix.max(axis=0)],
        },
        "calibration": {
            "segments": list(cal.segments),
            "residual_rms_deg": [float(x) for x in cal.residual_deg],
            "world_yaw_deg": [float(np.degrees(y)) for y in cal.world_yaw],
        },
        "foot_slide_total_m": {k: float(v) for k, v in slide_total.items()},
        "conflicts": conflicts,
        "joint_error_deg": joint_err,
        "root_pos_rmse_m": float(np.sqrt(np.mean(root_err ** 2))),
        "warnings": sorted({w for r in results for w in r.warnings}),
    }


def _random_mounts(rng: np.random.Generator, max_tilt_deg: float):
    from .synthetic import random_mount_rotations
    return random_mount_rotations(DEFAULT_SENSOR_PLACEMENT, rng, max_tilt_deg)


def _encode_frame(order: tuple[str, ...], result) -> dict:
    """单帧求解结果 -> 纯 Python（位置/朝向/残差/足滑/仲裁）。"""
    positions = {j: [float(v) for v in result.world_pos[j]] for j in order}
    quats = {j: [float(v) for v in result.world_rot[j]] for j in order}
    return {
        "t": float(result.t),
        "root": [float(v) for v in result.root_pos],
        "positions": positions,
        "rotations": quats,
        "residual_deg": {
            seg: float(v) for seg, v in result.inertial_residual_deg.items()
        },
        "foot_slide_m": {
            seg: float(v) for seg, v in result.foot_slide_m.items()
        },
        "anchors": {
            seg: [float(v) for v in pos]
            for seg, pos in result.anchors.items()
        },
        "conflict": result.conflict,
    }


def _parse_remap(raw, cal: Calibration) -> dict[int, tuple[int, int, int]]:
    """解析视窗提交的轴向映射 {sensor_id_str: [a,b,c]}，做正当性校验。"""
    if not raw:
        return {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    out: dict[int, tuple[int, int, int]] = {}
    for key, value in raw.items():
        sid = int(key)
        if sid not in cal.sensor_ids:
            raise ValueError(f"传感器 {sid} 未参与标定，无法切换其轴向")
        remap = tuple(int(x) for x in value)
        if remap not in LEGAL_REMAPS:
            raise ValueError(
                f"传感器 {sid} 的轴向重排 {remap} 非法：只允许行列式 +1 的"
                f"正当置换 {LEGAL_REMAPS}；镜像反射物理上不构成旋转")
        out[sid] = remap  # type: ignore[assignment]
    return out


def _bounded_float(value, low: float, high: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not (low <= value <= high):
        raise ValueError(f"{label}必须落在 [{low}, {high}]，收到 {value}")
    return value


# --------------------------------------------------------------------------- #
# HTTP 层（标准库，无第三方依赖）
# --------------------------------------------------------------------------- #
_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>惯性姿态角色捕捉视窗</title>
<style>
  :root { --bd:#2a2f3a; --bg:#14171d; --panel:#1c2129; --fg:#d7dce4;
          --muted:#8b93a3; --accent:#5aa2ff; --ok:#4cc38a; --warn:#e0a93b;
          --err:#ef6a6a; }
  * { box-sizing:border-box; }
  body { margin:0; font:13px/1.5 -apple-system,"Segoe UI",Roboto,"PingFang SC",
         "Microsoft YaHei",sans-serif; background:var(--bg); color:var(--fg); }
  header { padding:10px 16px; border-bottom:1px solid var(--bd);
           display:flex; align-items:baseline; gap:12px; }
  header h1 { font-size:15px; margin:0; }
  header .sub { color:var(--muted); font-size:12px; }
  .layout { display:grid; grid-template-columns:300px 1fr 320px; gap:0;
            height:calc(100vh - 44px); }
  .panel { overflow:auto; padding:14px; border-right:1px solid var(--bd);
           background:var(--panel); }
  .panel:last-child { border-right:none; border-left:1px solid var(--bd); }
  .panel h2 { font-size:12px; text-transform:uppercase; letter-spacing:.08em;
              color:var(--muted); margin:16px 0 8px; }
  .panel h2:first-child { margin-top:0; }
  label { display:block; margin:8px 0 3px; color:var(--muted); font-size:12px; }
  input,select,button { width:100%; padding:6px 8px; background:#101318;
        border:1px solid var(--bd); border-radius:6px; color:var(--fg);
        font-size:13px; }
  button { cursor:pointer; margin-top:8px; }
  button.primary { background:var(--accent); color:#08111f; border-color:var(--accent);
        font-weight:600; }
  button:hover { filter:brightness(1.1); }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .remap-row { display:grid; grid-template-columns:90px 1fr; gap:6px;
        align-items:center; margin:4px 0; font-size:12px; }
  .remap-row select { padding:4px; }
  .stage { position:relative; background:#0f1216; overflow:hidden; }
  canvas { display:block; width:100%; height:100%; }
  .replay { position:absolute; left:12px; right:12px; bottom:12px;
        background:rgba(20,23,29,.92); border:1px solid var(--bd);
        border-radius:8px; padding:10px; }
  .replay .controls { display:flex; gap:8px; align-items:center; }
  .replay .controls button { width:auto; margin:0; padding:5px 12px; }
  .replay input[type=range] { flex:1; }
  .frameinfo { color:var(--muted); font-variant-numeric:tabular-nums;
        min-width:96px; text-align:right; font-size:12px; }
  .speed { display:flex; gap:6px; align-items:center; margin-top:8px;
        color:var(--muted); font-size:12px; }
  .speed select { width:auto; }
  .metric { display:flex; justify-content:space-between; padding:3px 0;
        border-bottom:1px dotted var(--bd); font-variant-numeric:tabular-nums; }
  .metric b { font-weight:600; }
  .bar { height:7px; border-radius:4px; background:#101318; overflow:hidden;
        margin:3px 0 8px; }
  .bar > i { display:block; height:100%; background:var(--accent); }
  .tag { display:inline-block; padding:1px 7px; border-radius:10px; font-size:11px; }
  .tag.ok { background:rgba(76,195,138,.18); color:var(--ok); }
  .tag.warn { background:rgba(224,169,59,.18); color:var(--warn); }
  .tag.err { background:rgba(239,106,106,.18); color:var(--err); }
  .conflict { border:1px solid var(--bd); border-radius:6px; padding:7px 8px;
        margin:6px 0; font-size:12px; }
  .conflict .ch { font-weight:600; }
  .ch.correct_heading { color:var(--ok); }
  .ch.release_anchor { color:var(--warn); }
  #errorbox { display:none; margin:10px 14px 0; padding:10px 12px;
        border:1px solid var(--err); border-radius:8px;
        background:rgba(239,106,106,.12); color:#ffb3b3; white-space:pre-wrap; }
  .muted { color:var(--muted); }
  .legend { display:flex; gap:14px; font-size:11px; color:var(--muted);
        padding:6px 4px; }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:2px;
        margin-right:4px; }
</style>
</head>
<body>
<header>
  <h1>惯性姿态角色捕捉视窗</h1>
  <span class="sub">观测 → 显式标定（安装/世界/骨架三框架分离）→ 层级求解 → 角色姿态</span>
</header>
<div id="errorbox"></div>
<div class="layout">
  <!-- 左：数据与观测轴向控制 -->
  <div class="panel">
    <h2>合成数据</h2>
    <label>已知动作</label>
    <select id="action">
      <option value="walk">步行 walk（含足锁）</option>
      <option value="squat">深蹲 squat</option>
      <option value="arm_raise">举臂 arm_raise（无锚点先验）</option>
    </select>
    <div class="row">
      <div><label>时长 (s)</label><input id="duration" type="number" value="6" min="1" max="30" step="0.5"></div>
      <div><label>随机种子</label><input id="seed" type="number" value="0" step="1"></div>
    </div>
    <div class="row">
      <div><label>观测噪声 (°)</label><input id="noise" type="number" value="0" min="0" max="20" step="0.5"></div>
      <div><label>安装偏幅 (°)</label><input id="mount" type="number" value="15" min="0" max="45" step="1"></div>
    </div>
    <label>运行期航向漂移 (rad/s，制造足锁冲突)</label>
    <input id="drift" type="number" value="0" min="0" max="1" step="0.01">

    <h2>冲突仲裁权重</h2>
    <div class="row">
      <div><label>足滑权重</label><input id="slide_w" type="number" value="0.10" min="0" max="10" step="0.01"></div>
      <div><label>航向跃变权重</label><input id="jump_w" type="number" value="1.0" min="0" max="10" step="0.05"></div>
    </div>

    <h2>观测轴向（传感器换绑）</h2>
    <p class="muted" style="font-size:11px;margin:4px 0">
      仅允许行列式 +1 的正当轴置换；反射（det −1）由后端显式拒绝并在上方报出。</p>
    <div id="remaps"></div>

    <button class="primary" id="run">生成并求解</button>
  </div>

  <!-- 中：3D 姿态舞台 + 回放 -->
  <div class="stage">
    <canvas id="view"></canvas>
    <div class="replay">
      <div class="controls">
        <button id="play">▶ 播放</button>
        <input id="scrub" type="range" min="0" max="0" value="0">
        <span class="frameinfo" id="frameinfo">0 / 0</span>
      </div>
      <div class="speed">
        <label style="margin:0">速度</label>
        <select id="rate">
          <option value="0.25">0.25×</option>
          <option value="0.5">0.5×</option>
          <option value="1" selected>1×</option>
          <option value="2">2×</option>
        </select>
        <label style="margin:0"><input type="checkbox" id="loop" checked style="width:auto"> 循环</label>
        <label style="margin:0"><input type="checkbox" id="autorot" style="width:auto"> 自动旋转视角</label>
        <span style="flex:1"></span>
        <span>视角偏航 <input type="range" id="viewyaw" min="-180" max="180" value="35" style="width:120px"></span>
      </div>
    </div>
  </div>

  <!-- 右：结果区（姿态指标 + 惯性偏差 + 仲裁） -->
  <div class="panel">
    <h2>运行结果</h2>
    <div id="summary"><span class="muted">尚未求解。</span></div>

    <h2>惯性偏差（求解朝向 vs 惯性观测，°）</h2>
    <div id="residual"><span class="muted">—</span></div>

    <h2>足锁与航向仲裁</h2>
    <div id="anchors"></div>
    <div id="conflicts"></div>

    <h2>标定核验</h2>
    <div id="calib"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);
let DATA = null, frame = 0, playing = false, lastTs = null, viewYaw = 35 * Math.PI/180;
let autoRotate = false;
const canvas = $("view"), ctx = canvas.getContext("2d");

function showError(msg){
  const box = $("errorbox");
  if(!msg){ box.style.display="none"; box.textContent=""; return; }
  box.style.display="block"; box.textContent = "⚠ " + msg;
}
function clearError(){ showError(null); }

async function runCapture(){
  clearError();
  const remap = {};
  document.querySelectorAll(".remap-sel").forEach(sel=>{
    const v = sel.value.split(",").map(Number);
    remap[sel.dataset.sid] = v;
  });
  const params = new URLSearchParams({
    action: $("action").value,
    duration: $("duration").value,
    dt: "0.05",
    seed: $("seed").value,
    noise_deg: $("noise").value,
    mount_tilt_deg: $("mount").value,
    drift_rate: $("drift").value,
    slide_weight: $("slide_w").value,
    jump_weight: $("jump_w").value,
    axis_remap: JSON.stringify(remap),
  });
  try{
    const resp = await fetch("/api/solve?" + params.toString());
    const payload = await resp.json();
    if(!payload.ok){ showError(payload.error || "求解失败"); return; }
    DATA = payload; frame = 0;
    $("scrub").max = DATA.meta.num_frames - 1;
    $("scrub").value = 0;
    buildRemapSelectors(payload);
    renderSummary(payload); renderResidual(payload);
    renderAnchors(payload); renderCalib(payload);
  }catch(e){ showError(String(e)); }
}

function buildRemapSelectors(data){
  const box = $("remaps"); box.innerHTML = "";
  const LEGAL = [[0,1,2],[1,2,0],[2,0,1]];
  for(const [seg, sid] of Object.entries(data.placement)){
    const row = document.createElement("div"); row.className="remap-row";
    const lab = document.createElement("span"); lab.textContent = seg;
    const sel = document.createElement("select");
    sel.className="remap-sel"; sel.dataset.sid = sid;
    LEGAL.forEach(p=>{
      const o=document.createElement("option");
      o.value=p.join(","); o.textContent="("+p.join(",")+")";
      sel.appendChild(o);
    });
    row.append(lab, sel); box.appendChild(row);
  }
}

function fmt(x, n=3){ return (x===undefined||x===null) ? "—" : Number(x).toFixed(n); }
function tagFor(v, good, warn){
  const cls = v <= good ? "ok" : (v <= warn ? "warn" : "err");
  return '<span class="tag '+cls+'">'+fmt(v)+'°</span>';
}
function barFor(v, maxV){
  const pct = Math.max(0, Math.min(100, 100 * v / Math.max(maxV, 1e-6)));
  const col = v <= 3 ? "var(--ok)" : (v <= 8 ? "var(--warn)" : "var(--err)");
  return '<div class="bar"><i style="width:'+pct+'%;background:'+col+'"></i></div>';
}

function renderSummary(d){
  const je = d.joint_error_deg;
  const means = Object.values(je).map(x=>x.mean), p95s = Object.values(je).map(x=>x.p95);
  const slide = Object.values(d.foot_slide_total_m).reduce((a,b)=>a+b,0);
  $("summary").innerHTML =
    metric("动作", d.meta.action) +
    metric("帧数 / 时长", d.meta.num_frames + " / " + fmt(d.meta.duration_s,2)+"s") +
    metric("关节角误差 均值", fmt(means.reduce((a,b)=>a+b,0)/means.length,3)+"°") +
    metric("关节角误差 p95 最大", fmt(Math.max(...p95s),3)+"°") +
    metric("根位置 RMSE", fmt(d.root_pos_rmse_m,4)+" m") +
    metric("累计足滑（双脚）", fmt(slide,5)+" m") +
    metric("航向冲突事件", d.conflicts.length + " 次") +
    (d.warnings.length ? metric("先验告警", d.warnings.join("；")) : "");
}
function metric(k,v){ return '<div class="metric"><span class="muted">'+k+
       '</span><b>'+v+'</b></div>'; }

function renderResidual(d){
  const segs = d.residual.segments;
  const maxV = Math.max(1, ...d.residual.max_deg);
  $("residual").innerHTML = segs.map((s,i)=>
    '<div style="font-size:12px"><span style="display:flex;justify-content:space-between">'+
    '<span>'+s+'</span>'+tagFor(d.residual.mean_deg[i],3,8)+'</span>'+
    barFor(d.residual.mean_deg[i], maxV)+
    '<div class="muted" style="font-size:11px">最大 '+fmt(d.residual.max_deg[i],3)+'°</div></div>'
  ).join("");
}

function renderAnchors(d){
  const slide = d.foot_slide_total_m;
  let html = Object.entries(slide).map(([seg,v])=>metric(seg+" 累计滑动", fmt(v,5)+" m")).join("");
  $("anchors").innerHTML = html || '<span class="muted">无锚点</span>';
  const cf = d.conflicts;
  if(!cf.length){ $("conflicts").innerHTML = '<p class="muted">无航向冲突。</p>'; return; }
  $("conflicts").innerHTML = cf.slice(0, 12).map(c=>
    '<div class="conflict">帧 '+c.frame+' · t='+fmt(c.t,2)+'s · '+c.segment+
    ' <span class="ch '+c.choice+'">'+(c.choice==="correct_heading"?"修正航向":"释放锚点")+'</span>'+
    '<div class="muted">足部航向偏差 '+fmt(c.foot_heading_err_deg,2)+'° · '+
    '若释放假设足滑 '+fmt(c.slide_if_release_m,4)+' m<br>'+
    '代价：航向 '+fmt(c.cost_heading,3)+' vs 释放 '+fmt(c.cost_release,3)+'</div></div>'
  ).join("") + (cf.length>12 ? '<p class="muted">仅显示前 12 条，共 '+cf.length+' 条</p>':'');
}

function renderCalib(d){
  const c = d.calibration;
  $("calib").innerHTML = c.segments.map((s,i)=>
    '<div style="font-size:12px"><span style="display:flex;justify-content:space-between">'+
    '<span>'+s+'</span>'+tagFor(c.residual_rms_deg[i],1,5)+'</span>'+
    '<div class="muted" style="font-size:11px">世界参考航向 '+
    fmt(c.world_yaw_deg[i],2)+'°（独立于安装旋转）</div></div>'
  ).join("");
}

// ---------------- 3D 渲染（手写正交投影，无第三方库） ----------------
function resize(){
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
}
window.addEventListener("resize", resize);

function project(p, center, scale, pitch){
  // 世界坐标：Z 向上，角色面向 +X，左 +Y。先绕 Z 转视角偏航，再绕 X 俯仰投影。
  const cy=Math.cos(viewYaw), sy=Math.sin(viewYaw);
  const x = p[0]*cy - p[1]*sy;
  const y = p[0]*sy + p[1]*cy;
  const cp=Math.cos(pitch), sp=Math.sin(pitch);
  const y2 = y*cp - p[2]*sp;
  const z2 = y*sp + p[2]*cp;
  return [center[0] + x*scale, center[1] - z2*scale, y2];
}

function draw(){
  if(!canvas.width) resize();
  const W=canvas.width, H=canvas.height;
  ctx.clearRect(0,0,W,H);
  const pitch = 0.35;
  // 地面网格
  ctx.strokeStyle="#20252e"; ctx.lineWidth=1;
  for(let g=-4;g<=4;g++){
    const a=project([g,-4,0],[W/2,H*0.62],Math.min(W,H)*0.42,pitch);
    const b=project([g,4,0],[W/2,H*0.62],Math.min(W,H)*0.42,pitch);
    ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
    const c=project([-4,g,0],[W/2,H*0.62],Math.min(W,H)*0.42,pitch);
    const d=project([4,g,0],[W/2,H*0.62],Math.min(W,H)*0.42,pitch);
    ctx.beginPath(); ctx.moveTo(c[0],c[1]); ctx.lineTo(d[0],d[1]); ctx.stroke();
  }
  if(!DATA){ requestAnimationFrame(draw); return; }
  const fr = DATA.frames[frame];
  const center=[W/2,H*0.60], scale=Math.min(W,H)*0.42;
  const P = {};
  for(const j of DATA.bones) P[j]=project(fr.positions[j],center,scale,pitch);

  // 锚点（足锁位置）
  for(const seg of Object.keys(fr.anchors)){
    const a=project(fr.anchors[seg],center,scale,pitch);
    ctx.fillStyle="#e0a93b"; ctx.beginPath();
    ctx.arc(a[0],a[1],5*devicePixelRatio,0,7); ctx.fill();
  }
  // 骨架线段（按深度排序简单处理：左右异色）
  ctx.lineWidth=3*devicePixelRatio;
  for(let i=1;i<DATA.bones.length;i++){
    const j=DATA.bones[i];
    const parent = parentOf(j);
    const a=P[parent], b=P[j];
    ctx.strokeStyle = j.startsWith("l_") ? "#5aa2ff" :
                      (j.startsWith("r_") ? "#7ddfa0" : "#c8d0de");
    ctx.beginPath(); ctx.moveTo(a[0],a[1]); ctx.lineTo(b[0],b[1]); ctx.stroke();
  }
  // 关节点
  for(const j of DATA.bones){
    const p=P[j];
    ctx.fillStyle="#d7dce4"; ctx.beginPath();
    ctx.arc(p[0],p[1],3*devicePixelRatio,0,7); ctx.fill();
  }
  $("frameinfo").textContent = frame+" / "+(DATA.meta.num_frames-1)+
    "  t="+fr.t.toFixed(2)+"s";
  requestAnimationFrame(draw);
}
const PARENT={};
function parentOf(j){ return PARENT[j]; }

function buildParentMap(d){
  // 由位置无法直接得父子；用骨架固定关系（与后端 skeleton.order 同构）
  const map={pelvis:null,spine:"pelvis",neck:"spine",head:"neck",
    l_shoulder:"spine",l_elbow:"l_shoulder",l_wrist:"l_elbow",
    r_shoulder:"spine",r_elbow:"r_shoulder",r_wrist:"r_elbow",
    l_hip:"pelvis",l_knee:"l_hip",l_ankle:"l_knee",
    r_hip:"pelvis",r_knee:"r_hip",r_ankle:"r_knee"};
  for(const k in map) PARENT[k]=map[k];
}

function tick(ts){
  if(!DATA){ return; }
  if(playing){
    if(lastTs==null) lastTs=ts;
    const dtSim=(ts-lastTs)/1000*parseFloat($("rate").value);
    lastTs=ts;
    const adv=dtSim/DATA.meta.dt;
    let next=frame+adv;
    if(next>=DATA.meta.num_frames){
      if($("loop").checked){ next=next%DATA.meta.num_frames; }
      else { next=DATA.meta.num_frames-1; playing=false; $("play").textContent="▶ 播放"; }
    }
    frame=Math.floor(next); $("scrub").value=frame;
  } else { lastTs=null; }
}
function loop(ts){ tick(ts); requestAnimationFrame(loop); }

$("run").addEventListener("click", runCapture);
$("play").addEventListener("click", ()=>{
  if(!DATA) return;
  playing=!playing;
  $("play").textContent = playing ? "⏸ 暂停" : "▶ 播放";
});
$("scrub").addEventListener("input", e=>{ frame=parseInt(e.target.value,10); playing=false;
  $("play").textContent="▶ 播放"; });
$("viewyaw").addEventListener("input", e=>{ viewYaw=parseInt(e.target.value,10)*Math.PI/180; });
$("autorot").addEventListener("change", e=>{ autoRotate=e.target.checked; });
setInterval(()=>{
  if(!autoRotate) return;
  viewYaw+=0.02;
  let deg=Math.round(viewYaw*180/Math.PI);
  $("viewyaw").value=((deg+180)%360)-180;
}, 33);

buildParentMap(); resize(); draw(); requestAnimationFrame(loop);
runCapture();
</script>
</body>
</html>"""


class _CaptureHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 安静日志，避免污染终端
        pass

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj: dict, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802 (stdlib 命名)
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send(200, _PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/options":
            self._send_json({
                "ok": True,
                "actions": list(ACTIONS),
                "legal_remaps": [list(p) for p in LEGAL_REMAPS],
                "placement": {seg: sid for seg, sid in DEFAULT_SENSOR_PLACEMENT.items()},
            })
            return
        if parsed.path == "/api/solve":
            try:
                from urllib.parse import parse_qs
                qs = parse_qs(parsed.query)
                params = {k: v[0] for k, v in qs.items()}
                self._send_json(build_capture_payload(params))
            except Exception as exc:  # 链路异常 -> 视窗异常区，而非 500 崩溃
                self._send_json({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """启动捕捉视窗 HTTP 服务并阻塞。"""
    httpd = ThreadingHTTPServer((host, port), _CaptureHandler)
    print(f"惯性姿态角色捕捉视窗已启动：http://{host}:{port}/")
    print("按 Ctrl+C 停止。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m inertial_pose.capture",
        description="惯性姿态角色捕捉视窗（图形入口，浏览器打开）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    serve(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
