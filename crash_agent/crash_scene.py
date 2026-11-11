"""Builds, simulates and renders one crash "run" in Blender (bpy).

Coordinates: meters, Z up, cars drive along +X. Ground top is z = 0.
Verified physics notes (bpy 5.3):
  * MOTOR constraints drive around the empty's local X axis, HINGE rotates around local Z.
  * Negative motor velocity drives the car forward (+X) with the empty rotations used here.
  * rigidbody_world.time_scale can be keyframed -> real slow motion.
"""
import math
import random
import warnings

import bpy
import bmesh
from mathutils import Vector, Matrix, Euler

warnings.filterwarnings("ignore", category=DeprecationWarning)

FPS = 30
SLOW_SCALE = 0.2          # slow-motion speed
APPROACH_A = 60           # frames: front 3/4 tracking shot
APPROACH_B = 50           # frames: chase shot
SLOW_FRAMES = 90          # frames rendered in slow motion around the impact
AFTER_FRAMES = 80         # frames of aftermath (normal speed)
PRE_IMPACT = 4            # slow motion starts this many frames before impact

PAINTS = [
    (0.80, 0.05, 0.04), (0.98, 0.70, 0.02), (0.05, 0.25, 0.85), (0.95, 0.35, 0.02),
    (0.10, 0.55, 0.15), (0.92, 0.92, 0.92), (0.45, 0.10, 0.65), (0.02, 0.60, 0.70),
    (0.08, 0.08, 0.09), (0.85, 0.15, 0.45),
]

STYLES = {
    #          length width height cabinL cabinH cabinX wheelR wheelX  mass
    "sedan":  dict(L=4.4, W=1.85, H=0.55, cabL=2.3, cabH=0.62, cabX=-0.35, R=0.36, WX=1.40, mass=1000),
    "hatch":  dict(L=3.9, W=1.80, H=0.60, cabL=2.4, cabH=0.72, cabX=-0.50, R=0.34, WX=1.25, mass=900),
    "pickup": dict(L=5.0, W=1.95, H=0.70, cabL=1.7, cabH=0.75, cabX=0.45, R=0.42, WX=1.65, mass=1300),
    "van":    dict(L=4.8, W=1.95, H=0.65, cabL=3.6, cabH=1.05, cabX=-0.35, R=0.38, WX=1.60, mass=1400),
    "sports": dict(L=4.4, W=1.95, H=0.45, cabL=1.9, cabH=0.50, cabX=-0.45, R=0.34, WX=1.45, mass=950),
    "tiny":   dict(L=2.4, W=1.40, H=0.50, cabL=1.4, cabH=0.65, cabX=-0.20, R=0.28, WX=0.80, mass=500),
    "truck":  dict(L=7.0, W=2.40, H=1.00, cabL=2.2, cabH=1.20, cabX=2.10, R=0.55, WX=2.50, mass=6000),
}
NORMAL_STYLES = ["sedan", "hatch", "pickup", "van", "sports"]


# ----------------------------------------------------------------------------- helpers
def _sock(node, name, value):
    if name in node.inputs:
        node.inputs[name].default_value = value


def mat_basic(name, color, metal=0.0, rough=0.5, coat=0.0, emit=None, emit_strength=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    _sock(b, "Base Color", (*color, 1))
    _sock(b, "Metallic", metal)
    _sock(b, "Roughness", rough)
    _sock(b, "Coat Weight", coat)
    if emit:
        _sock(b, "Emission Color", (*emit, 1))
        _sock(b, "Emission Strength", emit_strength)
    m.diffuse_color = (*color, 1)
    return m


def mat_noise(name, c1, c2, scale=8.0, rough=0.8, detail=6.0):
    """Principled material whose colour varies between c1 and c2 with noise."""
    m = mat_basic(name, c1, rough=rough)
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise")
    _sock(nz, "Scale", scale)
    _sock(nz, "Detail", detail)
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*c1, 1)
    ramp.color_ramp.elements[1].color = (*c2, 1)
    nt.links.new(tc.outputs["Object"], nz.inputs["Vector"])
    nt.links.new(nz.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], b.inputs["Base Color"])
    return m


class Builder:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.sc = bpy.context.scene
        self.col = self.sc.collection
        self.mats = {}

    # --- meshes
    def box(self, name, size, loc, mats, mat_fn=None, bevel=0.0, extra=None, taper_top=1.0, rot=(0, 0, 0)):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        for v in bm.verts:
            x = v.co.x * size[0]
            if v.co.z > 0:
                x *= taper_top
            v.co = Vector((x, v.co.y * size[1], v.co.z * size[2]))
        if mat_fn:
            for f in bm.faces:
                f.material_index = mat_fn(f.normal)
        if extra:  # list of (center, size, material_index) small boxes merged into this mesh
            for c, s, mi in extra:
                mtx = Matrix.Translation(Vector(c)) @ Matrix.Diagonal((*s, 1.0))
                res = bmesh.ops.create_cube(bm, size=1.0, matrix=mtx)
                faces = {f for v in res["verts"] for f in v.link_faces}
                for f in faces:
                    f.material_index = mi
        bm.to_mesh(me)
        bm.free()
        for m in mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(name, me)
        self.col.objects.link(ob)
        ob.location = loc
        ob.rotation_euler = rot
        if bevel > 0:
            mod = ob.modifiers.new("bevel", "BEVEL")
            mod.width = bevel
            mod.segments = 2
            mod.limit_method = "ANGLE"
        return ob

    def cylinder(self, name, r, depth, loc, mats, rot=(math.pi / 2, 0, 0), segs=28, cap_mat=1):
        me = bpy.data.meshes.new(name)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs, radius1=r, radius2=r, depth=depth)
        for f in bm.faces:
            f.material_index = cap_mat if abs(f.normal.z) > 0.9 else 0
            f.smooth = abs(f.normal.z) <= 0.9
        bm.to_mesh(me)
        bm.free()
        for m in mats:
            me.materials.append(m)
        ob = bpy.data.objects.new(name, me)
        self.col.objects.link(ob)
        ob.location = loc
        ob.rotation_euler = rot
        return ob

    # --- physics
    def rb(self, ob, typ="ACTIVE", mass=1.0, shape="BOX", fric=0.5, bounce=0.05, start_asleep=False):
        vl = bpy.context.view_layer
        for o in vl.objects:
            o.select_set(False)
        vl.objects.active = ob
        ob.select_set(True)
        bpy.ops.rigidbody.object_add(type=typ)
        ob.select_set(False)
        r = ob.rigid_body
        r.collision_shape = shape
        r.mass = mass
        r.friction = fric
        r.restitution = bounce
        r.use_margin = True
        r.collision_margin = 0.02
        if start_asleep:
            r.use_deactivation = True
            r.use_start_deactivated = True
        return ob

    def constraint(self, kind, a, b, loc, rot=(0, 0, 0), thr=None):
        e = bpy.data.objects.new(f"C_{kind}_{b.name}", None)
        self.col.objects.link(e)
        e.location = loc
        e.rotation_euler = rot
        e.empty_display_size = 0.2
        vl = bpy.context.view_layer
        for o in vl.objects:
            o.select_set(False)
        vl.objects.active = e
        e.select_set(True)
        bpy.ops.rigidbody.constraint_add(type=kind)
        e.select_set(False)
        c = e.rigid_body_constraint
        c.object1 = a
        c.object2 = b
        if thr:
            c.use_breaking = True
            c.breaking_threshold = thr
        return e


# ----------------------------------------------------------------------------- world
def setup_world(B, quality, sun_rot):
    sc = B.sc
    sc.render.engine = "CYCLES"
    cy = sc.cycles
    cy.device = "CPU"
    cy.samples = quality["samples"]
    cy.use_adaptive_sampling = True
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    cy.max_bounces = 4
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 2
    cy.transmission_bounces = 2
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    sc.render.use_persistent_data = True
    sc.render.resolution_x = quality["w"]
    sc.render.resolution_y = quality["h"]
    sc.render.resolution_percentage = 100
    sc.render.fps = FPS
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    try:
        sc.view_settings.view_transform = "AgX"
        sc.view_settings.look = "AgX - Punchy"
        sc.view_settings.exposure = -0.1
    except Exception:
        pass

    w = bpy.data.worlds.new("Sky")
    w.use_nodes = True
    sc.world = w
    nt = w.node_tree
    bg = nt.nodes["Background"]
    sky = nt.nodes.new("ShaderNodeTexSky")
    try:
        sky.sky_type = "MULTIPLE_SCATTERING"
    except Exception:
        pass
    sky.sun_elevation = math.radians(32)
    sky.sun_rotation = sun_rot
    try:
        sky.sun_intensity = 0.6
    except Exception:
        pass
    nt.links.new(sky.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 0.22

    sun_data = bpy.data.lights.new("Sun", "SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(1.5)
    sun = bpy.data.objects.new("Sun", sun_data)
    B.col.objects.link(sun)
    sun.rotation_euler = Euler((math.radians(58), 0, sun_rot + math.pi / 2))


def build_ground(B, x_center, theme):
    if theme == "desert":
        g1, g2 = (0.55, 0.42, 0.28), (0.70, 0.56, 0.38)
        h1, h2 = (0.50, 0.36, 0.24), (0.62, 0.48, 0.32)
    else:
        g1, g2 = (0.05, 0.20, 0.02), (0.12, 0.30, 0.04)
        h1, h2 = (0.06, 0.20, 0.04), (0.14, 0.30, 0.06)
    m_ground = mat_noise("ground", g1, g2, scale=0.08, rough=0.95)
    m_road = mat_noise("asphalt", (0.045, 0.045, 0.05), (0.09, 0.09, 0.095), scale=3.0, rough=0.85)
    m_line = mat_basic("line", (0.85, 0.85, 0.80), rough=0.6)
    m_hill = mat_noise("hill", h1, h2, scale=0.02, rough=1.0)

    ground = B.box("Ground", (4000, 600, 2), (x_center, 0, -1), [m_ground])
    B.rb(ground, "PASSIVE", fric=1.0)
    # visual road (no physics)
    B.box("Road", (4000, 10, 0.01), (x_center, 0, 0.004), [m_road])
    extra = []
    for sgn in (-1, 1):
        extra.append(((0, sgn * 4.6, 0.0), (4000, 0.18, 0.01), 0))
    lines = B.box("EdgeLines", (0.001, 0.001, 0.001), (x_center, 0, 0.011), [m_line], extra=extra)
    # centre dashes near the action
    dashes = []
    for i in range(-80, 30):
        dashes.append(((i * 7.0, 0, 0.0), (3.0, 0.15, 0.01), 0))
    B.box("Dashes", (0.001, 0.001, 0.001), (x_center, 0, 0.012), [m_line], extra=dashes)
    # distant hills and props for depth
    rng = B.rng
    for i in range(14):
        me = bpy.data.meshes.new(f"hill{i}")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(m_hill)
        for p in me.polygons:
            p.use_smooth = True
        ob = bpy.data.objects.new(f"hill{i}", me)
        B.col.objects.link(ob)
        side = rng.choice((-1, 1))
        ob.location = (x_center + rng.uniform(-500, 400), side * rng.uniform(160, 320), -rng.uniform(5, 15))
        ob.scale = (rng.uniform(60, 140), rng.uniform(40, 90), rng.uniform(18, 45))
    cols = [(0.75, 0.2, 0.1), (0.1, 0.35, 0.6), (0.85, 0.55, 0.1), (0.2, 0.5, 0.3)]
    for i in range(16):
        c = rng.choice(cols)
        m = mat_basic(f"cont{i}", c, rough=0.6, metal=0.2)
        side = rng.choice((-1, 1))
        B.box(f"container{i}", (6.0, 2.4, 2.6), (x_center + rng.uniform(-250, 60), side * rng.uniform(22, 45), 1.3),
              [m], rot=(0, 0, rng.uniform(-0.3, 0.3)))
    poles = mat_basic("pole", (0.6, 0.6, 0.62), metal=0.8, rough=0.35)
    for i in range(-12, 4):
        for sgn in (-1, 1):
            B.cylinder(f"pole{i}{sgn}", 0.08, 6.0, (x_center + i * 25.0, sgn * 10.0, 3.0), [poles, poles], rot=(0, 0, 0), segs=12)
    # tyre barriers along the road edge
    tyre = mat_basic("tyrewall", (0.03, 0.03, 0.03), rough=0.9)
    for i in range(-30, 6):
        B.box(f"barrier{i}", (5.5, 0.6, 0.8), (x_center + i * 6.0, 8.2, 0.4), [tyre], bevel=0.08)
    return ground


# ----------------------------------------------------------------------------- cars
def build_car(B, style_name, paint, origin, yaw=0.0, speed=20.0, breakable=True, tag="A", rail_anchor=None):
    """Creates a car made of breakable rigid parts. Returns dict with chassis, parts, motor empties."""
    s = STYLES[style_name]
    L, W, H, R = s["L"], s["W"], s["H"], s["R"]
    k = s["mass"] / 1000.0
    m_paint = mat_basic(f"paint{tag}", paint, metal=0.35, rough=0.28, coat=1.0)
    m_glass = mat_basic(f"glass{tag}", (0.02, 0.025, 0.03), metal=0.1, rough=0.05)
    m_dark = mat_basic(f"dark{tag}", (0.02, 0.02, 0.02), rough=0.7)
    m_tire = mat_basic(f"tire{tag}", (0.015, 0.015, 0.015), rough=0.9)
    m_rim = mat_basic(f"rim{tag}", (0.75, 0.75, 0.78), metal=1.0, rough=0.25)
    m_head = mat_basic(f"head{tag}", (1, 1, 0.9), emit=(1, 0.95, 0.8), emit_strength=6.0)
    m_tail = mat_basic(f"tail{tag}", (0.6, 0, 0), emit=(1, 0.05, 0.02), emit_strength=4.0)

    clear = R * 0.85
    cz = clear + H / 2
    top = clear + H
    parts = []

    lights = [
        ((L / 2 - 0.02, W / 2 - 0.35, H / 2 - 0.15), (0.06, 0.38, 0.14), 1),
        ((L / 2 - 0.02, -(W / 2 - 0.35), H / 2 - 0.15), (0.06, 0.38, 0.14), 1),
        ((-L / 2 + 0.02, W / 2 - 0.3, H / 2 - 0.15), (0.06, 0.35, 0.12), 2),
        ((-L / 2 + 0.02, -(W / 2 - 0.3), H / 2 - 0.15), (0.06, 0.35, 0.12), 2),
    ]
    ch = B.box(f"Chassis{tag}", (L, W, H), (0, 0, cz), [m_paint, m_head, m_tail], extra=lights, bevel=0.06)
    B.rb(ch, mass=s["mass"], fric=0.5)
    cab = B.box(f"Cabin{tag}", (s["cabL"], W * 0.88, s["cabH"]), (s["cabX"], 0, top + s["cabH"] / 2 + 0.02),
                [m_paint, m_glass], mat_fn=lambda n: 0 if n.z > 0.9 else 1, bevel=0.07, taper_top=0.78)
    B.rb(cab, mass=150 * k, fric=0.5)
    parts.append(cab)
    hood_len = max(0.6, (L / 2) - (s["cabX"] + s["cabL"] / 2) - 0.12)
    hood_x = L / 2 - hood_len / 2 - 0.06
    hood = B.box(f"Hood{tag}", (hood_len, W * 0.92, 0.07), (hood_x, 0, top + 0.055), [m_paint], bevel=0.02)
    B.rb(hood, mass=30 * k, fric=0.4)
    parts.append(hood)
    bumpers = []
    for sgn in (1, -1):
        bp = B.box(f"Bumper{tag}{'F' if sgn > 0 else 'R'}", (0.2, W * 0.98, 0.26), (sgn * (L / 2 + 0.12), 0, clear + 0.2),
                   [m_dark], bevel=0.05)
        B.rb(bp, mass=20 * k, fric=0.5)
        bumpers.append(bp)
        parts.append(bp)

    big = 1e9
    thr_body = 260 * k if breakable else big
    thr_light = 120 * k if breakable else big
    thr_wheel = 650 * k if breakable else big
    empties = [B.constraint("FIXED", ch, cab, (s["cabX"], 0, top), thr=thr_body),
               B.constraint("FIXED", ch, hood, (hood_x, 0, top), thr=thr_light)]
    for bp in bumpers:
        empties.append(B.constraint("FIXED", ch, bp, (bp.location.x, 0, clear + 0.2), thr=thr_light))

    wheels, motors = [], []
    for i, (x, y) in enumerate([(s["WX"], W / 2 + 0.08), (s["WX"], -W / 2 - 0.08),
                                (-s["WX"], W / 2 + 0.08), (-s["WX"], -W / 2 - 0.08)]):
        w = B.cylinder(f"Wheel{tag}{i}", R, 0.3, (x, y, R), [m_tire, m_rim])
        B.rb(w, mass=25 * k, shape="CYLINDER", fric=1.6)
        empties.append(B.constraint("HINGE", ch, w, (x, y, R), rot=(math.pi / 2, 0, 0), thr=thr_wheel))
        mot = B.constraint("MOTOR", ch, w, (x, y, R), rot=(0, 0, math.pi / 2))
        empties.append(mot)
        c = mot.rigid_body_constraint
        c.use_motor_ang = True
        c.motor_ang_target_velocity = -speed / R
        c.motor_ang_max_impulse = 1000.0 * k * max(1.0, speed / 22.0)
        wheels.append(w)
        motors.append(mot)
        parts.append(w)

    # invisible straight "rail" so the car cannot drift sideways before the impact (switched off at impact)
    rail = None
    if rail_anchor is not None:
        rail = B.constraint("GENERIC", rail_anchor, ch, (0, 0, cz))
        rc = rail.rigid_body_constraint
        rc.disable_collisions = False
        rc.use_limit_lin_y = True
        rc.limit_lin_y_lower = 0.0
        rc.limit_lin_y_upper = 0.0
        rc.use_limit_ang_z = True
        rc.limit_ang_z_lower = 0.0
        rc.limit_ang_z_upper = 0.0
        empties.append(rail)

    # move the whole car (parts + constraint empties) into place
    M = Matrix.Translation(Vector(origin)) @ Matrix.Rotation(yaw, 4, "Z")
    for ob in [ch] + parts + empties:
        ob.matrix_world = M @ ob.matrix_world
    front = L / 2 + 0.22
    return dict(chassis=ch, parts=parts, motors=motors, rail=rail, front=front, style=style_name, length=L)


# ----------------------------------------------------------------------------- obstacles
def build_obstacle(B, scenario, face_x, twist, rng):
    """face_x = x of the obstacle face the car hits. Returns description dict."""
    info = dict(focus=Vector((face_x, 0, 1.2)))
    if scenario == "wall":
        m = mat_noise("concrete", (0.42, 0.42, 0.40), (0.62, 0.61, 0.58), scale=4.0, rough=0.9)
        stripe = mat_basic("stripe", (0.9, 0.7, 0.02), rough=0.5)
        wall = B.box("Wall", (1.5, 12, 3.6), (face_x + 0.75, 0, 1.8), [m, stripe],
                     extra=[((-0.76, 0, -0.9), (0.02, 12, 0.5), 1)], bevel=0.05)
        B.rb(wall, "PASSIVE", fric=0.7)
    elif scenario == "bricks":
        m1 = mat_noise("brick", (0.45, 0.12, 0.06), (0.62, 0.22, 0.10), scale=12.0, rough=0.9)
        bx, by, bz = 0.5, 0.95, 0.42
        cols = 13 if twist else 9
        rows = 7
        for r in range(rows):
            off = (by / 2) if r % 2 else 0.0
            for c in range(cols + (1 if r % 2 else 0)):
                y = (c - cols / 2 + 0.5) * by - off
                if abs(y) > cols * by / 2:
                    continue
                b = B.box(f"brick{r}_{c}", (bx, by - 0.02, bz - 0.01), (face_x + bx / 2, y, bz / 2 + r * bz), [m1], bevel=0.02)
                B.rb(b, mass=40, fric=0.8, bounce=0.05, start_asleep=True)
        info["focus"] = Vector((face_x, 0, 1.4))
    elif scenario == "bollards":
        m = mat_basic("bollard", (0.95, 0.72, 0.02), metal=0.3, rough=0.4)
        mb = mat_basic("bollardcap", (0.05, 0.05, 0.05), rough=0.6)
        for i in range(5):
            y = (i - 2) * 1.05
            bo = B.cylinder(f"bollard{i}", 0.17, 1.2, (face_x + 0.17, y, 0.6), [m, mb], rot=(0, 0, 0))
            B.rb(bo, "PASSIVE", shape="CYLINDER", fric=0.6)
        if twist:  # hidden launch ramp
            mr = mat_basic("ramp", (0.3, 0.3, 0.32), metal=0.6, rough=0.4)
            ramp = B.box("Ramp", (9.0, 3.4, 0.2), (face_x - 7.0, 0, 1.0), [mr], rot=(0, math.radians(-14), 0))
            B.rb(ramp, "PASSIVE", fric=0.9)
            ramp2 = B.box("RampBase", (9.0, 3.2, 0.1), (face_x - 7.0, 0, 0.05), [mr])
            info["focus"] = Vector((face_x, 0, 2.5))
    elif scenario == "head_on":
        pass
    return info


# ----------------------------------------------------------------------------- simulation
def make_scene(cfg, with_obstacle, impact_frame=None):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    B = Builder(cfg["seed"])
    sc = B.sc
    rng = random.Random(cfg["seed"])
    sc.frame_start = 1
    sc.frame_end = cfg["sim_end"]
    sc.render.fps = FPS
    bpy.ops.rigidbody.world_add()
    rw = sc.rigidbody_world
    rw.substeps_per_frame = 20
    rw.solver_iterations = 30
    rw.point_cache.frame_start = 1
    rw.point_cache.frame_end = cfg["sim_end"]

    setup_world(B, cfg["quality"], cfg["sun_rot"])
    face_x = cfg["face_x"]
    ground = build_ground(B, face_x, cfg["theme"])

    cars = []
    v = cfg["speed_mps"]
    scen = cfg["scenario"]
    twist = cfg["twist"]
    if scen == "head_on":
        st_a = "tiny" if twist else cfg["style"]
        st_b = "truck" if twist else cfg["style_b"]
        cars.append(build_car(B, st_a, cfg["paint"], (face_x - cfg["runway"], 0, 0), 0.0, v, True, "A", ground))
        cars.append(build_car(B, st_b, cfg["paint_b"], (face_x + cfg["runway"], 0, 0), math.pi, v, True, "B", ground))
    elif scen == "bricks" and twist:
        for j, y in enumerate((-3.0, 0.0, 3.0)):
            st = NORMAL_STYLES[(j + cfg["seed"]) % len(NORMAL_STYLES)]
            cars.append(build_car(B, st, PAINTS[(j * 3 + cfg["seed"]) % len(PAINTS)], (face_x - cfg["runway"] - j * 0.7, y, 0), 0.0, v, True, f"T{j}", ground))
    else:
        st = "tiny" if (twist and scen == "wall") else cfg["style"]
        cars.append(build_car(B, st, cfg["paint"], (face_x - cfg["runway"], 0, 0), 0.0, v,
                              breakable=not (twist and scen == "wall"), tag="A", rail_anchor=ground))

    info = dict(focus=Vector((face_x, 0, 1.2)))
    if with_obstacle and scen != "head_on":
        info = build_obstacle(B, scen, face_x, twist, rng)

    if impact_frame is not None:
        f0 = impact_frame - PRE_IMPACT
        rw.time_scale = 1.0
        rw.keyframe_insert("time_scale", frame=f0)
        rw.time_scale = SLOW_SCALE
        rw.keyframe_insert("time_scale", frame=f0 + 3)
        rw.keyframe_insert("time_scale", frame=f0 + SLOW_FRAMES - 4)
        rw.time_scale = 1.0
        rw.keyframe_insert("time_scale", frame=f0 + SLOW_FRAMES + 4)
        # motors and rails off at impact so the wrecks move freely
        for car in cars:
            for mot in car["motors"] + ([car["rail"]] if car["rail"] else []):
                c = mot.rigid_body_constraint
                c.enabled = True
                c.keyframe_insert("enabled", frame=impact_frame - 2)
                c.enabled = False
                c.keyframe_insert("enabled", frame=impact_frame - 1)
    return B, cars, info


def run_sim(cars, end):
    """Steps the simulation from frame 1, logs positions and time scale."""
    sc = bpy.context.scene
    rw = sc.rigidbody_world
    log = []
    for f in range(1, end + 1):
        sc.frame_set(f)
        entry = dict(f=f, ts=rw.time_scale, cars=[])
        for car in cars:
            entry["cars"].append(dict(
                p=tuple(car["chassis"].matrix_world.translation),
                parts=[tuple(p.matrix_world.translation) for p in car["parts"]],
            ))
        log.append(entry)
    return log


def car_speeds(log, ci):
    out = [0.0]
    for a, b in zip(log, log[1:]):
        d = (Vector(b["cars"][ci]["p"]) - Vector(a["cars"][ci]["p"])).length
        out.append(d * FPS / max(b["ts"], 1e-3))
    return out


# ----------------------------------------------------------------------------- cameras
def look_rot(cam_pos, target):
    d = Vector(target) - Vector(cam_pos)
    return d.to_track_quat("-Z", "Y").to_euler()


def smooth(points, k=6):
    out = []
    n = len(points)
    for i in range(n):
        a, b = max(0, i - k), min(n, i + k + 1)
        s = Vector((0, 0, 0))
        for j in range(a, b):
            s += Vector(points[j])
        out.append(s / (b - a))
    return out


def add_camera(name, lens):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    cd.clip_end = 3000
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 24
    ob = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def setup_cameras(log, cars, info, impact, rng, scenario, twist):
    sc = bpy.context.scene
    car_pos = [Vector(e["cars"][0]["p"]) for e in log]
    sm = smooth(car_pos, 5)
    side = rng.choice((-1, 1))
    focus = info["focus"]

    a0 = impact - PRE_IMPACT - APPROACH_B - APPROACH_A
    b0 = impact - PRE_IMPACT - APPROACH_B
    c0 = impact - PRE_IMPACT
    d0 = c0 + SLOW_FRAMES
    end = d0 + AFTER_FRAMES

    # Shot A: front 3/4 tracking (camera ahead of car, car approaching)
    camA = add_camera("CamA", 22)
    for f in range(a0, b0 + 1):
        p = sm[f - 1]
        pos = p + Vector((9.5, side * 6.0, 1.4))
        camA.location = pos
        camA.rotation_euler = look_rot(pos, p + Vector((0, 0, 0.6)))
        camA.keyframe_insert("location", frame=f)
        camA.keyframe_insert("rotation_euler", frame=f)
    # Shot B: chase cam, obstacle visible ahead
    camB = add_camera("CamB", 20)
    for f in range(b0, c0 + 1):
        p = sm[f - 1]
        pos = p + Vector((-8.5, -side * 2.2, 2.6))
        camB.location = pos
        camB.rotation_euler = look_rot(pos, p + Vector((7.0, 0, 0.4)))
        camB.keyframe_insert("location", frame=f)
        camB.keyframe_insert("rotation_euler", frame=f)
    # Shot C: low static impact camera (slow motion)
    camC = add_camera("CamC", 15)
    if scenario == "head_on":
        hit = Vector((log[impact - 1]["cars"][0]["p"][0] + 2.2, 0, 0.9))
        posC = hit + Vector((0.0, side * 7.4, 1.3))
        tgt = hit
    else:
        hit = Vector((focus.x, 0, 0.9))
        posC = hit + Vector((-3.0, side * 7.4, 1.2))
        tgt = hit + Vector((-3.0, 0, 0.1))
        if scenario == "bollards" and twist:
            posC = hit + Vector((4.0, side * 16.0, 2.5))
            tgt = hit + Vector((2.0, 0, 2.0))
    # static position, but the camera pans to keep the car in frame during the slow motion
    camC.location = posC
    for f in range(c0, d0 + 1):
        car = sm[min(f, len(sm)) - 1]
        aim = tgt.lerp(Vector((car.x, car.y, max(car.z, 0.6))), 0.55)
        camC.rotation_euler = look_rot(posC, aim)
        camC.keyframe_insert("rotation_euler", frame=f)
    # Shot D: slow orbit around the wreck
    camD = add_camera("CamD", 22)
    # orbit around the (heavily smoothed) centre of all cars so moving wrecks stay in frame
    centres, radii = [], []
    for e in log:
        pts = [Vector(car["p"]) for car in e["cars"]]
        c = sum(pts, Vector((0, 0, 0))) / len(pts)
        spread = max((p - c).length for p in pts)
        centres.append(c)
        radii.append(Vector((min(40.0, 11.0 + 1.6 * spread), 0, 0)))
    centres = smooth(centres, 4)
    radii = smooth(radii, 8)
    ang0 = rng.uniform(0, 2 * math.pi)
    for i, f in enumerate(range(d0, end + 1)):
        centre = centres[min(f, len(centres)) - 1].copy()
        centre.z = max(0.8, centre.z)
        rad = radii[min(f, len(radii)) - 1].x
        a = ang0 + i * 0.006
        pos = centre + Vector((math.cos(a) * rad, math.sin(a) * rad, 4.5 * rad / 11.0))
        camD.location = pos
        camD.rotation_euler = look_rot(pos, centre)
        camD.keyframe_insert("location", frame=f)
        camD.keyframe_insert("rotation_euler", frame=f)

    for cam, f in ((camA, a0), (camB, b0), (camC, c0), (camD, d0)):
        m = sc.timeline_markers.new(cam.name, frame=f)
        m.camera = cam
    sc.camera = camA
    return a0, end, dict(slow_start=c0, slow_end=d0)


# ----------------------------------------------------------------------------- one run
def plan_run(cfg):
    """Pass 1: drive without obstacle to find where the car reaches its speed."""
    v = cfg["speed_mps"]
    cfg["runway"] = 0.0
    cfg["face_x"] = 0.0
    cfg["sim_end"] = 400
    make_scene(cfg, with_obstacle=False)
    probe = run_sim_probe(cfg)
    return probe


def run_sim_probe(cfg):
    sc = bpy.context.scene
    ch = bpy.data.objects["ChassisA"]
    v = cfg["speed_mps"]
    prev = ch.matrix_world.translation.x
    reach = None
    xs = [prev]
    for f in range(2, cfg["sim_end"] + 1):
        sc.frame_set(f)
        x = ch.matrix_world.translation.x
        xs.append(x)
        if reach is None and (x - prev) * FPS >= 0.97 * v:
            reach = f
        prev = x
        if reach and f > reach + 5:
            break
    return dict(reach=reach or len(xs), xs=xs)


def render_run(cfg, out_dir):
    cfg = dict(cfg)
    """Full pipeline for one run. Returns metadata used by audio/overlay steps."""
    v = cfg["speed_mps"]
    cfg2 = dict(cfg)
    cfg2["scenario"] = "probe"   # single car on an empty road
    cfg2["twist"] = False
    probe = plan_run(cfg2)
    # impact must happen after the car has reached speed and after the approach shots start
    lead = APPROACH_A + APPROACH_B + PRE_IMPACT
    impact = max(probe["reach"] + 20, lead + 10)
    # distance travelled by frame `impact` at full speed
    xs = probe["xs"]
    if impact <= len(xs):
        dist = xs[impact - 1] - xs[0]
    else:
        dist = xs[-1] - xs[0] + (impact - len(xs)) * v / FPS
    if cfg["scenario"] == "head_on":
        runway = dist + STYLES["truck" if cfg["twist"] else cfg["style_b"]]["L"] / 2  # each car drives `dist`
    else:
        runway = dist + STYLES["tiny" if (cfg["twist"] and cfg["scenario"] == "wall") else cfg["style"]]["L"] / 2 + 0.25
    cfg["runway"] = runway
    cfg["face_x"] = 0.0
    cfg["sim_end"] = impact + SLOW_FRAMES + AFTER_FRAMES + 2
    if cfg["scenario"] == "head_on":
        # two cars starting at +-runway meet near x = 0
        cfg["runway"] = dist + 2.6

    # pass 2: find real impact frame with the obstacle (normal speed)
    B, cars, info = make_scene(cfg, with_obstacle=True)
    log = run_sim(cars, cfg["sim_end"])
    sp = car_speeds(log, 0)
    real_imp = impact
    for i in range(max(probe["reach"], impact - 20), min(len(sp) - 1, impact + 20)):
        if sp[i] < 0.75 * v:
            real_imp = i  # first frame with the speed drop
            break
    cfg["sim_end"] = real_imp + SLOW_FRAMES + AFTER_FRAMES + 6
    # pass 3: final simulation with slow motion around impact
    B, cars, info = make_scene(cfg, with_obstacle=True, impact_frame=real_imp)
    rng = random.Random(cfg["seed"] * 7 + 1)
    log = run_sim(cars, cfg["sim_end"])
    start, end, slow = setup_cameras(log, cars, info, real_imp, rng, cfg["scenario"], cfg["twist"])
    sc = bpy.context.scene
    sc.frame_start = start
    sc.frame_end = end
    sc.render.filepath = f"{out_dir}/f_"
    if cfg.get("render", True):
        bpy.ops.render.render(animation=True)

    # data for sound design (per rendered frame)
    frames = []
    speeds = car_speeds(log, 0)
    for f in range(start, end + 1):
        e = log[f - 1]
        frames.append(dict(f=f, ts=e["ts"], v=speeds[f - 1]))
    # impact events: big drops in any part's velocity
    events = []
    prev_vel = None
    for f in range(max(2, start), end + 1):
        a, b = log[f - 2], log[f - 1]
        vel = []
        for ca, cb in zip(a["cars"], b["cars"]):
            pa = [ca["p"]] + ca["parts"]
            pb = [cb["p"]] + cb["parts"]
            for x, y in zip(pa, pb):
                vel.append((Vector(y) - Vector(x)) * FPS / max(b["ts"], 1e-3))
        if prev_vel is not None:
            dv = max((u - w).length for u, w in zip(vel, prev_vel))
            if dv > 4.0:
                events.append(dict(f=f, strength=min(1.0, dv / 30.0)))
        prev_vel = vel
    return dict(start=start, end=end, impact=real_imp, slow_start=slow["slow_start"], slow_end=slow["slow_end"],
                frames=frames, events=events, speed_mps=v)
