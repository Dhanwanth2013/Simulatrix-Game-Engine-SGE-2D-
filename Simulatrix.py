"""
Ultimate Single-File 2D Simulation Engine
Author: Dhanwanth 
Features:
 - Spatial hash broadphase (fast)
 - Frame-friendly memory usage (minimal temporaries)
 - Semi-implicit Euler & Velocity Verlet integration (toggle with V)
 - Circle-circle collisions with impulse resolution
 - Object pooling for short-lived particles
 - Debug HUD, grid overlay, profiler
 - Recording frames (PNG sequence)
 - Example constraints (spring)
 - Runtime hotkeys: G,H,Q,V,P,R,+,-,Esc
"""

import pygame
import random
import math
import time
import os
from collections import deque
from typing import List, Tuple

# ---------------------------
# ========== CONFIG =========
# ---------------------------
WIDTH, HEIGHT = 1200, 740
FPS_TARGET = 60

# initial scene parameters
NUM_BALLS = 600
MIN_RADIUS, MAX_RADIUS = 4, 10
AUTO_CELL_FACTOR = 3         # cell_size = AUTO_CELL_FACTOR * max_radius

# physics defaults
GLOBAL_GRAVITY = 0.0
USE_SPATIAL_HASH_DEFAULT = True
USE_OBJECT_POOL = True
POOL_SIZE = 2000

# recording frames
FRAME_DIR = "frames"
RECORD_DEFAULT = False

# ---------------------------
# ======= UTILITIES =========
# ---------------------------
SQRT = math.sqrt
INV_2PI = 1.0 / (2 * math.pi)

def clamp(x, a, b): return a if x < a else (b if x > b else x)

# ---------------------------
# === SPATIAL HASH GRID ====
# ---------------------------
class SpatialHashGrid:
    """High-performance spatial hash using preallocated cells and neighbor cache."""
    __slots__ = ("width","height","cell_size","cols","rows","cells","neighbor_cache")
    def __init__(self, width:int, height:int, cell_size:int):
        self.width = width
        self.height = height
        self.cell_size = max(4, int(cell_size))
        self.cols = max(1, (self.width + self.cell_size - 1) // self.cell_size)
        self.rows = max(1, (self.height + self.cell_size - 1) // self.cell_size)
        total = self.cols * self.rows
        # preallocate cell lists
        self.cells = [[] for _ in range(total)]
        # neighbor cache: for each cell store indices of 3x3 neighbors (clamped)
        self.neighbor_cache = [None] * total
        for cy in range(self.rows):
            for cx in range(self.cols):
                idx = cy * self.cols + cx
                neigh = []
                y0 = max(0, cy - 1)
                y1 = min(self.rows - 1, cy + 1)
                x0 = max(0, cx - 1)
                x1 = min(self.cols - 1, cx + 1)
                for ny in range(y0, y1 + 1):
                    base = ny * self.cols
                    for nx in range(x0, x1 + 1):
                        neigh.append(base + nx)
                self.neighbor_cache[idx] = tuple(neigh)

    def clear(self):
        # clear lists in place without reallocation
        for lst in self.cells:
            lst.clear()

    def insert(self, ent):
        cx = int(ent.x // self.cell_size)
        cy = int(ent.y // self.cell_size)
        if cx < 0: cx = 0
        elif cx >= self.cols: cx = self.cols - 1
        if cy < 0: cy = 0
        elif cy >= self.rows: cy = self.rows - 1
        idx = cy * self.cols + cx
        self.cells[idx].append(ent)

    def get_neighbors_indices(self, ent):
        cx = int(ent.x // self.cell_size)
        cy = int(ent.y // self.cell_size)
        if cx < 0: cx = 0
        elif cx >= self.cols: cx = self.cols - 1
        if cy < 0: cy = 0
        elif cy >= self.rows: cy = self.rows - 1
        idx = cy * self.cols + cx
        return self.neighbor_cache[idx]

# ---------------------------
# ======== ENTITIES =========
# ---------------------------
class Entity:
    __slots__ = ("x","y","vx","vy","ax","ay","radius","mass","color",
                 "restitution","friction","alive","tag")
    def __init__(self, x:float=0.0, y:float=0.0, radius:float=6.0, color:Tuple[int,int,int]=(200,200,200), mass:float=None):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0
        self.radius = float(radius)
        self.mass = float(mass if mass is not None else max(0.1, self.radius * 0.6))
        self.color = color
        self.restitution = 0.9
        self.friction = 0.999
        self.alive = True
        self.tag = None

    def apply_force(self, fx:float, fy:float):
        self.ax += fx / self.mass
        self.ay += fy / self.mass

    def integrate_euler(self, engine):
        """Semi-implicit Euler integration (v then x)."""
        # apply gravity (acceleration units assumed)
        self.ay += engine.gravity
        # integrate velocity
        self.vx += self.ax
        self.vy += self.ay
        # integrate position
        self.x += self.vx
        self.y += self.vy
        # reset acceleration
        self.ax = 0.0
        self.ay = 0.0
        # damping
        self.vx *= self.friction
        self.vy *= self.friction
        # boundary collisions (contain in screen)
        r = self.radius
        if self.x - r < 0.0:
            self.x = r
            self.vx = -self.vx * self.restitution
        elif self.x + r > engine.width:
            self.x = engine.width - r
            self.vx = -self.vx * self.restitution
        if self.y - r < 0.0:
            self.y = r
            self.vy = -self.vy * self.restitution
        elif self.y + r > engine.height:
            self.y = engine.height - r
            self.vy = -self.vy * self.restitution

    # for Verlet we use previous positions stored in physics world
    def integrate_verlet(self, prev_x, prev_y, engine):
        # acceleration includes gravity (engine.gravity)
        ax = self.ax + engine.gravity * self.mass
        ay = self.ay + engine.gravity * self.mass
        # velocity approx via (x - prev) / dt if needed; but we store vx/vy as estimate
        # new position = 2*current - prev + a * dt^2. We use dt=1 scaled so acceleration inserted directly.
        # Here integration steps are scaled similarly to Euler code (consistent scale).
        new_x = 2.0 * self.x - prev_x + ax
        new_y = 2.0 * self.y - prev_y + ay
        # update vx/vy estimate
        self.vx = (new_x - prev_x) * 0.5
        self.vy = (new_y - prev_y) * 0.5
        # reset accel
        self.ax = 0.0
        self.ay = 0.0
        # friction/damping applied to velocity
        self.vx *= self.friction
        self.vy *= self.friction
        # boundary handling (quick clamp + reflect)
        r = self.radius
        if new_x - r < 0.0:
            new_x = r
            self.vx = -self.vx * self.restitution
        elif new_x + r > engine.width:
            new_x = engine.width - r
            self.vx = -self.vx * self.restitution
        if new_y - r < 0.0:
            new_y = r
            self.vy = -self.vy * self.restitution
        elif new_y + r > engine.height:
            new_y = engine.height - r
            self.vy = -self.vy * self.restitution
        # set prev before returning new pos
        return new_x, new_y

    def draw(self, surf, outline=True):
        pygame.draw.circle(surf, self.color, (int(self.x), int(self.y)), int(self.radius))
        if outline:
            pygame.draw.circle(surf, (0,0,0), (int(self.x), int(self.y)), int(self.radius), 1)

# ---------------------------
# ======= OBJECT POOL =======
# ---------------------------
class ObjectPool:
    """Simple pool for Entity objects to avoid frequent allocation."""
    def __init__(self, cls, size:int, *args, **kwargs):
        self.cls = cls
        self.pool = deque()
        self.args = args
        self.kwargs = kwargs
        for _ in range(size):
            obj = cls(*args, **kwargs)
            obj.alive = False
            self.pool.append(obj)
    def acquire(self, **overrides):
        if self.pool:
            obj = self.pool.pop()
            obj.alive = True
            # reset important fields
            obj.x = overrides.get('x', obj.x)
            obj.y = overrides.get('y', obj.y)
            obj.vx = overrides.get('vx', 0.0)
            obj.vy = overrides.get('vy', 0.0)
            obj.ax = 0.0; obj.ay = 0.0
            obj.radius = overrides.get('radius', obj.radius)
            obj.color = overrides.get('color', obj.color)
            obj.mass = overrides.get('mass', obj.mass)
            return obj
        else:
            # fallback allocate
            obj = self.cls(**self.kwargs)
            obj.alive = True
            return obj
    def release(self, obj):
        obj.alive = False
        self.pool.append(obj)

# ---------------------------
# ======= SPRING (EXAMPLE) ==
# ---------------------------
class Spring:
    """Simple linear spring between two entities (hooke)."""
    __slots__ = ("a","b","rest_length","k","damping")
    def __init__(self, a:Entity, b:Entity, k:float=0.2, rest_length:float=None, damping:float=0.02):
        self.a = a
        self.b = b
        self.k = k
        self.damping = damping
        self.rest_length = rest_length if rest_length is not None else math.hypot(b.x-a.x, b.y-a.y)

    def apply(self):
        dx = self.b.x - self.a.x
        dy = self.b.y - self.a.y
        dist = math.hypot(dx,dy) if dx*dx+dy*dy != 0 else 0.0001
        nx = dx / dist
        ny = dy / dist
        # spring force (Hooke)
        fs = self.k * (dist - self.rest_length)
        # damping - relative velocity along spring
        rv = (self.b.vx - self.a.vx) * nx + (self.b.vy - self.a.vy) * ny
        fd = self.damping * rv
        fx = (fs + fd) * nx
        fy = (fs + fd) * ny
        # apply opposite forces
        self.a.apply_force(fx, fy)
        self.b.apply_force(-fx, -fy)

# ---------------------------
# ===== PHYSICS WORLD =======
# ---------------------------
class PhysicsWorld:
    """
    Manage objects, integrator switching (euler/verlet), spatial hash broadphase,
    collision resolution, constraints, pooling.
    """
    __slots__ = ("width","height","gravity","entities","springs",
                 "use_spatial_hash","grid","max_radius","cell_size",
                 "use_pool","pool","integrator","prev_positions",
                 "collision_pairs_checked","time_profile")

    def __init__(self, width:int, height:int, max_radius:float=MAX_RADIUS,
                 use_spatial_hash:bool=True, use_pool:bool=True, integrator:str="euler"):
        self.width = width
        self.height = height
        self.gravity = GLOBAL_GRAVITY
        self.entities: List[Entity] = []
        self.springs: List[Spring] = []
        self.use_spatial_hash = use_spatial_hash
        self.max_radius = max_radius
        self.cell_size = max(4, int(AUTO_CELL_FACTOR * max_radius))
        self.grid = SpatialHashGrid(self.width, self.height, self.cell_size)
        self.use_pool = use_pool
        self.pool = ObjectPool(Entity, POOL_SIZE) if use_pool else None
        self.integrator = integrator  # "euler" or "verlet"
        # previous positions mapping for verlet integration
        self.prev_positions = {}  # entity -> (prev_x, prev_y)
        # simple profiling stats
        self.collision_pairs_checked = 0
        self.time_profile = {"integrate":0.0, "broadphase":0.0, "resolve":0.0, "constraints":0.0}

    # ---- entity management
    def add_entity(self, ent:Entity):
        self.entities.append(ent)
        # if verlet, initialize previous pos (current - velocity)
        self.prev_positions[ent] = (ent.x - ent.vx, ent.y - ent.vy)

    def spawn_ball(self, x:float, y:float, radius:float, color=None):
        if self.use_pool:
            e = self.pool.acquire(x=x, y=y, radius=radius, color=(random.randint(60,255),random.randint(60,255),random.randint(60,255)))
        else:
            e = Entity(x, y, radius)
            e.color = color if color is not None else (random.randint(60,255),random.randint(60,255),random.randint(60,255))
        e.vx = random.uniform(-1.6, 1.6)
        e.vy = random.uniform(-1.6, 1.6)
        e.restitution = 0.92
        e.friction = 0.999
        e.mass = max(0.1, radius * 0.6)
        self.add_entity(e)
        return e

    def remove_entity(self, ent:Entity):
        # safe removal (release to pool if appropriate)
        try:
            self.entities.remove(ent)
        except ValueError:
            pass
        if self.use_pool and self.pool:
            self.pool.release(ent)
        if ent in self.prev_positions:
            del self.prev_positions[ent]

    # ---- constraints
    def add_spring(self, a:Entity, b:Entity, k=0.2, rest=None, damping=0.02):
        s = Spring(a,b,k,rest,damping)
        self.springs.append(s)
        return s

    # ---- main step
    def step(self):
        t0 = time.perf_counter()
        # integrate velocities/positions
        self.time_profile["integrate"] = 0.0
        self.time_profile["broadphase"] = 0.0
        self.time_profile["resolve"] = 0.0
        self.time_profile["constraints"] = 0.0

        tA = time.perf_counter()
        if self.integrator == "euler":
            for e in self.entities:
                e.integrate_euler(self)
        else:  # verlet
            # verlet requires prev positions map
            for e in self.entities:
                prev = self.prev_positions.get(e, (e.x - e.vx, e.y - e.vy))
                new_x, new_y = e.integrate_verlet(prev[0], prev[1], self)
                # update prev and current
                self.prev_positions[e] = (e.x, e.y)
                e.x = new_x; e.y = new_y
        tB = time.perf_counter()
        self.time_profile["integrate"] = tB - tA

        # apply constraints (springs)
        tC = time.perf_counter()
        if self.springs:
            for s in self.springs:
                s.apply()
        tD = time.perf_counter()
        self.time_profile["constraints"] = tD - tC

        # broadphase & collision resolution
        tE = time.perf_counter()
        if self.use_spatial_hash:
            self._broadphase_spatial()
        else:
            self._broadphase_naive()
        tF = time.perf_counter()
        self.time_profile["broadphase"] = tF - tE

        # done
        self.collision_pairs_checked = 0

    # ---- broadphase spatial
    def _broadphase_spatial(self):
        grid = self.grid
        grid.clear()
        # insert entities into cells
        for e in self.entities:
            grid.insert(e)
        # check neighbors
        checked = set()
        resolve_time = 0.0
        tR0 = time.perf_counter()
        for idx, cell in enumerate(grid.cells):
            if not cell:
                continue
            neigh_idxs = grid.neighbor_cache[idx]
            # for each entity in this cell compare with neighbor cells
            for a in cell:
                if not a.alive:
                    continue
                for nidx in neigh_idxs:
                    other_list = grid.cells[nidx]
                    if not other_list:
                        continue
                    for b in other_list:
                        if a is b or not b.alive:
                            continue
                        aid = id(a); bid = id(b)
                        if aid < bid:
                            pair = (aid,bid)
                        else:
                            pair = (bid,aid)
                        if pair in checked:
                            continue
                        checked.add(pair)
                        # precise collision check (squared)
                        dx = b.x - a.x
                        dy = b.y - a.y
                        dist2 = dx*dx + dy*dy
                        rsum = a.radius + b.radius
                        if dist2 < (rsum * rsum):
                            self._resolve_collision_pair(a,b,dx,dy,dist2,rsum)
        tR1 = time.perf_counter()
        self.time_profile["resolve"] = tR1 - tR0
        self.collision_pairs_checked = len(checked)

    # ---- broadphase naive O(n^2)
    def _broadphase_naive(self):
        n = len(self.entities)
        tR0 = time.perf_counter()
        for i in range(n):
            a = self.entities[i]
            if not a.alive: continue
            for j in range(i+1, n):
                b = self.entities[j]
                if not b.alive: continue
                dx = b.x - a.x
                dy = b.y - a.y
                dist2 = dx*dx + dy*dy
                rsum = a.radius + b.radius
                if dist2 < (rsum*rsum):
                    self._resolve_collision_pair(a,b,dx,dy,dist2,rsum)
        tR1 = time.perf_counter()
        self.time_profile["resolve"] = tR1 - tR0

    # ---- collision resolution
    def _resolve_collision_pair(self, a:Entity, b:Entity, dx:float, dy:float, dist2:float, rsum:float):
        # Avoid division by zero
        if dist2 == 0.0:
            # jitter tiny vector
            nx, ny = 1.0, 0.0
            dist = 1.0
        else:
            dist = SQRT(dist2)
            nx = dx / dist
            ny = dy / dist

        # overlap correction (minimum translation distance)
        overlap = 0.5 * (rsum - dist)
        if overlap > 0:
            a.x -= overlap * nx
            a.y -= overlap * ny
            b.x += overlap * nx
            b.y += overlap * ny

        # relative velocity
        rvx = b.vx - a.vx
        rvy = b.vy - a.vy
        vel_along_normal = rvx * nx + rvy * ny

        if vel_along_normal > 0:
            # moving apart
            return

        # restitution (bounciness)
        e = min(a.restitution, b.restitution)

        inv_mass_a = 0.0 if a.mass == 0.0 else (1.0 / a.mass)
        inv_mass_b = 0.0 if b.mass == 0.0 else (1.0 / b.mass)

        j = -(1.0 + e) * vel_along_normal
        denom = inv_mass_a + inv_mass_b
        if denom == 0.0:
            return
        j = j / denom

        impulse_x = j * nx
        impulse_y = j * ny

        a.vx -= impulse_x * inv_mass_a
        a.vy -= impulse_y * inv_mass_a
        b.vx += impulse_x * inv_mass_b
        b.vy += impulse_y * inv_mass_b

# ---------------------------
# ======== RENDERER =========
# ---------------------------
class Renderer:
    def __init__(self, screen, world:PhysicsWorld,clock):
        self.screen = screen
        self.world = world
        self.clock = clock
        self.font = pygame.font.SysFont(None, 20)

    def draw(self, draw_grid=False, draw_hud=True):
        # draw entities
        for e in self.world.entities:
            if e.alive:
                e.draw(self.screen)

        # optional grid overlay
        if draw_grid:
            self._draw_grid_overlay()

        if draw_hud:
            self._draw_hud()

    def _draw_grid_overlay(self):
        cs = self.world.grid.cell_size
        cols = self.world.grid.cols
        rows = self.world.grid.rows
        for cx in range(1, cols):
            x = cx * cs
            pygame.draw.line(self.screen, (30,30,30), (x,0), (x,self.world.height))
        for cy in range(1, rows):
            y = cy * cs
            pygame.draw.line(self.screen, (30,30,30), (0,y), (self.world.width,y))
        # draw counts
        # small text for cell counts
        # (disabled by default to reduce overhead)
    def _draw_hud(self):
            w = self.world
            fps = int(self.clock.get_fps())
            text_lines = [
                f"Entities: {len(w.entities)}   Integrator: {w.integrator.upper()}   SpatialHash: {w.use_spatial_hash}",
                f"CellSize: {w.grid.cell_size}   PairsChecked: {w.collision_pairs_checked}",
                f"Integrate: {w.time_profile['integrate']*1000:.2f}ms  Broad: {w.time_profile['broadphase']*1000:.2f}ms  Resolve: {w.time_profile['resolve']*1000:.2f}ms  Constraints: {w.time_profile['constraints']*1000:.2f}ms",
                f"Gravity: {round(w.gravity,2)}  Pool: {'On' if w.use_pool else 'Off'}",
                f"FPS: {fps}"
            ]
            y = 6
            for line in text_lines:
                surf = self.font.render(line, True, (220,220,220))
                self.screen.blit(surf, (6,y))
                y += 18

# ---------------------------
# ======== ENGINE = =========
# ---------------------------
class Engine:
    def __init__(self):
        pygame.init()
        self.width = WIDTH
        self.height = HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Ultimate 2D Simulation Engine")
        self.clock = pygame.time.Clock()
        # world & renderer
        self.world = PhysicsWorld(self.width, self.height, max_radius=MAX_RADIUS,
                                  use_spatial_hash=USE_SPATIAL_HASH_DEFAULT, use_pool=USE_OBJECT_POOL,
                                  integrator="euler")
        # spawn initial balls
        self._spawn_initial()
        self.renderer = Renderer(self.screen, self.world,self.clock)
        self.running = True
        # toggles
        self.draw_grid = False
        self.draw_hud = True
        self.paused = False
        self.recording = RECORD_DEFAULT
        self.frame_count = 0

    def _spawn_initial(self):
        margin = MAX_RADIUS + 16
        for _ in range(NUM_BALLS):
            r = random.randint(MIN_RADIUS, MAX_RADIUS)
            x = random.uniform(margin, self.width - margin)
            y = random.uniform(margin, self.height - margin)
            self.world.spawn_ball(x,y,r)

    def run(self):
        # prepare frame dir if recording
        if self.recording and not os.path.exists(FRAME_DIR):
            os.makedirs(FRAME_DIR)
        fps_target = FPS_TARGET
        while self.running:
            t0 = time.perf_counter()
            # events
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    self._on_key(ev.key)
                elif ev.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    if ev.button == 1:
                        # left click spawn small burst
                        for _ in range(12):
                            r = random.randint(3,7)
                            e = self.world.spawn_ball(mx + random.uniform(-8,8), my + random.uniform(-8,8), r)
                            e.vx = random.uniform(-3,3)
                            e.vy = random.uniform(-3,3)
            if not self.paused:
                self.world.step()
            # draw
            self.screen.fill((18,18,24))
            self.renderer.draw(draw_grid=self.draw_grid, draw_hud=self.draw_hud)
            pygame.display.flip()
            # recording
            if self.recording:
                self._record_frame()
            # timing cap
            self.clock.tick(fps_target)
        pygame.quit()

    def _on_key(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_g:
            self.draw_grid = not self.draw_grid
        elif key == pygame.K_h:
            self.draw_hud = not self.draw_hud
        elif key == pygame.K_q:
            self.world.use_spatial_hash = not self.world.use_spatial_hash
            print("Spatial Hash:", self.world.use_spatial_hash)
        elif key == pygame.K_v:
            # toggle integrator
            self.world.integrator = "verlet" if self.world.integrator == "euler" else "euler"
            # ensure prev positions exist for verlet
            if self.world.integrator == "verlet":
                for e in self.world.entities:
                    # approximate prev pos as x - vx
                    self.world.prev_positions[e] = (e.x - e.vx, e.y - e.vy)
            print("Integrator:", self.world.integrator)
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_r:
            self.recording = not self.recording
            if self.recording:
                if not os.path.exists(FRAME_DIR):
                    os.makedirs(FRAME_DIR)
                print("Recording frames to", FRAME_DIR)
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            # spawn more
            for _ in range(100):
                r = random.randint(MIN_RADIUS, MAX_RADIUS)
                x = random.uniform(16 + r, self.width - 16 - r)
                y = random.uniform(16 + r, self.height - 16 - r)
                self.world.spawn_ball(x,y,r)
            print("Spawned +100, total:", len(self.world.entities))
        elif key == pygame.K_MINUS or key == pygame.K_UNDERSCORE:
            # remove 100 oldest (naive)
            toremove = min(100, len(self.world.entities))
            for _ in range(toremove):
                ent = self.world.entities.pop()
                if self.world.use_pool and self.world.pool:
                    self.world.pool.release(ent)
            print("Removed -100, total:", len(self.world.entities))
        elif key == pygame.K_c:
            # clear all and respawn small set
            self._clear_and_respawn(100)
            print("Cleared & respawned 100")
        elif key == pygame.K_s:
            # toggle spatial hash on/off
            self.world.use_spatial_hash = not self.world.use_spatial_hash
            print("Spatial hash:", self.world.use_spatial_hash)
        elif key == pygame.K_t:
            # toggle pool
            self.world.use_pool = not self.world.use_pool
            print("Pool:", self.world.use_pool)

        elif key == pygame.K_UP:   # increase gravity
            self.world.gravity += 0.1
            print("Gravity:", self.world.gravity)

        elif key == pygame.K_DOWN: # decrease gravity
            self.world.gravity -= 0.1
            print("Gravity:", self.world.gravity)

        elif key == pygame.K_t:  # pool toggle
            self.world.use_pool = not self.world.use_pool
            print("Pool:", "On" if self.world.use_pool else "Off")


    def _clear_and_respawn(self, n):
        # clear entities
        self.world.entities.clear()
        if self.world.prev_positions: self.world.prev_positions.clear()
        # spawn n
        for _ in range(n):
            r = random.randint(MIN_RADIUS, MAX_RADIUS)
            x = random.uniform(16 + r, self.width - 16 - r)
            y = random.uniform(16 + r, self.height - 16 - r)
            self.world.spawn_ball(x,y,r)

    def _record_frame(self):
        fname = os.path.join(FRAME_DIR, f"frame_{self.frame_count:06d}.png")
        pygame.image.save(self.screen, fname)
        self.frame_count += 1

# ---------------------------
# ======== ENTRYPOINT =======
# ---------------------------
def main():
    print("Ultimate Simulation Engine")
    print("Hotkeys: G grid, H hud, V integrator toggle, P pause, R record, +/- spawn, C clear, Esc quit")
    eng = Engine()
    eng.run()

if __name__ == "__main__":
    main()
