import ctypes
import time
import os
import math
import random
import pickle
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import pygetwindow as gw
import win32gui
import win32con
import pydirectinput
import pyautogui
from PIL import ImageGrab
from collections import namedtuple, deque

# DPI-Awareness für korrekte Koordinaten
ctypes.windll.shcore.SetProcessDpiAwareness(1)

# Setup der Variablen und Pfade
window_title = "Microsoft Flight Simulator"
pixel_pos_in_window = (130, 1095)
POS_WELTKARTE = (1496, 481)
POS_LOSFLIEGEN = (1766, 1141)
FLT_PATH = b"C:\\Users\\nickl\\AppData\\Local\\Packages\\Microsoft.FlightSimulator_8wekyb3d8bbwe\\LocalState\\ai_new.FLT"
MSFS_LAUNCH_CMD = r'cmd.exe /C start shell:AppsFolder\Microsoft.FlightSimulator_8wekyb3d8bbwe!App -FastLaunch'

# Parameter
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 10000#20000 
TAU = 0.005
LR = 3e-4
num_episodes = 2000
max_steps = 1000

# Rotate Speed
vr = 145

TEST_MODE = False
NUM_TEST_EPISODES = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verbindung zur SimConnect SDK von Microsoft
os.add_dll_directory(r"C:\MSFS SDK\SimConnect SDK\lib")
dll_path = "./Simbridge.dll"
dll = ctypes.CDLL(dll_path)

class AircraftState(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("pitch", ctypes.c_double), 
        ("bank", ctypes.c_double),
        ("altitude", ctypes.c_double), 
        ("airspeed", ctypes.c_double),
        ("vs", ctypes.c_double), 
        ("heading", ctypes.c_double),
        ("alt_above_ground", ctypes.c_double),
        ("sim_on_ground", ctypes.c_double),
        ("runway_x", ctypes.c_double),
        ("runway_y", ctypes.c_double),
        ("runway_z", ctypes.c_double)
    ]

dll.sim_init.restype = ctypes.c_int
dll.get_state.argtypes = [ctypes.POINTER(AircraftState), ctypes.POINTER(ctypes.c_int)]
dll.set_controls.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
dll.reset_flight.argtypes = [ctypes.c_char_p]
dll.send_event.argtypes = [ctypes.c_int, ctypes.c_uint32]

# Observer prüft, ob MSFS2020 gerade läuft oder neu gestartet werden muss (und Navigation im MSFS Haupmenü)
def find_msfs_hwnd():
    hwnd_list = []
    def callback(hwnd, extra):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if window_title in title:
                hwnd_list.append(hwnd)
    win32gui.EnumWindows(callback, None)
    return hwnd_list[0] if hwnd_list else None

def force_focus():
    hwnd = find_msfs_hwnd()
    if hwnd:
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        try:
            win32gui.SetForegroundWindow(hwnd)
        except:
            pyautogui.press('win')
            time.sleep(0.5)
            pyautogui.press('win')
            win32gui.SetForegroundWindow(hwnd)
        time.sleep(2)
        return True
    return False

# Click etwas komplexer, damit das mit dem MSFS Fenster funktioniert
def click(pos, clicks=4):
    pydirectinput.moveTo(pos[0], pos[1], duration=0.8)
    time.sleep(0.5)
    for _ in range(clicks):
        pydirectinput.mouseDown()
        time.sleep(0.1)
        pydirectinput.mouseUp()
        time.sleep(0.15)

# Checkt die Pixel Farbe am Anfang des Ladebildschirms und wartet bis dieser verschwunden ist
def wait_for_loading_in_window(pixel_pos):
    loading_color = (0, 177, 250)
    print("[Observer] Prüfe Ladevorgang...")
    time.sleep(5)
    while True:
        hwnd = find_msfs_hwnd()
        if not hwnd: break
        try:
            rect = win32gui.GetWindowRect(hwnd)
            screenshot = ImageGrab.grab(bbox=rect)
            pixel_color = screenshot.getpixel(pixel_pos)
            if all(abs(pixel_color[i] - loading_color[i]) <= 15 for i in range(3)):
                time.sleep(1.5)
            else: break
        except:
            time.sleep(2)
    print("[Observer] Ladevorgang abgeschlossen!")

# Stoppt MSFS, da es ca. alle 13 Epochen Probleme mit der FBW A320 Steuerung gibt
def kill_msfs():
    print("[Observer] Schließe Verbindung und MSFS...")
    try:
        dll.sim_exit()
    except:
        print("[Observer] Couldn't close Sim Connection")
        pass
    os.system("taskkill /f /im FlightSimulator.exe")
    time.sleep(15)

# RL Environment
class TakeoffFlightEnv:
    def __init__(self, dll):
        self.dll = dll
        self.state_buffer = AircraftState()
        # Agent hat 7 mögliche Aktionen
        self.action_space = 7
        # 11 Parameter werden vom Agenten eingesehen
        self.observation_size = 11
        self.speed_history = deque(maxlen=15)
        self.reset_controls()

    def reset_controls(self):
        self.curr_aileron, self.curr_elevator, self.curr_rudder, self.curr_throttle = 0.0, 0.0, 0.0, 0.0

    # Überprüft, ob der Simulator überhaupt läuft
    def ensure_sim_ready(self, force_restart=False):
        hwnd = find_msfs_hwnd()
        if not hwnd or force_restart:
            if force_restart: kill_msfs()
            print("[Observer] Starte MSFS neu...")
            subprocess.Popen(MSFS_LAUNCH_CMD, shell=True)
            while not find_msfs_hwnd():
                time.sleep(5)
            # Dauert ca. 60 Sekunden aber sicherheitshalber ist ein Puffer von 20 Sekunden eingebaut, falls es länger braucht
            print("[Observer] Initialisierung (80s)...")
            time.sleep(80)
            # Navigiert eigenständig durch das Menü, um den Flug zu laden
            force_focus()
            click(POS_WELTKARTE)
            time.sleep(5)
            self.dll.sim_init()
            time.sleep(5)
            self.dll.reset_flight(FLT_PATH)
            time.sleep(3)
            force_focus()
            click(POS_LOSFLIEGEN)
            wait_for_loading_in_window(pixel_pos_in_window)
            return False
        return True

    def reset(self, epoch_count=0):
        # Alle 10 Epochen harter Neustart, da bald die Verbindung zur Steuerung abbrechen würde
        if epoch_count > 0 and epoch_count % 10 == 0:
            print(f"[Observer] Geplanter Neustart in Epoche {epoch_count}")
            self.ensure_sim_ready(force_restart=True)
        else:
            self.ensure_sim_ready()

        # Damit das Flugzeug nicht mit alten Control states startet, werden diese resettet
        self.reset_controls()
        set_airplane_controls(0, 0, 0, 0)
        self.speed_history.clear()
        
        try:
            self.dll.reset_flight(FLT_PATH)
            wait_for_loading_in_window(pixel_pos_in_window)
        except:
            self.ensure_sim_ready(force_restart=True)

        for _ in range(20):
            self.dll.get_state(ctypes.byref(self.state_buffer), None, None)
            if self.state_buffer.sim_on_ground > 0.5: break
            time.sleep(0.1)
        
        raw_hdg = self.state_buffer.heading
        self.target_heading = float(math.ceil(raw_hdg / 10.0) * 10) % 360
        self.dll.send_event(5, 0) # Löst die Parkbremse automatisch
        return np.array(self._get_state(), dtype=np.float32), {}

    def _get_state(self):
        self.dll.get_state(ctypes.byref(self.state_buffer), None, None)
        s = self.state_buffer
        diff_hdg = (s.heading - self.target_heading + 180) % 360 - 180
        return [
            np.clip(s.bank / 45.0, -1.0, 1.0), np.clip(s.pitch / 25.0, -1.0, 1.0),
            np.clip(s.airspeed / 200.0, 0.0, 1.0), np.clip(s.vs / 3000.0, -1.0, 1.0),
            np.clip(s.alt_above_ground / 1000.0, 0.0, 1.0),
            1.0 if s.sim_on_ground > 0.5 else 0.0,
            np.clip(diff_hdg / 20.0, -1.0, 1.0),
            np.clip((s.airspeed - vr) / 40.0, -1.0, 1.0),
            self.curr_elevator,
            np.clip(s.runway_x / 20.0, -1.0, 1.0),
            np.clip(s.runway_y / 10.0, 0.0, 1.0)
        ]

    def step(self, action, steps, epoch):
        self._apply_action(action)
        time.sleep(0.05)
        next_obs = self._get_state()
        reward = self._compute_reward(next_obs, action, steps)
        done, reason = self._check_done(next_obs, steps, epoch)
        return np.array(next_obs, dtype=np.float32), reward, done, reason, {}

    def _apply_action(self, action):
        spd = self.state_buffer.airspeed
        # Throttle wird automatisch auf Fullspeed gesetzt, um den Action Space geringer zu halten
        self.curr_throttle = 1.0
        
        centering_factor = 0.85 
        self.curr_rudder *= centering_factor
        self.curr_aileron *= centering_factor
    
        r_step = 0.05 if self.state_buffer.sim_on_ground > 0.5 else 0.025
        
        e_step = 0.05
        a_step = 0.05
        
        if action == 0: self.curr_aileron -= a_step
        elif action == 1: self.curr_aileron += a_step
        elif action == 2: self.curr_elevator -= e_step
        elif action == 3: self.curr_elevator += e_step
        elif action == 4: self.curr_rudder -= r_step
        elif action == 5: self.curr_rudder += r_step
        elif action == 6: self.curr_rudder *= 0.
        
        self.curr_elevator = np.clip(self.curr_elevator, -1.0, 1.0)
        self.curr_rudder = np.clip(self.curr_rudder, -1.0, 1.0)
        self.curr_aileron = np.clip(self.curr_aileron, -1.0, 1.0)
        set_airplane_controls(self.curr_aileron, self.curr_elevator, self.curr_rudder, self.curr_throttle)

    def _compute_reward(self, s, action, step_count):
        # Mapping 
        bank_norm      = s[0]
        pitch_norm     = s[1]
        speed_norm     = s[2]
        vs_norm        = s[3]
        alt_g_norm     = s[4]
        on_ground      = s[5] > 0.5
        hdg_err_norm   = s[6]
        curr_elev      = s[8]
        runway_x_norm  = s[9] 
        runway_y_norm  = s[10]

        true_speed   = speed_norm * 200.0
        true_x_error = abs(runway_x_norm * 20.0)
        true_hdg_err = abs(hdg_err_norm * 20.0)
        true_vs      = vs_norm * 3000.0
        
        reward = 0.0

        # Basisrewards
        # Belohnung/Bestrafung für (nicht) halten der Centerline
        reward += 30.0 * math.exp(-(true_x_error**2) / 15.0) 
        reward -= (true_x_error ** 2) * 0.5 # Progressive Strafe für Abweichen von der Centerline

        # Richtung des Flugzeugs
        reward -= true_hdg_err * 0.5

        # Belohnungen und Bestrafungen auf dem Boden
        if on_ground:
            # Zu starkes Rudder bewegen soll bestraft werden
            reward -= (abs(self.curr_rudder) ** 2) * 15.0
            
            # Wenn die Geschwindigkeit höher wird und die Abweichung der Centerline gering bleibt, wird Agent belohnt
            if true_x_error < 12.0:
                reward += (speed_norm * 10.0)

            # Heading ist am Boden kritischer als X-Error (falsches Heading führt zu X-Error)
            if true_hdg_err < 2.0:
                reward += 10.0
            else:
                reward -= true_hdg_err * 2.0
            
            # Bestrafung für Elevator Ziehen unter Vrotate
            if true_speed < (vr - 5):
                if curr_elev < -0.1: 
                    penalty_factor = (vr - true_speed) / vr 
                    reward -= 60.0 * penalty_factor
                # Belohnung für leichtes Drücken (darf aber nicht zu stark sein)
                elif curr_elev > 0.05:
                    reward += 10.0 
            
            # Bei genug Geschwindigkeit:
            else:
                if curr_elev < -0.3:
                    reward += 150.0 # Starker Bonus für's Nase hochziehen
                elif curr_elev > 0.0:
                    reward -= 50.0  # Der Agent darf nicht mehr die Nase nach unten drücken

        # Belohnungen und Bestrafungen in der Luft
        else:
            # Bei Vrotate in der Luft
            if true_speed >= (vr - 5):
                reward += 100.0
            # Deutlich unter Vrotate in der Luft
            elif true_speed < (vr - 20):
                reward -= 300.0
            # Alle anderen Optionen
            else:
                reward -= 50
            # In der Luft soll die KI das Rudder nicht mehr benutzen, da sonst die Stabilität leidet (müsste für Fälle wie Triebwerksausfall noch besser durchdacht werden)
            reward -= abs(self.curr_rudder) * 80.0 

            # Vertical Speed soll zwichen 1000 und 2500 fpm liegen
            if 1000 <= true_vs <= 2500:
                reward += 50.0
            elif true_vs < 500: # Zu flach oder die KI sinkt
                reward -= 50.0
            elif true_vs > 3500: # Zu steil -> Stallgefahr
                reward -= 50.0

            # Pitch soll in einem bestimmten Bereich gehalten werden
            true_pitch = pitch_norm * 25.0
            if 7.0 <= true_pitch <= 18.0:
                reward += 30.0
            
            # Bank sollte gerade bleiben, damit die Triebwerke nicht direkt nach dem Start mit dem Boden kollidieren
            reward -= abs(bank_norm) * 50.0
            
            # Heading soll gehalten werden
            reward -= true_hdg_err * 2.0

        # Terminal Reward
        if not on_ground and runway_y_norm > 1.5:
            if true_hdg_err < 8.0 and true_vs > 800:
                reward += 2000.0

        return reward
    
    def _check_done(self, s, steps, epoch):
        # Mapping der realen Werte
        spd          = s[2] * 200.0
        alt_g        = s[4] * 1000.0
        on_ground    = s[5] > 0.5
        hdg_err      = abs((self.state_buffer.heading - self.target_heading + 180) % 360 - 180)
        true_x_error = abs(s[9] * 20.0)  # Seitliche Abweichung in Metern
        
        # Curriculum Learning
        x_tol = max(15.0, 30.0 - (epoch // 50) * 2.0)
        
        # Um später zu checken, ob das Flugzeug feststeckt
        self.speed_history.append(spd)

        # Minimum an Schritten vergehen lassen
        if steps > 20:
            # Runway wurde verlassen
            if on_ground and true_x_error > x_tol:
                return True, f"Left Runway by ({true_x_error:.1f}m)"

            # Zu tief
            if not on_ground and alt_g < 3: 
                return True, "Crashed / too low"

            # Heading wird nicht befolgt
            if hdg_err > 40: 
                return True, "Course lost (Heading)"

            # Zu steile Seitenneigung
            if abs(s[0] * 45.0) > 60: 
                return True, "Unnormal Bank"
            
            # Zu früh abgehoben
            if not on_ground and spd < (vr - 15.0) and alt_g > 2.0:
                return True, f"Dangerous premature Takeoff ({spd:.1f} kts < {vr-15:.0f})"

            # Ziel erreicht
            if alt_g > 1500: 
                return True, "SUCCESS (Altitude reached)"

        # Timeout
        if steps >= 850: 
            return True, "Timeout"

        # Geschwindigkeit erhöht sich nicht
        if on_ground and steps > 150 and len(self.speed_history) == self.speed_history.maxlen:
            if max(self.speed_history) - min(self.speed_history) < 0.5: 
                return True, "Stagnation (No speed increase)"

        return False, ""
    

# Hauptfunktionen des Agenten
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args): self.memory.append(Transition(*args))
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_observations, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    def forward(self, x): return self.net(x)

def set_airplane_controls(aileron, elevator, rudder, throttle):
    dll.set_controls(ctypes.c_float(aileron), ctypes.c_float(elevator), 
                     ctypes.c_float(rudder), ctypes.c_float(throttle))

env = TakeoffFlightEnv(dll)
policy_net = DQN(env.observation_size, env.action_space).to(device)
target_net = DQN(env.observation_size, env.action_space).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.AdamW(policy_net.parameters(), lr=LR)
memory = ReplayMemory(10000)

steps_done, start_epoch, best_avg_reward = 0, 0, -float('inf')
rewards_history = []
success_window = deque(maxlen=10)

# Checkpoint laden
checkpoint_path = "latest_checkpoint.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['episode'] + 1
    # Zum weitertrainieren am besten erstmal wieder erhöhen
    steps_done  =  checkpoint['steps_done']
    print(f"Resuming from epoch {start_epoch}")

def select_action(state):
    global steps_done
    eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps:
        with torch.no_grad(): return policy_net(state).max(1).indices.view(1, 1)
    return torch.tensor([[random.randrange(env.action_space)]], device=device, dtype=torch.long)

def select_action_test(state):
    with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)


def optimize_model():
    if len(memory) < BATCH_SIZE: return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    non_final_mask = torch.tensor([s is not None for s in batch.next_state], device=device, dtype=torch.bool)
    non_final_next = torch.cat([s for s in batch.next_state if s is not None])

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next).max(1).values

    expected = (next_state_values * GAMMA) + reward_batch
    loss = nn.SmoothL1Loss()(state_action_values, expected.unsqueeze(1))
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# Funktioniert noch nicht gut
def run_test_mode():
    print("*** Test Modus ***")

    results = []
    
    if os.path.exists("latest_checkpoint.pth"):
        checkpoint = torch.load("latest_checkpoint.pth", map_location=device)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        policy_net.eval()
        print("Loaded latest_checkpoint.pth for evaluation")
    else:
        print("Warning: latest_checkpoint.pth not found, using current weights")

    for episode in range(NUM_TEST_EPISODES):
        state_np, _ = env.reset(epoch_count=0)
        state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)

        ep_reward = 0
        for steps in range(max_steps):
            action = select_action_test(state)
            obs, reward_val, done, reason, _ = env.step(action.item(), steps, episode)

            ep_reward += reward_val

            if done:
                print(f"Episode {episode} finished: {reason} | Reward={ep_reward:.1f}")
                results.append((ep_reward, reason))
                break

            state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    print("\n=== TEST SUMMARY ===")
    avg = np.mean([r[0] for r in results])
    print(f"Average reward: {avg:.1f}")
    print(f"Successes: {sum('SUCCESS' in r[1] for r in results)} / {len(results)}")


# Haupttrainigsschleife
if TEST_MODE:
    run_test_mode()
    raise SystemExit
else:
    print("*** Observer & Reinforcement Learning Training ***")

    ep_hdg_errors = []
    ep_elevator_inputs = []
    done_reason = "Timeout"

    for epoch in range(start_epoch, num_episodes):
        try:
            state_np, _ = env.reset(epoch_count=epoch)
            state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            
            ep_hdg_errors = []
            done_reason = "Timeout"
            
            for steps in range(max_steps):
                action = select_action(state)
                obs, reward_val, done, done_reason, _ = env.step(action.item(), steps, epoch)
                
                current_hdg_err = abs((env.state_buffer.heading - env.target_heading + 180) % 360 - 180)
                ep_hdg_errors.append(current_hdg_err)
                ep_elevator_inputs.append(env.curr_elevator)
                
                reward = torch.tensor([reward_val], device=device)
                next_state = None if done else torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                
                memory.push(state, action, next_state, reward)
                state = next_state
                optimize_model()
                
                # Soft Update Target Net
                target_dict, policy_dict = target_net.state_dict(), policy_net.state_dict()
                for key in policy_dict:
                    target_dict[key] = policy_dict[key] * TAU + target_dict[key] * (1 - TAU)
                target_net.load_state_dict(target_dict)

                if done:
                    rewards_history.append(reward_val)
                    max_spd, max_alt = env.state_buffer.airspeed, env.state_buffer.alt_above_ground
                    avg_reward = np.mean(rewards_history[-10:])
                    current_eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
                    str_reason = str(done_reason)
                    
                    print(f"\n--- EPISODE {epoch} SUMMARY ---")
                    status_text = '[SUCCESS]' if "SUCCESS" in str_reason else '[FAILED]'
                    print(f"Status:      {status_text}")
                    print(f"Reason:      {done_reason}")
                    print(f"Final Rew:   {reward_val:.1f} (Avg10: {avg_reward:.1f})")
                    # Nur berechnen, wenn Liste nicht leer ist
                    if ep_hdg_errors:
                        print(f"Avg Hdg Err: {np.mean(ep_hdg_errors):.2f}°")
                        print(f"Avg Elev:    {np.mean(ep_elevator_inputs):.2f}")
                    print(f"Spd/Alt:     {max_spd:.1f} kts / {max_alt:.0f} ft")
                    print(f"Epsilon:     {current_eps:.4f}")
                    print("-" * 30)
                        
                    # CSV mit Trainingsdaten
                    if not os.path.exists("training_stats.csv"):
                        with open("training_stats.csv", "w") as f:
                            f.write("Epoch;Status;Reason;AvgHdgErr;MaxSpd;MaxAlt;Epsilon\n")

                    with open("training_stats.csv", "a") as f:
                        f.write(f"{epoch};{status_text};{done_reason};{np.mean(ep_hdg_errors):.2f};{max_spd:.1f};{max_alt:.0f};{current_eps:.4f}\n")


                    is_success = "SUCCESS" in str_reason
                    success_window.append(is_success)
                    
                    if len(success_window) == 10 and not any(success_window):
                        if steps_done > 15000:
                            print("*** NO SUCCESS IN 10 EPS. Resetting Exploration (Steps Done)...")
                            steps_done = 7000
                    
                    # Checkpoints
                    ckpt = {'episode': epoch, 'policy_net_state_dict': policy_net.state_dict(),
                            'target_net_state_dict': target_net.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                            'steps_done': steps_done}
                    torch.save(ckpt, "latest_checkpoint.pth")
                    
                    if epoch % 10 == 0:
                        torch.save(ckpt, f"checkpoint_episode_{epoch}.pth")
                        with open("memory.pkl", "wb") as f: pickle.dump(memory, f)
                    
                    if avg_reward > best_avg_reward and len(rewards_history) >= 10:
                        best_avg_reward = avg_reward
                        torch.save(policy_net.state_dict(), "best_model.pth")
                    
                    break

        except Exception as e:
            print(f"Fehler: {e}")
            time.sleep(10)