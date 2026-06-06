# Running Isaac Lab on Ubuntu Remotely from Mac

## Your Setup
- **Mac**: Your local machine in Iran
- **Ubuntu 22.04**: At university in Iran, RTX 3080, Isaac Lab already installed
- **Goal**: Run Isaac Lab on Ubuntu, see the simulation on your Mac

---

## Part 1: Setup Ubuntu for Remote Access (at university, do once)

Go to the Ubuntu machine (you have a monitor there).

### 1.1 — Find your Ubuntu's IP address

Open a terminal on Ubuntu and run:

```bash
ip addr show
```

Look for the IP address under your network adapter (usually `eth0` or `enp...`).
It will look like `192.168.x.x` or `10.x.x.x`. Write this down.

Or simpler:

```bash
hostname -I
```

The first IP shown is what you need. **Write it down.**

### 1.2 — Install and enable SSH server

```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl enable ssh
sudo systemctl start ssh
```

Verify it's running:

```bash
sudo systemctl status ssh
```

You should see "active (running)".

### 1.3 — Install tmux

```bash
sudo apt install tmux
```

### 1.4 — Note your Ubuntu username

```bash
whoami
```

**Write this down too.** You now have two things:
- Username (e.g., `ali`)
- IP address (e.g., `192.168.1.50`)

### 1.5 — Test SSH locally

On the Ubuntu machine itself, test that SSH works:

```bash
ssh localhost
```

Type `yes` if prompted, enter your password. If you get a shell, SSH works. Type `exit`.

### 1.6 — Install Tailscale (for access from home)

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

It will print a URL. Open it in the browser, create a free Tailscale account (use Google/GitHub login), and authorize the machine.

Then get your Tailscale IP:

```bash
tailscale ip
```

**Write down the Tailscale IP** (looks like `100.x.x.x`). You'll use this from home.

---

## Part 2: Setup Your Mac (do once)

### 2.1 — Install Tailscale on Mac

Open Terminal on your Mac:

```bash
brew install tailscale
```

Or install "Tailscale" from the Mac App Store (easier).

Log in with the **same account** you used on Ubuntu.

### 2.2 — Verify connection

With Tailscale running on both machines:

```bash
ping 100.x.x.x    # the Tailscale IP from step 1.6
```

If it responds, you're connected.

### 2.3 — Test SSH from Mac to Ubuntu

**If at university (same network):**

```bash
ssh username@192.168.x.x
```

**If at home (through Tailscale + VPN):**

1. Connect ExpressVPN or X-VPN first
2. Make sure Tailscale is running
3. Then:

```bash
ssh username@100.x.x.x
```

Enter your Ubuntu password. If you see the Ubuntu terminal, everything works. Type `exit`.

---

## Part 3: Running Isaac Lab Remotely (every time)

### Step 1 — Connect to the internet

**If at university:** Just connect to the university WiFi/LAN. No VPN needed.

**If at home:**
1. Open ExpressVPN or X-VPN and connect
2. Make sure Tailscale is running on your Mac

### Step 2 — SSH into Ubuntu with video tunnel

Open Terminal on your Mac and run:

**From university (same network):**
```bash
ssh -L 8211:localhost:8211 username@192.168.x.x
```

**From home:**
```bash
ssh -L 8211:localhost:8211 username@100.x.x.x
```

Enter your password.

### Step 3 — Start tmux

```bash
tmux new -s sim
```

This keeps your simulation running even if SSH disconnects.

### Step 4 — Start Isaac Lab with livestream

```bash
conda activate env_isaaclab
cd /path/to/IsaacLab
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --livestream 2
```

Wait until it's fully loaded (first time takes a few minutes).

### Step 5 — Watch the simulation

Open Chrome or Safari on your Mac and go to:

```
http://localhost:8211/streaming/webrtc-client
```

You should see the Isaac Sim viewport live.

### Step 6 — Running your own scripts

Replace the script path with your own:

```bash
./isaaclab.sh -p your_script.py --livestream 2
```

### Step 7 — When you're done

- `Ctrl+C` to stop the simulation
- Type `exit` to close SSH

Or to leave it running in the background:
- Press `Ctrl+B` then `D` to detach tmux
- Type `exit` to close SSH
- The simulation keeps running on Ubuntu

### Step 8 — Reconnecting later

```bash
ssh -L 8211:localhost:8211 username@<ip>
tmux attach -t sim
```

Your simulation is still running. Open the browser URL again to view it.

---

## Quick Cheat Sheet

| What | Command |
|------|---------|
| SSH + tunnel (university) | `ssh -L 8211:localhost:8211 user@192.168.x.x` |
| SSH + tunnel (home) | `ssh -L 8211:localhost:8211 user@100.x.x.x` |
| Start tmux | `tmux new -s sim` |
| Reattach tmux | `tmux attach -t sim` |
| Activate env | `conda activate env_isaaclab` |
| Run with stream | `./isaaclab.sh -p script.py --livestream 2` |
| Watch in browser | `http://localhost:8211/streaming/webrtc-client` |
| Stop simulation | `Ctrl+C` |
| Detach tmux | `Ctrl+B` then `D` |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| SSH connection refused | Make sure `sudo systemctl status ssh` shows "active" on Ubuntu |
| SSH timeout from home | Check VPN is connected, Tailscale is running on both machines |
| Browser shows blank page | Make sure `--livestream 2` was used and SSH tunnel (`-L 8211:...`) is active |
| Livestream laggy | This is normal over VPN. For training, use `--headless` and only livestream when you need to watch |
| Simulation crashes (out of memory) | RTX 3080 has 10GB VRAM. Reduce number of environments or use simpler scenes |
| "Port 8211 in use" error | Kill old processes: `sudo fuser -k 8211/tcp` then try again |
| SSH drops frequently | Reconnect and `tmux attach -t sim` — your simulation is still running |
| Tailscale blocked in Iran | Connect ExpressVPN/X-VPN first, then start Tailscale |
