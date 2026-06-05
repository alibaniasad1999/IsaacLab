# My Experiments

Personal scripts and learning experiments built on top of Isaac Lab. Kept outside `scripts/tutorials/` so upstream pulls from `isaac-sim/IsaacLab` don't conflict with my changes.

## Scripts

| File | What it does |
|---|---|
| [cartpole_position.py](cartpole_position.py) | Runs the cartpole RL env, applies a simple proportional force on the cart, and prints the cart's slider position (min/max per episode). Starting point for understanding observations + actions. |

## Run

From the IsaacLab repo root, with the conda env active:

```bash
conda activate env_isaaclab
./isaaclab.sh -p scripts/my_experiments/cartpole_position.py --num_envs 4
```

Add `--headless` if no display is available.

## Cartpole observation layout

`obs["policy"]` is a tensor of shape `(num_envs, 4)`:

| Index | Meaning | Units |
|---|---|---|
| 0 | Cart position (slider joint) | m |
| 1 | Pole angle | rad |
| 2 | Cart velocity | m/s |
| 3 | Pole angular velocity | rad/s |

Action is a 1-D effort applied to the slider joint.

## Offline asset setup (low-bandwidth)

Isaac Sim normally streams USD assets from S3. To run without internet, assets are mirrored locally and Isaac Sim is pointed at the local root.

**Local asset root:** `~/isaac_assets/Assets/Isaac/5.1`

**Currently mirrored:**
- `Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usd`
- `Isaac/IsaacLab/Robots/Classic/Cartpole/Props/instanceable_meshes.usd`
- `Isaac/Environments/Grid/default_environment.usd`
- `Isaac/Environments/Grid/Materials/Textures/*.png`

**Config patched** (Kit user.config.json):
`~/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/kit/data/Kit/Isaac-Sim/5.1/user.config.json`

Keys under `persistent.isaac.asset_root`:
```json
{
  "default": "/home/ali/isaac_assets/Assets/Isaac/5.1",
  "cloud":   "/home/ali/isaac_assets/Assets/Isaac/5.1",
  "nvidia":  "/home/ali/isaac_assets/Assets/Isaac/5.1"
}
```
A `.bak` of the original lives next to the patched file.

### Adding a new asset

When a script raises `FileNotFoundError: Unable to open the usd file at path: /home/ali/isaac_assets/.../FOO.usd`:

1. List the matching folder on S3:
   ```bash
   curl -s "https://omniverse-content-production.s3-us-west-2.amazonaws.com/?prefix=Assets/Isaac/5.1/<RELATIVE_DIR>/" | grep -oE '<Key>[^<]+</Key>'
   ```
2. Download with resume support:
   ```bash
   mkdir -p ~/isaac_assets/Assets/Isaac/5.1/<RELATIVE_DIR>
   cd ~/isaac_assets/Assets/Isaac/5.1/<RELATIVE_DIR>
   wget -c "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/<RELATIVE_DIR>/FOO.usd"
   ```
3. Re-run the script. Repeat until no more missing assets.

## Hardware

- CPU: Intel Core i5-14400F
- GPU: NVIDIA RTX 3080 (10 GB VRAM)
- Isaac Lab branch: `main`
- Isaac Sim: 5.1
