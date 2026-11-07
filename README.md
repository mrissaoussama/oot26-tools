# OOT Remake collision + terrain viewer + CE Table

Tools for the Ocarina of Time remake (Switch 2) running in a Switch emulator (lichii, eden and other forks):

- **Cheat Engine table**: Link's position (writable, teleport), health, velocity, facing, game camera, moonjump and a speed multiplier. Addresses are found automatically each run and after zone changes.
- **Collision viewer**: draws the terrain collision, invisible walls, void-out zones and Link in 3D, live, with game camera, orbit and top-down views.

![Game camera view](screenshots/game_camera.png)

![Top-down view](screenshots/top_down.png)

## Download

Get `ootr-tools.zip` from the [latest release](../../releases/latest).

## Requirements

- Windows, Cheat Engine 7.5+, Python 3.10+ (python.org, tick "Add to PATH").
- Optional: the game's romfs dumped from your emulator (right-click the game > Dump RomFS). Only needed to draw terrain. Walls and void zones work without it.

## Cheat Engine

1. Start the game in the emulator.
2. Open Cheat Engine and attach to the emulator process.
3. Open `oot_link_tools.CT` and tick **Link tools**. It finds Link, health and camera, and opens the `cepipe` server for the viewer.
4. Optional: tick **Moonjump** (Space up, Ctrl down) or **Speed multiplier**. Double-click those scripts to change keys or values.

If the viewer says `cepipe not found`, open Cheat Engine's Lua engine (Memory View > Tools > Lua Engine), paste this and press Execute:

```lua
openLuaServer("cepipe")
```

## Viewer

1. Keep Cheat Engine running with **Link tools** ticked.
2. Double-click `Run viewer.bat`. The first run installs the Python packages (numpy, zstandard, PySide6).
3. For terrain, click **Set romfs folder (for terrain)** and pick the dumped romfs folder.
4. Views: Game camera, Orbit (drag to rotate, wheel to zoom), Top-down (wheel to zoom). The checkboxes choose what is drawn.

| Colour | Meaning |
|---|---|
| Grey | Terrain / visible collision |
| Red | Invisible walls that block Link |
| Orange | Climb-only walls |
| Blue | Push walls (doors etc.) |
| Purple | Void-out / fall-down zones |
| Teal | No-respawn areas |

After a game update, delete `viewer/geom.json`; the viewer rebuilds it from the romfs.

## Notes

- Offsets match the current game version; a game update may need new ones.
- Dungeon terrain is not drawn yet (dungeons store collision per room); invisible walls and zones still are.
