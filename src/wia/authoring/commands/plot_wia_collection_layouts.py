"""Plot authored collection locations over installed semantic room maps.

Run: behavior/bin/python -B scripts/plot_wia_collection_layouts.py
These CPU figures show design locations, not executed navigation paths.


Canonical invocation: python -m wia.authoring.commands.plot_wia_collection_layouts
"""
import json
from pathlib import Path
import sys

import numpy as np

from wia.resources import ROOT
from wia.authoring.commands._inputs import ARTIFACT_ROOT
from wia.authoring.commands.design_wia_collection_round import _static_map, static_room


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    design = ROOT / 'explore_to_achieve/wia_v1/furnished_v1/collection_selected_design.json'
    if not design.exists():
        design = design.with_name('collection_design_v4.json')
    entries = json.loads(design.read_text())['entries']
    output = ARTIFACT_ROOT / 'wia_v1_20261024_furnished/layout_figures'
    output.mkdir(exist_ok=True)
    for entry in entries:
        layout, alias = entry['layout'], entry['document_alias']
        rooms, names = _static_map(layout['scene_model'])
        size = rooms.shape[0]
        targets = [row for row in layout['native_inventory'] + layout['added_inventory'] if row['role'] == 'target']
        active_rooms = {static_room(layout['scene_model'], row['position'][:2], .1) for row in targets}
        image = np.ones((*rooms.shape, 3))
        image[rooms != 0] = [.88, .89, .91]
        for number, name in names.items():
            if name in active_rooms:
                image[rooms == number] = [.76, .85, .94]
        boundary = np.zeros(rooms.shape, dtype=bool)
        boundary[1:, :] |= rooms[1:, :] != rooms[:-1, :]
        boundary[:, 1:] |= rooms[:, 1:] != rooms[:, :-1]
        image[boundary & (rooms != 0)] = [.55, .59, .64]
        fig, (ax, text) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1.2, 1]})
        extent = [-size * .05, size * .05, -size * .05, size * .05]
        ax.imshow(image, origin='lower', extent=extent, interpolation='nearest')
        rows, cols = np.nonzero(rooms)
        ax.set_xlim(cols.min() * .1 - size * .05 - 1, cols.max() * .1 - size * .05 + 1)
        ax.set_ylim(rows.min() * .1 - size * .05 - 1, rows.max() * .1 - size * .05 + 1)
        for number, row in enumerate(targets, 1):
            x, y = row['position'][:2]
            ax.scatter(x, y, s=180, c='#164b80', edgecolor='white', zorder=3)
            ax.text(x, y, str(number), ha='center', va='center', fontsize=9, color='white', zorder=4)
        start = layout['robot_start']['position']
        ax.scatter(*start[:2], s=130, marker='*', c='#bd6500', edgecolor='white', zorder=5)
        ax.set_aspect('equal'); ax.set_xlabel('World X (m)'); ax.set_ylabel('World Y (m)')
        ax.set_title(layout['scene_model'])
        lines = [alias + ' | authored collection locations', '', 'Star: initial robot position',
                 'Blue: physical rooms containing targets', '', 'Targets' + (' in preserved visit order:' if layout.get('route') else ':')]
        for number, row in enumerate(targets, 1):
            lines.extend([f"{number}. {row['room']}", '   ' + row['public_name']])
        if alias == 'T07':
            lines += ['', 'Northern storage: raw room ID 11.',
                      'Route name uses the original 1 m map;',
                      'this 0.1 m physical map names it storage_2.']
        lines += ['', 'Open surfaces; no deliberate concealment.', '',
                  'Static semantic map and proposed positions.', 'No path, grasp or reachability trial has run.',
                  'Native initialization images are separate evidence.']
        text.axis('off'); text.text(0, 1, '\n'.join(lines), va='top', fontsize=10, linespacing=1.4)
        fig.tight_layout(); fig.savefig(output / (alias.lower() + '_planned_locations.png'), dpi=150); plt.close(fig)
    print('Wrote five static design figures:', output)


if __name__ == '__main__':
    main()
