#!/usr/bin/env python3
"""Standalone scientific figures from saved milestone 8 traces (no simulator)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    root, out = Path(data['root']), args.results.parent/'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    def save(fig, name):
        fig.savefig(out/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(out/(name+'.svg'), bbox_inches='tight')
        plt.close(fig)
    episodes = data['episodes']
    if episodes:
        labels = [e['run'] for e in episodes]
        wait_wall = [e['world_timing'].get('model_wait_wall_time_s', 0) for e in episodes]
        wait_sim = [e['world_timing'].get('simulation_time_during_model_wait_s', 0) for e in episodes]
        fig, axs = plt.subplots(1, 2, figsize=(12, max(3, len(episodes)*.42)), sharey=True)
        colors = ['#2b6d9a' if e['world_mode']=='run' else '#bd6a2e' for e in episodes]
        for ax, values, title in zip(axs, (wait_wall, wait_sim), ('Owner waiting: wall time (s)', 'Owner waiting: simulation time (s)')):
            ax.barh(labels, values, color=colors)
            ax.set_title(title)
            ax.grid(axis='x', alpha=.2)
            for i, v in enumerate(values):
                ax.text(v, i, f' {v:.1f}', va='center', fontsize=8)
            ax.set_xlim(0, max(values+[1])*1.18)
        axs[0].invert_yaxis()
        fig.suptitle('Run / pause are separate factors; waiting also includes tools and host preparation')
        fig.tight_layout()
        save(fig, 'world_wait')
        # Native measured base centerline. Geometry is shown for context only;
        # collision-free feasibility is established by the separate native audit.
        exploring = [e for e in episodes if e['variant'] in ('single_table_occlusion', 'multi_region_search')]
        if exploring:
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            for ax, variant, title in zip(axs, ('single_table_occlusion','multi_region_search'), ('Single-table occlusion','Two-region search')):
                matching = [e for e in exploring if e['variant']==variant]
                for e in matching:
                    trace = records(root/e['run']/'private_exploration/physical_trajectory.jsonl')
                    if trace:
                        xy = np.array([r['base_xy_m'] for r in trace])
                        line, = ax.plot(xy[:,0], xy[:,1], label=e['run'].replace('B-single-','').replace('C-multi-','').replace('D-single-',''), lw=1.1)
                        ax.scatter(*xy[0], marker='o', s=24, c=line.get_color())
                        ax.scatter(*xy[-1], marker='x', s=32, c=line.get_color())
                if matching:
                    setup = json.loads((root/matching[0]['run']/'private_scene/setup.json').read_text())
                    low, high = setup['support']['aabb_min'], setup['support']['aabb_max']
                    ax.add_patch(Rectangle(low[:2],high[0]-low[0],high[1]-low[1],fc='#ece6db',ec='#b7aa94',zorder=-2))
                    for screen in setup['occluders']:
                        low, high=screen['aabb_min'],screen['aabb_max']
                        ax.add_patch(Rectangle(low[:2],high[0]-low[0],high[1]-low[1],fc='#506877',zorder=-1))
                    for candidate in setup['configuration']['target_placement_candidates']:
                        ax.scatter(*candidate['position'][:2], marker='+', color='#a51d25',s=75)
                ax.set(title=title,xlabel='World X (m)',ylabel='World Y (m)')
                ax.set_aspect('equal',adjustable='datalim')
                ax.grid(alpha=.2)
                if matching: ax.legend(fontsize=7,loc='best')
            fig.suptitle('Measured base paths; initial furniture outlines; red + = private candidate centers')
            fig.tight_layout()
            save(fig,'base_trajectories')
        fig, ax = plt.subplots(figsize=(12,max(3,len(episodes)*.62)))
        stages=[('first_visible_step','Visible pixels','#2578a8','o'),
                ('first_recognizable_step','Private pixel gate','#d0a03a','P'),
                ('first_approach_step','Approach','#39a178','s'),
                ('first_grasp_step','GRASP','#c88825','^'),('first_attachment_step','Attachment','#905ab5','D'),
                ('first_lift_step','Lift >=5cm','#c94545','*'),
                ('first_target_drop_below_initial_5cm_step','Target drop >5cm','#727272','x')]
        for order,(key,label,color,marker) in enumerate(stages):
            points=[(e.get(key),i+(order-3)*.115) for i,e in enumerate(episodes) if e.get(key) is not None]
            if points:
                x,y=zip(*points);ax.scatter(x,y,c=color,marker=marker,s=52,label=label)
        ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()
        ax.set_xlabel('Accepted action step (objective private stages, not semantic recognition)')
        ax.grid(axis='x',alpha=.2);ax.legend(loc='upper center',bbox_to_anchor=(.5,1.30),ncol=4,fontsize=8)
        fig.tight_layout();save(fig,'physical_stages')
        # Saved raw robot images, with captions outside the image. The montage
        # is report-only; original lossless PNGs remain in the episode directory.
        gallery=[]
        gallery_episodes=[]
        for variant in ('single_table_occlusion','multi_region_search'):
            candidates=[e for e in episodes if e['variant']==variant and e['observation_condition']=='current' and e['world_mode']=='run']
            if not candidates:continue
            for memory in (('none','model_notes') if variant=='multi_region_search' else ('none',)):
                matching=[e for e in candidates if e['memory_mode']==memory]
                if matching:gallery_episodes.append(next((e for e in matching if e['seed']==7),matching[0]))
        for episode in episodes:
            if episode['variant'] in ('single_table_occlusion', 'multi_region_search') and episode['task_success'] and episode not in gallery_episodes:
                gallery_episodes.append(episode)
        for e in gallery_episodes:
            run=root/e['run']
            steps=sorted((run/'episode').glob('step_*'))
            if not steps:continue
            first=next((p for p in steps if (p/'before/wrist_rgb.png').exists()),None)
            visible=e.get('first_recognizable_step')
            boundary=next((x for x in records(run/'private_exploration/trajectory.jsonl')
                           if x['action_step']==visible),{})
            revealing_camera=next((name for name,view in boundary.get('visibility',{}).items()
                                  if view['passes_pixel_gate']),None)
            revealing_rgb='agentview_rgb' if revealing_camera=='head' else str(revealing_camera)+'_rgb'
            revealed=next((p for p in steps if (p/'after/observation.json').exists() and
                json.loads((p/'after/observation.json').read_text())['action_step']==visible),None)
            last=next((p for p in reversed(steps) if (p/'after/wrist_rgb.png').exists()),None)
            gallery.append((e,first/'before/wrist_rgb.png' if first else None,
                            revealed/'after'/(revealing_rgb+'.png') if revealed else None,
                            last/'after/wrist_rgb.png' if last else None,revealing_camera))
        if gallery:
            fig,axs=plt.subplots(len(gallery),3,figsize=(10,3.65*len(gallery)),squeeze=False)
            provenance=[]
            for i,(e,initial,revealing,final,revealing_camera) in enumerate(gallery):
                for j,(path,title) in enumerate(zip((initial,revealing,final),('Initial wrist RGB',
                        'First recognizable RGB: '+str(revealing_camera),'Final saved wrist RGB'))):
                    ax=axs[i,j];ax.axis('off');ax.set_title(title,fontsize=9)
                    if path and path.exists():ax.imshow(Image.open(path));provenance.append({'run':e['run'],'panel':title,'source':str(path)})
                    else:ax.text(.5,.5,'No qualifying saved frame',ha='center',va='center')
                axs[i,0].text(0,-.08,e['run']+' | success='+str(e['task_success']),transform=axs[i,0].transAxes,fontsize=8)
            fig.tight_layout(h_pad=3.0);save(fig,'robot_observations')
            (out/'observation_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    native=data.get('native_world_verification')
    if native:
        fig,ax=plt.subplots(figsize=(8,3))
        labels=[r['action_mode']+' / '+r['world_mode'] for r in native['records']]
        x=np.arange(len(labels));ax.bar(x-.17,[r['wait_wall_time_s'] for r in native['records']],width=.34,label='Wall wait (s)')
        ax.bar(x+.17,[r['wait_sim_time_s'] for r in native['records']],width=.34,label='Simulation wait (s)')
        ax.set_xticks(x,labels);ax.set_ylabel('Seconds');ax.set_title('Native default run vs explicit pause: fixed and parameterized leases')
        ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();save(fig,'native_world_policy')
    qualification_runs = ['02-single-qualification-seed7-v5','03-multi-qualification-seed7-v5']
    available = [root/name for name in qualification_runs if (root/name/'qualification_summary.json').exists()]
    if available:
        fig,axs=plt.subplots(len(available),3,figsize=(10,3.65*len(available)),squeeze=False)
        provenance=[]
        for i,run in enumerate(available):
            qualification=json.loads((run/'qualification_summary.json').read_text())
            revealing_camera=next(name for name,view in qualification['revealing_visibility']['views'].items()
                                  if view['passes_pixel_gate'])
            revealing_rgb='agentview_rgb' if revealing_camera=='head' else revealing_camera+'_rgb'
            for j,(folder,view,title) in enumerate((('qualification_initial','agentview_rgb','Initial head: target occluded'),
                    ('qualification_revealing',revealing_rgb,'Reachable revealing pose: '+revealing_camera+' RGB'),
                    ('qualification_final','wrist_rgb','Grasp / lift boundary: wrist RGB'))):
                path=run/folder/(view+'.png');ax=axs[i,j];ax.axis('off');ax.set_title(title,fontsize=9)
                if path.exists():ax.imshow(Image.open(path));provenance.append({'run':run.name,'panel':title,'source':str(path)})
                else:ax.text(.5,.5,'No saved frame',ha='center')
            axs[i,0].text(0,-.08,run.name+' | offline private qualification',transform=axs[i,0].transAxes,fontsize=8)
        fig.tight_layout();save(fig,'scene_qualification')
        (out/'qualification_image_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Saved standalone figures:',out)


if __name__=='__main__':
    main()
