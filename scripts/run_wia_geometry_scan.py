"""Collect selected native geometry without robot or model actions.

Prepare: python scripts/run_wia_geometry_scan.py --prepare OUTPUT
Worker: python scripts/run_wia_geometry_scan.py --queue OUTPUT --gpu 0
Run workers in persistent terminals. Each launch rechecks the selected GPU.
"""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from wia.resources import ROOT


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path);p.add_argument('--queue',type=Path);p.add_argument('--gpu',type=int);a=p.parse_args()
    if a.prepare:
        out=a.prepare.resolve();out.mkdir(parents=True,exist_ok=True)
        catalogue=json.loads((ROOT/'explore_to_achieve/wia_v1/release/catalogue.json').read_text())
        selected=catalogue['tasks']
        aliases=[r['alias'] for r in selected]
        if len(aliases)!=len(set(aliases)):raise ValueError('Duplicate selected aliases are not allowed.')
        candidate_count=sum(r.get('origin')=='interactiveworld_candidate' for r in selected)
        # These four explicitly tracked pilots are owned by the main reviewer.
        pilots={'T01','T43','R-S1','T51'}
        jobs=[{'alias':r['alias'],'state':'pending','task_file':r['task_file'],'scene_file':r['scene_file']} for r in selected if r['alias'] not in pilots]
        if (out/'jobs.json').exists():raise ValueError('Existing scan state is never overwritten.')
        write(out/'jobs.json',jobs);write(out/'scope.json',{'selected_count':len(selected),'original_count':len(selected)-candidate_count,'candidate_count':candidate_count,'batch_count':len(jobs),'separately_tracked_pilots':sorted(pilots),'robot_actions':0,'model_calls':0});return
    queue=a.queue.resolve();gpu=a.gpu
    if gpu is None:raise ValueError('An explicit physical GPU is required.')
    while True:
        with (queue/'state.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads((queue/'jobs.json').read_text())
            if not any(x['state']=='pending' for x in jobs):return
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
        row=next(line.split(',') for line in raw.splitlines() if int(line.split(',')[0])==gpu)
        if int(row[1])>=5 or int(row[2])>=512:time.sleep(15);continue
        with (queue/'state.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads((queue/'jobs.json').read_text());job=next((x for x in jobs if x['state']=='pending'),None)
            if job is None:return
            job.update(state='running',gpu=gpu,worker_pid=os.getpid(),started=time.time());alias=job['alias'];write(queue/'jobs.json',jobs)
        output=queue/'captures'
        cmd=[sys.executable,str(ROOT/'scripts/capture_interactiveworld_candidates.py'),'--gpu',str(gpu),'--ids',alias,'--output',str(output)]
        result=subprocess.run(cmd,cwd=ROOT)
        receipt=output/alias/'capture.json';native=output/alias/'launch.json'
        capture=json.loads(receipt.read_text()) if receipt.exists() else {}
        with (queue/'state.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);jobs=json.loads((queue/'jobs.json').read_text());job=next(x for x in jobs if x['alias']==alias)
            if not native.exists():job.update(state='pending',last_launch_reason='GPU became occupied before native launch')
            else:job.update(state='captured' if capture.get('status')=='captured' else 'failed',returncode=result.returncode,finished=time.time(),receipt=str(receipt),inventory=str(output/alias/'inventory.json'),error=capture.get('error'))
            write(queue/'jobs.json',jobs)
        print(alias,job['state'],flush=True)

if __name__=='__main__':main()
