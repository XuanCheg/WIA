"""Run bounded native candidate captures on an explicitly selected idle GPU.

Run: python scripts/capture_interactiveworld_candidates.py --gpu 0 --ids iw_h5_01 --output OUTPUT.
Use in a persistent tmux terminal. No policy or robot action is executed.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from wia.resources import ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--ids',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    for identity in a.ids:
        query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
        row=next(r.split(',') for r in query.splitlines() if int(r.split(',')[0])==a.gpu)
        if int(row[2])>=5 or int(row[3])>=512:raise RuntimeError('Selected GPU occupied; no launch.')
        out=a.output/identity
        if (out/'launch.json').exists():
            previous=json.loads((out/'launch.json').read_text())
            if 'returncode' not in previous:raise RuntimeError('Existing native attempt has no completion receipt; refusing overwrite.')
            archive=a.output/'previous_attempts'/identity/str(int(previous['started']*1000))
            archive.parent.mkdir(parents=True,exist_ok=True)
            out.rename(archive)
        out.mkdir(parents=True,exist_ok=True)
        cache_root=Path(os.environ.get('WIA_NATIVE_CACHE_ROOT',f'/dev/shm/wia_native_review_20261030_{os.getuid()}'))
        cache_root.mkdir(parents=True,exist_ok=True)
        import shutil
        if shutil.disk_usage(cache_root).free < 12*1024**3:raise RuntimeError('Native cache volume has less than 12 GiB free.')
        cache=cache_root/(hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:16]+'_'+str(time.time_ns()))
        (cache/'tmp').mkdir(parents=True,exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=row[1].strip(),CUDA_DEVICE_ORDER='PCI_BUS_ID',OMNIGIBSON_GPU_ID='0',OMNIGIBSON_HEADLESS='1',OMNI_KIT_ACCEPT_EULA='YES',OMNIGIBSON_DATA_PATH='/local_home2/zhouxinyang/data/behavior',VK_ICD_FILENAMES=str(ROOT/'configs/esi_vulkan_egl.json'),LD_LIBRARY_PATH='/local_home2/zhouxinyang/runtime_libs',PYTHONPATH=str(ROOT/'src')+':'+str(ROOT.parent/'Show-Harness-Explore'),OMNIGIBSON_APPDATA_PATH=str(cache),TMPDIR=str(cache/'tmp'),XDG_CACHE_HOME=str(cache/'xdg'),CUDA_CACHE_PATH=str(cache/'cuda'),__GL_SHADER_DISK_CACHE_PATH=str(cache/'gl'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',NUMEXPR_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
        cmd=[sys.executable,'-u','-m','wia.candidates.capture','--id',identity,'--output',str(out)]
        receipt={'id':identity,'gpu_index':a.gpu,'gpu_uuid':row[1].strip(),'command':cmd,'started':time.time(),'model_calls':0,'robot_actions':0,'temporary_cache':str(cache)}
        with (out/'native.log').open('w') as log:
            process=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);receipt['pid']=process.pid
            (out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
            try:receipt['returncode']=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                receipt.update(returncode=process.returncode,error='bounded native capture timeout')
        receipt['wall_time_s']=time.time()-receipt['started']
        (out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n')
        try:
            shutil.rmtree(cache)
            receipt['temporary_cache_removed']=True
        except OSError as error:
            receipt['temporary_cache_removed']=False
            receipt['cache_cleanup_error']=f'{type(error).__name__}: {error}'
        receipt['wall_time_s']=time.time()-receipt['started'];(out/'launch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(identity,receipt['returncode'],flush=True)

if __name__=='__main__':main()
