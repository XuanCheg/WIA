"""Audit measured candidate geometry. Run: python scripts/audit_interactiveworld_candidates.py INVENTORY OUTPUT.

Run in a persistent CPU terminal. This performs no robot commands or GPU work.
"""
import json,os,subprocess,sys
from pathlib import Path

if __name__=='__main__':
    usd=next((Path(sys.prefix)/'lib/python3.10/site-packages/isaacsim/extscache').glob('omni.usd.libs-*'))
    env=dict(os.environ,LD_LIBRARY_PATH=str(usd/'bin')+':'+str(usd/'bin/deps'),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    raise SystemExit(subprocess.call([sys.executable,'-m','wia.candidates.reachability',*sys.argv[1:]],env=env))
