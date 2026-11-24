"""Start the two persistent tmux workers for an already checked study."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from tools.run_pair_sweep import DEFAULT_OUTPUT, digest, read, verify_inputs, write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args();output=args.output.resolve()
    verify_inputs(output)
    check=read(output/'preflight.json')
    if check['status']!='passed' or check['manifest_sha256']!=digest(output/'manifest.json'):
        raise ValueError('Preflight has not passed for this frozen manifest')
    protocol=read(output/'protocol.json')
    socket='acd-seven-arms-20261118'
    sessions=[]
    for gpu in protocol['gpu_allowlist']:
        session=f'gpu{gpu}'
        exists=subprocess.run(['tmux','-L',socket,'has-session','-t',session],capture_output=True).returncode==0
        if not exists:
            command=[sys.executable,'-B','-u','-m','tools.run_pair_sweep','worker',
                     '--output',str(output),'--gpu',str(gpu)]
            shell_command='exec '+shlex.join(command)+' >> '+shlex.quote(str(output/f'worker-{gpu}.log'))+' 2>&1'
            subprocess.run(['tmux','-L',socket,'new-session','-d','-s',session,
                            '-c',str(output/'source'),shell_command],check=True)
        sessions.append({'session':session,'gpu':gpu,'reused':exists})
    receipt={'started_at':time.time(),'tmux_socket':socket,'sessions':sessions,
             'source_manifest_sha256':digest(output/'manifest.json')}
    write(output/'launch.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
