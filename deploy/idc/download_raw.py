"""Download the inventoried raw training data only, one prefix at a time on IDC.

Run only after reviewing inventory and verifying pilot success. Uses eip-kit config.
Existing pilot files are retained by tosutil update mode through eip-kit semantics.
"""
import json
from pathlib import Path
import subprocess
from inventory_tos import BASE, call
from eip_kit.tos.client import TOSClient
from eip_kit.tos.config import load_config, get_config_path


if __name__ == '__main__':
    if subprocess.check_output(['hostname'], text=True).strip() != 'dev-instance-shenrongtian':
        raise RuntimeError('IDC hostname mismatch')
    manifest = json.loads((BASE/'inventory.json').read_text())
    call('auth','check')
    if get_config_path() != Path.home()/'.eip-kit/eip.config.yaml':
        raise ValueError('Only the global eip-kit credential configuration is allowed')
    client = TOSClient(load_config())
    for source in manifest['sources']:
        dest = Path(source['root']).resolve()
        if not dest.is_relative_to(BASE/'raw'):
            raise ValueError('Download destination escapes dedicated raw directory')
        for part in ('meta','data','videos'):
            # The public eip-kit cp CLI does not expose recursive flags. Use its configured
            # TOSClient without modifying eip-kit or putting credentials in command arguments.
            print(json.dumps({'source':source['id'], 'part':part, 'status':'started'}), flush=True)
            args = ['cp',source['tos']+part+'/',str(dest/part),'-r','-flat','-u',
                    '-j=2','-p=1','-nfj=2','-vchecksum',
                    '-o='+str(BASE/'logs/tos'),'-cpd='+str(BASE/'cache/tos_checkpoints')]
            try:
                summary = client.run_command(args)
            except Exception:
                raise RuntimeError('TOS batch failed; stopped without retry') from None
            print(summary, flush=True)
            print(json.dumps({'source':source['id'], 'part':part, 'status':'downloaded'}), flush=True)
