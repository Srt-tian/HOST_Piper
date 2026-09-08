"""Report dependency conflicts; isolate Decord's known wheel-tag metadata defect.

This does not make pip check clean, nor establish video/GPU correctness. No metadata is
modified. Decord's published py3-none wheel internally declares cp36-cp36m; retain a visible
warning and validate import, while rejecting every other pip check failure.
"""
import subprocess
import sys
import importlib.metadata as metadata

result = subprocess.run([sys.executable,'-m','pip','check'],text=True,capture_output=True)
lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
known = 'decord 0.6.0 is not supported on this platform'
if result.returncode and (not lines or any(line != known for line in lines)):
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    raise SystemExit(result.returncode)
if known in lines:
    wheel = metadata.distribution('decord').read_text('WHEEL')
    if metadata.version('decord') != '0.6.0' or 'Tag: cp36-cp36m-manylinux2010_x86_64' not in wheel:
        raise RuntimeError('Decord warning differs from the audited wheel metadata defect')
    import decord
    print('WARNING: raw pip check reports Decord cp36 wheel tag; import works; '
          'real-video random/sequential decode still requires validation.')
else:
    print(result.stdout, end='')
print('Dependency requirements OK; reported wheel-tag exception is NOT suppressed or repaired.')
