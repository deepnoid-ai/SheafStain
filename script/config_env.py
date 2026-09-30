import sys
import yaml
import shlex

path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'

try:
    with open(path) as f: cfg = yaml.safe_load(f) or {}

except FileNotFoundError:
    sys.stderr.write("config_env: file not found: %s\n" % path)
    sys.exit(1)

if not isinstance(cfg, dict):
    sys.stderr.write("config_env: top level of %s must be a mapping\n" % path)
    sys.exit(1)

for key, val in cfg.items():
    if val is None or isinstance(val, (list, dict)): continue
    if not str(key).replace('_', '').isalnum(): continue  # skip keys that are not valid shell identifiers
    if isinstance(val, bool): val = 'true' if val else 'false'
    
    print("export %s=%s" % (key, shlex.quote(str(val))))
