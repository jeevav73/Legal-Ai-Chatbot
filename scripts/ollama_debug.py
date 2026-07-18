import json
import subprocess
import urllib.request

print('=== HTTP API Test ===')
try:
    url = 'http://127.0.0.1:11434/api/generate'
    data = json.dumps({'model': 'llama3.2:3b', 'prompt': 'Hello'}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req)
    body = resp.read().decode('utf-8')
    print('status', resp.getcode())
    print('headers', resp.info())
    print('body', body)
except Exception as exc:
    print('HTTP error:', type(exc).__name__, exc)

print('\n=== CLI Test ===')
try:
    cmd = [r'C:\Users\dorem\AppData\Local\Programs\Ollama\ollama.exe', 'run', 'llama3.2:3b', '--hidethinking', 'Hello']
    print('cmd:', cmd)
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    print('stdout:', out)
except subprocess.CalledProcessError as exc:
    print('CLI return code', exc.returncode)
    print('stdout/stderr:', exc.output)
except Exception as exc:
    print('CLI error:', type(exc).__name__, exc)
