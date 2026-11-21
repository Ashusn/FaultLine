import collections
import os
import re
import subprocess
import time

from IPython.display import clear_output

os.environ["HF_TOKEN"] = "PASTE_NEW_HF_TOKEN"
os.environ["RUNPOD_API_KEY"] = "PASTE_NEW_RUNPOD_KEY"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["AUTO_TERMINATE"] = "1"
MODE = "dry"
ONLY_MODELS = ""
REPO = "https://github.com/amareshhebbar/FaultLine"

CONFIGS = {"dry": "configs/config_dry.yaml", "small": "configs/config.yaml", "big": "configs/config_big_budget.yaml"}
BASE = "/workspace"
CODE = f"{BASE}/FaultLine/models"
WORK = f"{BASE}/faultline_models_{MODE}"
LOGS = f"{BASE}/lab_logs"
LOG = f"{LOGS}/night.log"
os.makedirs(LOGS, exist_ok=True)


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, executable="/bin/bash")
    return (r.stdout + r.stderr).strip()


print("pod id:", os.environ.get("RUNPOD_POD_ID", "MISSING"))
print(sh("nvidia-smi --query-gpu=name,memory.used,memory.total,compute_cap,driver_version --format=csv,noheader"))
print(sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader") or "GPU empty")
print(sh(f"df -h {BASE} | tail -n 1"))

if sh("ps aux | grep -E '[r]eal\\.sh|[r]un\\.py pipeline'"):
    print("ALREADY RUNNING, showing the log")
else:
    print(sh(f"git -C {BASE}/FaultLine pull 2>&1 || git clone {REPO} {BASE}/FaultLine 2>&1"))
    real = """#!/bin/bash
cd __CODE__
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLM_CONFIG=__CFG__
export FLM_WORK=__WORK__
export ONLY_MODELS="__ONLY__"
echo "######## RUN START __MODE__ $(date) ########"
bash setup.sh
.venv/bin/python run.py diffcheck
.venv/bin/python run.py pipeline
[ -f __WORK__/report/HEADLINE.md ] && echo "PROOF OK: FaultLine models __MODE__" || echo "PROOF MISSING: FaultLine models __MODE__"
.venv/bin/python run.py status
echo "NIGHT RUN FINISHED"
"""
    for k, v in {"__CODE__": CODE, "__CFG__": CONFIGS[MODE], "__WORK__": WORK, "__ONLY__": ONLY_MODELS, "__MODE__": MODE}.items():
        real = real.replace(k, v)
    open(f"{BASE}/real.sh", "w").write(real)
    mirror = r"""#!/bin/bash
L=__LOGS__
up() {
  PY=__CODE__/.venv/bin/python
  [ -x "$PY" ] || return 0
  { echo "updated $(date)"; echo; df -h /workspace | tail -n 1; echo; grep -aE "RUN START|PROOF|FAILED|PASSED|STOPPED|CHECK PROBLEM|GB free|weights deleted|finished in|leak check|NIGHT RUN|TERMINATING|everything works" $L/night.log | tail -n 150; } > $L/summary.txt
  rm -rf /tmp/logsnap; mkdir -p /tmp/logsnap/live
  cp $L/*.log $L/summary.txt /tmp/logsnap/live/ 2>/dev/null
  [ -d __WORK__/results ] && cp -r __WORK__/results /tmp/logsnap/results
  [ -d __WORK__/report ] && cp -r __WORK__/report /tmp/logsnap/report
  HF_HUB_DISABLE_XET=1 $PY - <<'EOF'
import os
from huggingface_hub import HfApi
a = HfApi(token=os.environ["HF_TOKEN"])
r = a.whoami()["name"] + "/FaultLine_Proof"
a.create_repo(r, repo_type="dataset", private=True, exist_ok=True)
a.upload_folder(folder_path="/tmp/logsnap", path_in_repo="__MODE__", repo_id=r, repo_type="dataset", commit_message="Live logs")
EOF
}
sleep 60
while ps aux | grep -q '[r]eal[.]sh'; do up; sleep 600; done
echo "TERMINATING pod $RUNPOD_POD_ID at $(date)" >> $L/night.log
up
[ "$AUTO_TERMINATE" = "1" ] || exit 0
__CODE__/.venv/bin/python - <<'EOF'
import os, httpx
url = "https://api.runpod.io/graphql?api_key=" + os.environ["RUNPOD_API_KEY"]
pid = os.environ["RUNPOD_POD_ID"]
t = httpx.post(url, json={"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pid}, timeout=60)
print("terminate:", t.status_code, t.text[:300], flush=True)
if t.status_code != 200 or "error" in t.text.lower():
    httpx.post(url, json={"query": 'mutation { podStop(input: {podId: "%s"}) { id } }' % pid}, timeout=60)
EOF
"""
    for k, v in {"__CODE__": CODE, "__WORK__": WORK, "__LOGS__": LOGS, "__MODE__": MODE}.items():
        mirror = mirror.replace(k, v)
    open(f"{BASE}/mirror.sh", "w").write(mirror)
    open(LOG, "a").write("\n")
    for s in ["real.sh", "mirror.sh"]:
        subprocess.Popen(["setsid", "nohup", "bash", f"{BASE}/{s}"], stdout=open(LOG if s == "real.sh" else f"{LOGS}/{s}.out", "a"), stderr=subprocess.STDOUT, env=os.environ.copy(), start_new_session=True)
    print(f"STARTED. phone: https://huggingface.co/datasets/AmareshHebbar/FaultLine_Proof/blob/main/{MODE}/live/summary.txt")
    time.sleep(5)

skip = re.compile(r"(\d+%\|)|(it/s\])|(s/it\])|(examples/s)|(^INFO )|(^\s*$)|(^=+$)")
keep = collections.deque(maxlen=60)
pos = max(0, os.path.getsize(LOG) - 20000)
done = False
while not done:
    with open(LOG, "r", errors="ignore") as f:
        f.seek(pos)
        chunk = f.read()
        pos = f.tell()
    for raw in re.split(r"[\r\n]", chunk):
        if not skip.search(raw):
            keep.append(raw[:220])
        if "NIGHT RUN FINISHED" in raw:
            done = True
    clear_output(wait=True)
    print(time.strftime("%H:%M:%S"), "| live log | safe to close the tab, the run keeps going")
    print("\n".join(keep))
    if not done:
        time.sleep(15)
print("\nAll done. The pod terminates itself within about 10 minutes.")
