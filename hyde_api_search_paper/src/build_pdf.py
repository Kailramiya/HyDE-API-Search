"""
build_pdf.py  —  compile research_paper.tex to PDF
Run: python build_pdf.py
"""
import subprocess, os, sys, shutil

TEX_FILE = "research_paper.tex"
MIKTEX   = r"C:\Users\hp\AppData\Local\Programs\MiKTeX\miktex\bin\x64"

# ── Clean PATH: keep only real directories, strip npm/Roaming junk ──────
raw_path = os.environ.get("PATH", "")
clean_dirs = []
for p in raw_path.split(os.pathsep):
    p = p.strip()
    if not p:
        continue
    if "npm" in p.lower() or ("roaming" in p.lower() and "miktex" not in p.lower()):
        continue
    if os.path.isdir(p):
        clean_dirs.append(p)

if MIKTEX not in clean_dirs:
    clean_dirs.insert(0, MIKTEX)

env = os.environ.copy()
env["PATH"] = os.pathsep.join(clean_dirs)

# ── Locate pdflatex ──────────────────────────────────────────────────────
pdflatex = os.path.join(MIKTEX, "pdflatex.exe")
if not os.path.isfile(pdflatex):
    pdflatex = shutil.which("pdflatex", path=env["PATH"])
    if not pdflatex:
        print("ERROR: pdflatex not found. Install MiKTeX from https://miktex.org")
        sys.exit(1)

print(f"Using: {pdflatex}")

# ── Run pdflatex twice (resolves references/citations) ──────────────────
for run in (1, 2):
    print(f"\n--- Pass {run}/2 ---")
    result = subprocess.run(
        [pdflatex, "-interaction=nonstopmode", "-halt-on-error", TEX_FILE],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    # Show last 20 lines of log (errors visible here)
    lines = (result.stdout + result.stderr).splitlines()
    for line in lines[-20:]:
        print(line)
    if result.returncode != 0:
        print(f"\nERROR on pass {run}. See research_paper.log for details.")
        sys.exit(result.returncode)

# ── Report result ────────────────────────────────────────────────────────
pdf = TEX_FILE.replace(".tex", ".pdf")
if os.path.isfile(pdf):
    size = os.path.getsize(pdf) // 1024
    print(f"\nDone! -> {os.path.abspath(pdf)}  ({size} KB)")
    os.startfile(pdf)   # open PDF in default viewer
else:
    print("\nCompilation finished but PDF not found. Check research_paper.log")
