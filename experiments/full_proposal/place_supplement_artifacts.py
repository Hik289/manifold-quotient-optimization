"""Keep supplement figures/tables with their study; preserve all content."""
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parent.parent.parent/'paper'
files=[ROOT/'figures'/name for name in [
    'new_task_initial_scores.tex','new_task_scores.tex','new_task_curves_input.tex',
    'new_task_convergence_input.tex','initialization_confirmation_tables.tex',
    'matrix_initialization_tables.tex']]
files.append(ROOT/'sections/initial_central_cases.tex')
pattern=re.compile(r'\\begin\{(table|figure)\}(?:\[[^]]*\])?(.*?)\\end\{\1\}',re.S)
def inline(match):
    kind,body=match.group(1),match.group(2)
    assert body.count(r'\caption{')==1
    body=body.replace(r'\caption{',r'\captionof{'+kind+'}{',1)
    return '\\par\\medskip\\noindent\\begin{minipage}{\\linewidth}\n'+body+'\n\\end{minipage}\\par\\medskip'
for file in files:
    if file.exists():
        before=file.read_text();after=pattern.sub(inline,before)
        if before!=after:file.write_text(after)
print('Supplement artifacts placed beside their studies; all captions and data retained.')
