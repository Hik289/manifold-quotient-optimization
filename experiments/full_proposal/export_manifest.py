"""Materialize every logical grid slot for independent coverage auditing."""
import gzip,json,hashlib,time
from grid import ROOT,all_configs,run_id,summary
from worker import effective
dest=ROOT/'full_manifest.jsonl.gz';count=0;invalid=0
with gzip.open(dest,'wt',compresslevel=3) as stream:
    for ordinal,c in enumerate(all_configs()):
        feasible=True
        if c['family']=='graph' and c.get('placement')=='classifier':
            feasible=c['rank']<={'cora':7,'citeseer':6,'pubmed':3,'ogbn-arxiv':40}[c['dataset']]
        invalid+=not feasible
        stream.write(json.dumps(dict(ordinal=ordinal,logical_id=run_id(c),execution_id=run_id(effective(c)),preflight='regular_rank_feasible' if feasible else 'infeasible_rank',config=c),separators=(',',':'))+'\n')
        count+=1
report=dict(logical_configs=count,infeasible_rank_configs=invalid,bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),note='A manifest is a plan, not an experiment result.')
(ROOT/'manifest_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
