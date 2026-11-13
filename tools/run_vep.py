"""Annotate a candidate VCF with the recorded external VEP/dbNSFP resources."""
import argparse,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--resources',type=Path,required=True)
p.add_argument('--dry-run',action='store_true')
a=p.parse_args()
source=a.input.resolve(); output=a.output.resolve(); resources=a.resources.resolve()
if not source.is_file(): p.error('Candidate VCF missing')
if output.exists(): p.error('Output exists; choose a new output file')
for rel in ['homo_sapiens/113_GRCh38','dbnsfp/dbNSFP5.1a_grch38.gz','dbnsfp/dbNSFP5.1a_grch38.gz.tbi']:
    if not (resources/rel).exists(): p.error('External resource missing: '+rel)
fields=json.loads((ROOT/'config/vep_fields.json').read_text())
image='ensemblorg/ensembl-vep@sha256:d46fbc66e5ee1c90184d0ea7c692c76984bd9a0c7607ae83d5f3cbdfcc95767a'
command=['docker','run','--rm','-v',str(resources)+':/opt/vep/.vep:ro','-v',str(source.parent)+':/input:ro','-v',str(output.parent)+':/output',image,'vep','--assembly','GRCh38','--cache','--canonical','--dir_cache','/opt/vep/.vep','--dir_plugins','/plugins','--input_file','/input/'+source.name,'--offline','--output_file','/output/'+output.name,'--plugin','dbNSFP,/opt/vep/.vep/dbnsfp/dbNSFP5.1a_grch38.gz,'+','.join(fields),'--tab']
print(json.dumps(command,indent=2))
if not a.dry_run:
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(command,check=True)
