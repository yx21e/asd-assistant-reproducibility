"""Verify corrected table units and statistics without reconstructing missing logs."""
import argparse
import json
import statistics
from pathlib import Path
from replay_reported_tables import tex_rows


def verify(root):
    banks={k:[json.loads(s) for s in (root/'evidence/historical'/f'{k}.jsonl').read_text().splitlines()]
           for k in ('task2_curated','task2_large','task3_curated','task3_large')}
    units=json.loads((root/'evidence/historical/citation_unit_counts.json').read_text())
    table=tex_rows(root/'latex/tables/table_metric_audit.tex')
    lookup={r[0]:r for r in table}
    checks=[]

    def check(label, condition):
        if not condition: raise AssertionError(label)
        checks.append(label)

    t1=json.loads((root/'evidence/historical/task1_recorded_summary.json').read_text())['values']
    check('Task 1 original summary bank size',t1['samples']==1200)
    check('Task 1 original summary family sizes',[t1['per_mode'][m]['samples'] for m in ('coordinates','function','qtl')]==[200,400,600])
    accepted=round(t1['samples']*(1-t1['abstain_rate']))
    check('Task 1 recorded routing count',lookup['Task 1 route accuracy'][1]==f'{accepted}/{t1["samples"]}')

    labels={'task2_curated':'Task 2 curated','task2_large':'Task 2 larger-sample',
            'task3_curated':'Task 3 curated','task3_large':'Task 3 larger-sample'}
    for name,rows in banks.items():
        by_id={r['example_id']:r for r in units[name]}
        check(name+' unique audit IDs',len(by_id)==len(rows)==len(units[name]))
        check(name+' same query IDs',set(by_id)=={r['example_id'] for r in rows})
        for row in rows:
            audit=by_id[row['example_id']]
            check(name+row['example_id']+' mode',audit['citation_mode']=='paper_proxy')
            check(name+row['example_id']+' CSR ratio',audit['csr_total_count']>0 and
                  abs(audit['csr_supported_count']/audit['csr_total_count']-row['csr'])<1e-12)
        csr=[sum(r[k] for r in units[name]) for k in ('csr_supported_count','csr_total_count')]
        cvr=[sum(r[k] for r in rows) for k in ('verified_citation_count','citation_count')]
        check(name+' CSR table numerator/denominator',lookup[labels[name]+' CSR'][1]==f'{csr[0]}/{csr[1]}')
        check(name+' CVR table numerator/denominator',lookup[labels[name]+' CVR'][1]==f'{cvr[0]}/{cvr[1]}')

    t2=banks['task2_large']
    counts={r['example_id']:r['distinct_pmid_count'] for r in units['task2_large']}
    check('one/two PMID groups',set(counts.values())=={1,2})
    check('two entries per PMID in these saved answers',all(r['citation_count']==2*counts[r['example_id']] for r in t2))
    groups=[[r for r in t2 if r['tier']==v] for v in ('core','extended','weak_support')]
    groups += [[r for r in t2 if r['autism_doc']==v] for v in (True,False)]
    groups += [[r for r in t2 if counts[r['example_id']]==v] for v in (1,2)]
    actual=tex_rows(root/'latex/tables/table_task2_multifactor_robustness.tex')
    check('multifactor row count',len(actual)==len(groups))
    for i,(cells,rows) in enumerate(zip(actual,groups)):
        expected=[str(len(rows))]+[r'\meanstd{'+f'{statistics.mean(r[k] for r in rows):.4f}'+'}{'+
                  f'{statistics.stdev(r[k] for r in rows):.4f}'+'}' for k in ('faithfulness','ar')]
        expected += [f'{statistics.mean(r[k] for r in rows):.4f}' for k in ('csr','cvr')]
        check('multifactor cells '+str(i),cells[2:]==expected)
    check('reference counting labels',[r[:2] for r in actual[-2:]]==[['Distinct PMIDs','1'],['Distinct PMIDs','2']])
    caption=(root/'latex/tables/table_bootstrap_stability.tex').read_text()
    check('bootstrap versus point-estimate caption','CSR and CVR: original query-averaged point estimates' in caption)
    return dict(passed=True,checks=len(checks),records=246,
        scope='Corrected synthesis audit counts and multifactor cells plus bootstrap caption.',
        task1_original_output_reconstruction=False,
        warning='Task 1 historical summary rows are not reconstructed from original generated outputs.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(p.parse_args().root),indent=2))
