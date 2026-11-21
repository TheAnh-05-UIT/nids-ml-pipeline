"""Combine disjoint node evaluations of the same frozen weights exactly once."""
import math

from nids_ml.adapters.swarm.contract import require
from nids_ml.adapters.swarm.full_job import FullTemplate
from nids_ml.evaluation.thresholds import validate_threshold
from nids_ml.training.batched import BinaryMetrics
from nids_ml.training.full_plan import identity


def combine(manifest,reports):
    require(manifest['plan_id']==identity(manifest),'FULL_EVALUATION_PLAN_ID')
    require(len(reports)==2 and {r['client'] for r in reports}=={0,1},'FULL_EVALUATION_CLIENTS')
    spec=FullTemplate(manifest['plan_id']);first=reports[0];split=first['split']
    threshold=validate_threshold(first['threshold'])
    require(split in ('validation','test') and len({r['weights_sha256'] for r in reports})==1,
            'FULL_EVALUATION_SAME_CHECKPOINT')
    totals={key:BinaryMetrics(threshold) for key in ('overall',*manifest['family_coverage'])}
    for report in reports:
        require(type(report['client']) is int and report['status']=='PASS_FULL_PLAN_FROZEN_EVALUATION'
                and report['plan_id']==manifest['plan_id'] and report['model_id']==spec.identity
                and report['split']==split and report['threshold']==threshold and report['training'] is False
                and report['scope']==manifest['scope']
                and report['evaluation_placement']==manifest['evaluation_placement'],'FULL_EVALUATION_IDENTITY')
        metrics=report['metrics'];require(set(metrics['per_family'])==set(manifest['family_coverage']),'FULL_EVALUATION_FAMILIES')
        local_counts=manifest['clients'][str(report['client'])]['evaluation'][split]['class_counts']
        local_family=BinaryMetrics(threshold)
        for key,value in [('overall',metrics['overall']),*metrics['per_family'].items()]:
            require(type(value['samples']) is int and value['samples']>=0,'FULL_METRIC_SAMPLES')
            if value['samples']==0:
                require(value==dict(status='NOT_EVALUATED',samples=0),'FULL_EMPTY_METRIC')
                if key=='overall':require(sum(local_counts.values())==0,'FULL_EVALUATION_LOCAL_COUNT')
                continue
            require(value['threshold']==threshold and type(value['loss']) in (int,float)
                    and math.isfinite(value['loss']) and value['loss']>=0,'FULL_METRIC_LOSS')
            cm=value['confusion_matrix']
            require(type(cm) is list and len(cm)==2 and all(type(row) is list and len(row)==2 for row in cm)
                    and all(type(n) is int and n>=0 for row in cm for n in row)
                    and sum(map(sum,cm))==value['samples'],'FULL_METRIC_CONFUSION')
            if key=='overall':
                require([sum(row) for row in cm]==[local_counts.get(str(y),0) for y in (0,1)],'FULL_EVALUATION_LOCAL_COUNT')
            else:
                require(sum(cm[1 if key=='Normal' else 0])==0,'FULL_FAMILY_TARGET')
                local_family.n+=value['samples'];local_family.tn+=cm[0][0];local_family.fp+=cm[0][1]
                local_family.fn+=cm[1][0];local_family.tp+=cm[1][1]
            total=totals[key];total.n+=value['samples'];total.tn+=cm[0][0];total.fp+=cm[0][1]
            total.fn+=cm[1][0];total.tp+=cm[1][1];total.loss+=value['loss']*value['samples']
        require(local_family.n==metrics['overall']['samples'],'FULL_LOCAL_FAMILY_COUNT')
        if local_family.n:
            require(local_family.report()['confusion_matrix']==metrics['overall']['confusion_matrix'],'FULL_LOCAL_FAMILY_CONFUSION')
    require(totals['overall'].n==sum(manifest['split_counts'][split].values()),'FULL_GLOBAL_EVALUATION_COUNT')
    for family,counts in manifest['family_coverage'].items():
        require(totals[family].n==counts[split],'FULL_GLOBAL_FAMILY_COUNT')
    return dict(status='PASS_DISJOINT_FULL_PLAN_EVALUATION',plan_id=manifest['plan_id'],model_id=spec.identity,
        weights_sha256=first['weights_sha256'],split=split,threshold=threshold,training=False,
        metrics=dict(overall=totals['overall'].report(),per_family={f:totals[f].report() for f in manifest['family_coverage']}),
        evaluation_placement=manifest['evaluation_placement'],scope=manifest['scope'],
        label_provenance=manifest['label_provenance'],independent_ground_truth_verified=False)
