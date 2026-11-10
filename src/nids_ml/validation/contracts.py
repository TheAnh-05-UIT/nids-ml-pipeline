"""Validate supported local schema keywords and published-report integrity.
Kiểm schema keywords local được hỗ trợ và integrity của report.

Đây là validator giới hạn, không phải engine JSON Schema tổng quát: keyword
ngoài tập hỗ trợ gây lỗi, không âm thầm coi là hợp lệ. Không resolve HTTP refs.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

from nids_ml.validation.workspace import workspace_path
from nids_ml.data.inventory import fingerprint

SUPPORTED = {'$schema','title','description','$defs','$ref','type','const','enum','required',
             'properties','additionalProperties','items','minItems','maxItems','uniqueItems',
             'minLength','maxLength','pattern','minimum','maximum','allOf','format'}


def validate_schema(value, schema, schemas=None, at='$'):
    unknown=set(schema)-SUPPORTED
    if unknown:raise ValueError('Unsupported schema keywords: '+','.join(sorted(unknown)))
    if '$ref' in schema:
        ref=schema['$ref'];name,_,pointer=ref.partition('#')
        if not schemas or name not in schemas or not pointer.startswith('/'):
            raise ValueError('Only explicit local schema references supported')
        target=schemas[name]
        for part in pointer[1:].split('/'):target=target[part.replace('~1','/').replace('~0','~')]
        validate_schema(value,target,schemas,at)
    for clause in schema.get('allOf',[]):validate_schema(value,clause,schemas,at)
    types={'object':lambda x:type(x) is dict,'array':lambda x:type(x) is list,
           'string':lambda x:type(x) is str,'integer':lambda x:type(x) is int,
           'number':lambda x:type(x) in (float,int),'null':lambda x:x is None,
           'boolean':lambda x:type(x) is bool}
    if 'type' in schema:
        kinds=schema['type'] if type(schema['type']) is list else [schema['type']]
        if any(k not in types for k in kinds) or not any(types[k](value) for k in kinds):raise ValueError(at+': wrong type')
    if 'const' in schema and value!=schema['const']:raise ValueError(at+': const mismatch')
    if 'enum' in schema and value not in schema['enum']:raise ValueError(at+': invalid enum')
    if type(value) is dict:
        if not set(schema.get('required',[]))<=set(value):raise ValueError(at+': missing required keys')
        props=schema.get('properties',{})
        for key,item in value.items():
            if key in props:validate_schema(item,props[key],schemas,at+'.'+key)
            elif schema.get('additionalProperties') is False:raise ValueError(at+': unknown property '+key)
            elif type(schema.get('additionalProperties')) is dict:validate_schema(item,schema['additionalProperties'],schemas,at+'.'+key)
    if type(value) is list:
        if len(value)<schema.get('minItems',0) or len(value)>schema.get('maxItems',10**9):raise ValueError(at+': array length')
        if schema.get('uniqueItems') and len({json.dumps(x,sort_keys=True) for x in value})!=len(value):raise ValueError(at+': duplicate items')
        for i,item in enumerate(value):
            if 'items' in schema:validate_schema(item,schema['items'],schemas,at+'['+str(i)+']')
    if type(value) is str:
        if not schema.get('minLength',0)<=len(value)<=schema.get('maxLength',10**9):raise ValueError(at+': string length')
        if 'pattern' in schema and re.search(schema['pattern'],value) is None:raise ValueError(at+': pattern mismatch')
        if schema.get('format')=='date-time':
            try:parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
            except ValueError:raise ValueError(at+': invalid date-time') from None
            if 'T' not in value or parsed.utcoffset() is None:raise ValueError(at+': timezone required')
        elif 'format' in schema:raise ValueError('Unsupported format')
    if type(value) in (float,int) and not schema.get('minimum',float('-inf'))<=value<=schema.get('maximum',float('inf')):
        raise ValueError(at+': numeric bounds')


def verify_manifest(workspace,relative_manifest):
    workspace=Path(workspace).resolve()
    manifest=json.loads(workspace_path(workspace,relative_manifest).read_text(encoding='utf-8'))
    schemas={p.name:json.loads(p.read_text(encoding='utf-8')) for p in (workspace/'contracts').glob('*.schema.json')}
    validate_schema(manifest,schemas['dataset.schema.json'],schemas)
    for entry in manifest['files']:
        path=workspace_path(workspace,manifest['root_path']+'/'+entry['path'])
        if fingerprint(path)!={k:entry[k] for k in ('size_bytes','sha256')}:raise ValueError('Dataset manifest identity mismatch')
    return manifest


def read_published_report(directory,name):
    directory=Path(directory)
    if Path(name).name!=name:raise ValueError('Only report basename accepted')
    marker=json.loads((directory/'COMPLETE.json').read_text(encoding='utf-8'))
    if marker.get('publication_complete') is not True:raise ValueError('Incomplete publication')
    path=directory/name
    if path.is_symlink() or not path.is_file():raise ValueError('Nonregular report')
    if fingerprint(path)['sha256']!=marker['sha256'].get(name):raise ValueError('Report SHA256 mismatch')
    return json.loads(path.read_text(encoding='utf-8'))


def require_verified_dnn(identity,actual):
    if (identity.get('status')!='VERIFIED_KAGGLE_V5_OBJECT_CONTENT' or identity.get('upstream_md5_match') is not True
            or any(identity.get(k)!=actual[k] for k in ('size_bytes','sha256'))):
        raise ValueError('Unverified DNN identity; no comparative EDA allowed')
