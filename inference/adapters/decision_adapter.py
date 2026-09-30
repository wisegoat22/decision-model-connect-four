#!/usr/bin/env python3
"""Pinned local Von/Laya adapters: one JSON request/response per line, no game logic."""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os, pathlib, sys, time, traceback
ROOT = pathlib.Path(__file__).resolve().parent
os.environ['USE_TF'] = '0'
os.environ['VON_CHAINS_DIR'] = 'off'
os.environ['VON_ON_OVERFLOW'] = 'refuse'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
IDENTITIES = {
 'von': {'name':'Von 1.3.1 runtime, 1.2 weights, chains off', 'sdk':'von-sdk==1.3.1', 'weights_repo':'wfzyx/von', 'weights_revision':'43bbca0fb17f424539416ee8eb3de33cfc3dacf3','source_commit':'a867a49d56d147769f2180629f05ac86f5cc1b51'},
 'laya': {'name':'Laya English 421M', 'sdk':'laya==0.3.21', 'weights_repo':'convaiinnovations/laya', 'weights_revision':'55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851','source_commit':'9d955671415fc19f069b9cc998928075c1f255ec'},
}
def sha(path):
 with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def verify(model):
 m=json.loads((ROOT/'weights_manifest.json').read_text())
 for a in m['artifacts']:
  if a['model']!=model: continue
  p=ROOT/'weights'/model/a['file']
  if p.stat().st_size!=a['bytes'] or sha(p)!=a['sha256']:raise RuntimeError(f'Artifact changed: {p}')
 if model=='laya':
  m=json.loads((ROOT/'encoder_config_manifest.json').read_text())
  if sha(ROOT/'weights/laya/encoder/config.json')!=m['sha256']:raise RuntimeError('Laya encoder config changed')

def load(model, device):
 import torch
 torch.set_num_threads(4)
 torch.set_num_interop_threads(1)
 if device=='mps':
  if not torch.backends.mps.is_available():raise RuntimeError('MPS explicitly requested but unavailable')
  torch.mps.set_per_process_memory_fraction(0.35)
 verify(model)
 if model=='von':
  from von.backends.option_marker_backend import OptionMarkerBackend
  agent=OptionMarkerBackend(checkpoint_dir=str(ROOT/'weights/von'), device=device)
  assert agent.chain_runner is None
  agent._get_model()
  assert agent._independent_options, 'Pinned Von should use independent option attention'
 else:
  import laya
  agent=laya.load(str(ROOT/'weights/laya'),device=device,fast=False,compile=False)
 return agent

def predict(model, agent, request):
 import torch
 if set(request)!= {'state','questions'}:raise ValueError('Expected exactly state and questions')
 if not isinstance(request['state'],str):raise ValueError('State must be common plain text')
 if set(request['questions'])!={'move'}:raise ValueError('Expected one move question')
 q=request['questions']['move']
 if q.get('type')!='choice' or not isinstance(q.get('criteria'),dict) or not q['criteria']:raise ValueError('Expected nonempty choice criteria')
 if any(k not in {'c1','c2','c3','c4','c5','c6','c7'} for k in q['criteria']):raise ValueError('Unexpected column id')
 start=time.perf_counter()
 if model=='von':
  result=agent.evaluate(request['state'],request['questions']).model_dump()
  truncated=bool(result.get('truncation'))
 else:
  internal={'move':agent._to_internal(q)}
  actual=agent._encode_state(request['state'],['move'],internal)
  unbounded=agent._encode_state(request['state'],['move'],internal,max_len=8192,head_max_len=8192)
  truncated=actual[0]['ids']!=unbounded[0]['ids']
  if truncated:raise ValueError('Laya state/question/choices would be truncated')
  result=agent.predict(request['state'],request['questions'])
 if truncated:raise ValueError('Truncation is forbidden')
 if str(agent.device)=='mps':torch.mps.synchronize()
 answer=result['answers']['move'];choice=answer['choice']
 if choice not in q['criteria']:raise RuntimeError('Model returned an unavailable choice')
 return {'choice':choice,'metadata':{
  **IDENTITIES[model], 'device':str(agent.device), 'input_tokens':result['usage']['input_tokens'],
  'truncated':False, 'chains_enabled':False,'model_forward_passes':1,
  'probabilities':answer['probabilities'],'confidence':answer.get('confidence'),
  'elapsed_ms':round((time.perf_counter()-start)*1000,3),
 }}

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=IDENTITIES,required=True);p.add_argument('--device',choices=['cpu','mps'],default='cpu');a=p.parse_args()
 with contextlib.redirect_stdout(sys.stderr):agent=load(a.model,a.device)
 print(json.dumps({'status':'ready','model':a.model,'device':str(agent.device)}),file=sys.stderr,flush=True)
 for line in sys.stdin:
  if not line.strip():continue
  try:
   with contextlib.redirect_stdout(sys.stderr):res=predict(a.model,agent,json.loads(line))
   print(json.dumps(res,separators=(',',':'),allow_nan=False),flush=True)
  except Exception as e:
   traceback.print_exc(file=sys.stderr)
   print(json.dumps({'error':type(e).__name__,'message':str(e)}),flush=True)
if __name__=='__main__':main()
