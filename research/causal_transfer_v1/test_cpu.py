import importlib.util, tempfile, unittest
from pathlib import Path
import torch
from torch import nn
s=importlib.util.spec_from_file_location('probe',Path(__file__).with_name('causal_probe.py'))
p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
class Tests(unittest.TestCase):
 def test_restore_and_effect(self):
  m=nn.Sequential(nn.Linear(3,5),nn.ReLU(),nn.Linear(5,3)).eval()
  entries=p.params(m); base={k:v.detach().clone() for k,v in m.state_dict().items()}
  b={'input':torch.randn(1,3,32,32),'target':torch.randn(1,3,32,32)}
  # Distinct dummy model handles BCHW
  m=nn.Conv2d(3,3,1).eval();entries=p.params(m)
  base={k:v.detach().clone() for k,v in m.state_dict().items()}
  g=p.grad_for(m,entries,[b],torch.device('cpu'),32,.5)
  d=p.scaled_direction(g,list(range(len(entries))),1,p.norm(g))
  r=p.intervention(m,entries,d,.01,[b],torch.device('cpu'),32,.5)
  self.assertIsNotNone(r)
  self.assertTrue(all(torch.equal(v,base[k]) for k,v in m.state_dict().items()))
 def test_direction_and_group(self):
  m=nn.Linear(4,2);e=p.params(m)
  z=p.random_direction(e,2.5,33)
  self.assertAlmostEqual(p.norm(list(z.values())),2.5,places=5)
  self.assertEqual(p.group('encoders.2.0.weight'),'encoders.2')
if __name__=='__main__':unittest.main()