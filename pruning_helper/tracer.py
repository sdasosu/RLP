# depgraph_min.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Iterable

#==============================================================
# Utils
#==============================================================
def _infer_conv_type(m: nn.Module) -> Optional[str]:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        g = getattr(m, 'groups', 1)
        if g == m.in_channels and m.out_channels % m.in_channels == 0 and g > 1: return 'depthwise'
        ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
        if all(k == 1 for k in ks): return 'pointwise'
        if g > 1: return 'grouped'
        return 'standard'
    if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)): return 'transpose'
    return None

def _first_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, dict):
        for v in x.values():
            t = _first_tensor(v)
            if t is not None: return t
    if isinstance(x, (list, tuple)):
        for v in x:
            t = _first_tensor(v)
            if t is not None: return t
    return None

def _iter_tensors(x: Any) -> Iterable[torch.Tensor]:
    if isinstance(x, torch.Tensor): yield x
    elif isinstance(x, dict):
        for v in x.values(): yield from _iter_tensors(v)
    elif isinstance(x, (list, tuple)):
        for v in x: yield from _iter_tensors(v)

#==============================================================
# Dependency
#==============================================================
class DependencyGraph:
    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.module_to_node: Dict[str, int] = {}
        self.tensor_producer: Dict[int, int] = {}      # id(tensor) -> node_id
        self._hooks: List[Any] = []
        self._patched: bool = False
        self._op_idx: int = 0
        self._nid: int = 0
        self._orig: Dict[str, Any] = {}

    #==============================================================
    # Build
    #==============================================================
    def build(self, model: nn.Module, example_input: torch.Tensor):
        for name, m in model.named_modules():
            if len(list(m.children())) == 0:
                h = m.register_forward_hook(self._make_module_hook(name, m))
                self._hooks.append(h)

        self._patch()
        model.eval()
        try:
            x = example_input.detach().requires_grad_(True)
            y = model(x)
            t = _first_tensor(y)
            if t is None: raise RuntimeError("no tensor output")
            t.sum().backward()
        finally:
            self._unpatch()
            for h in self._hooks: h.remove()
        return self

    #==============================================================
    # Nodes
    #==============================================================
    def _new_node(self, name: str, typ: str, conv_type: Optional[str], is_module: bool):
        nid = self._nid; self._nid += 1
        node = {
            'id'        : nid,
            'name'      : name,
            'type'      : typ,
            'conv_type' : conv_type,
            'is_module' : is_module,
            'is_operation': not is_module,
            'inputs'    : set(),   # store node ids
            'outputs'   : set(),
        }
        self.nodes.append(node)
        return node

    def _link_parents(self, node: Dict[str, Any], parents: Iterable[int]):
        nid = node['id']
        for pid in parents:
            if pid is None: continue
            if pid == nid: continue
            if pid not in node['inputs']:
                node['inputs'].add(pid)
                self.nodes[pid]['outputs'].add(nid)

    #==============================================================
    # Hooks
    #==============================================================
    def _make_module_hook(self, name: str, module: nn.Module):
        def hook(mod, inputs, output):
            node = self._new_node(name=name, typ=type(module).__name__,
                                  conv_type=_infer_conv_type(module), is_module=True)
            self.module_to_node[name] = node['id']

            parent_ids = set()
            for t in _iter_tensors(inputs):
                pid = self.tensor_producer.get(id(t))
                if pid is not None: parent_ids.add(pid)
            self._link_parents(node, parent_ids)

            for t in _iter_tensors(output):
                self.tensor_producer[id(t)] = node['id']
        return hook

    #==============================================================
    # Patch
    #==============================================================
    def _patch(self):
        if self._patched: return
        dg = self

        self._orig = {
            'add': torch.Tensor.__add__, 'radd': torch.Tensor.__radd__, 'iadd': torch.Tensor.__iadd__,
            'sub': torch.Tensor.__sub__, 'rsub': torch.Tensor.__rsub__, 'isub': torch.Tensor.__isub__,
            'mul': torch.Tensor.__mul__, 'rmul': torch.Tensor.__rmul__, 'imul': torch.Tensor.__imul__,
            'truediv': torch.Tensor.__truediv__, 'rtruediv': torch.Tensor.__rtruediv__, 'itruediv': torch.Tensor.__itruediv__,
            'matmul': torch.Tensor.__matmul__, 'rmatmul': torch.Tensor.__rmatmul__,
            'view': torch.Tensor.view, 'reshape': torch.Tensor.reshape, 'permute': torch.Tensor.permute,
            'cat': torch.cat, 'stack': torch.stack, 'flatten': torch.flatten, 'transpose': torch.transpose,
            'relu': F.relu, 'gelu': getattr(F, 'gelu', None), 'silu': getattr(F, 'silu', None),
            'sigmoid': F.sigmoid, 'softmax': F.softmax,
            'interpolate': F.interpolate, 'linear': F.linear,
            'avg_pool2d': F.avg_pool2d, 'max_pool2d': F.max_pool2d,
        }

        def T(op, out, ins_tensors):
            node = dg._new_node(name=f"{op}_{dg._op_idx}", typ=op, conv_type=None, is_module=False)
            dg._op_idx += 1
            parents = set()
            for t in ins_tensors:
                pid = dg.tensor_producer.get(id(t))
                if pid is not None: parents.add(pid)
            dg._link_parents(node, parents)
            for t in _iter_tensors(out):
                dg.tensor_producer[id(t)] = node['id']
            return out

        # binary
        def t_add(a,b):    return T('add',    self._orig['add'](a,b),    [a,b])
        def t_radd(a,b):   return T('add',    self._orig['radd'](a,b),   [a,b])
        def t_iadd(a,b):   return T('add_',   self._orig['iadd'](a,b),   [a,b])
        def t_sub(a,b):    return T('sub',    self._orig['sub'](a,b),    [a,b])
        def t_rsub(a,b):   return T('sub',    self._orig['rsub'](a,b),   [a,b])
        def t_isub(a,b):   return T('sub_',   self._orig['isub'](a,b),   [a,b])
        def t_mul(a,b):    return T('mul',    self._orig['mul'](a,b),    [a,b])
        def t_rmul(a,b):   return T('mul',    self._orig['rmul'](a,b),   [a,b])
        def t_imul(a,b):   return T('mul_',   self._orig['imul'](a,b),   [a,b])
        def t_div(a,b):    return T('div',    self._orig['truediv'](a,b),[a,b])
        def t_rdiv(a,b):   return T('div',    self._orig['rtruediv'](a,b),[a,b])
        def t_idiv(a,b):   return T('div_',   self._orig['itruediv'](a,b),[a,b])
        def t_matmul(a,b): return T('matmul', self._orig['matmul'](a,b), [a,b])
        def t_rmatmul(a,b):return T('matmul', self._orig['rmatmul'](a,b),[a,b])

        # shape/layout
        def t_view(x,*s):        return T('view',      self._orig['view'](x,*s),       [x])
        def t_reshape(x,*s):     return T('reshape',   self._orig['reshape'](x,*s),    [x])
        def t_permute(x,*d):     return T('permute',   self._orig['permute'](x,*d),    [x])
        def t_flatten(x,a=0,b=-1): return T('flatten', self._orig['flatten'](x,a,b),   [x])
        def t_transpose(x,i,j):  return T('transpose', self._orig['transpose'](x,i,j), [x])

        # joins
        def t_cat(ts, dim=0):    return T('cat',   self._orig['cat'](ts, dim),   list(ts))
        def t_stack(ts, dim=0):  return T('stack', self._orig['stack'](ts, dim), list(ts))

        # activations
        def t_relu(x, inplace=False):
            y = self._orig['relu'](x, inplace=inplace)
            return T('relu_' if inplace else 'relu', y, [x])
        def t_gelu(x, approximate='none'):
            y = self._orig['gelu'](x, approximate=approximate) if self._orig['gelu'] else torch.nn.GELU()(x)
            return T('gelu', y, [x])
        def t_silu(x):
            y = self._orig['silu'](x) if self._orig['silu'] else torch.nn.SiLU()(x)
            return T('silu', y, [x])
        def t_sigmoid(x):        return T('sigmoid', self._orig['sigmoid'](x), [x])
        def t_softmax(x, dim=None): return T('softmax', self._orig['softmax'](x, dim=dim), [x])

        # others
        def t_interpolate(x, *a, **k): return T('interpolate', self._orig['interpolate'](x,*a,**k), [x])
        def t_linear(x, w, b=None):    return T('linear',      self._orig['linear'](x,w,b),        [x])
        def t_avgpool(x, *a, **k):     return T('avg_pool2d',  self._orig['avg_pool2d'](x,*a,**k), [x])
        def t_maxpool(x, *a, **k):     return T('max_pool2d',  self._orig['max_pool2d'](x,*a,**k), [x])

        # apply
        torch.Tensor.__add__     = t_add
        torch.Tensor.__radd__    = t_radd
        torch.Tensor.__iadd__    = t_iadd
        torch.Tensor.__sub__     = t_sub
        torch.Tensor.__rsub__    = t_rsub
        torch.Tensor.__isub__    = t_isub
        torch.Tensor.__mul__     = t_mul
        torch.Tensor.__rmul__    = t_rmul
        torch.Tensor.__imul__    = t_imul
        torch.Tensor.__truediv__ = t_div
        torch.Tensor.__rtruediv__= t_rdiv
        torch.Tensor.__itruediv__= t_idiv
        torch.Tensor.__matmul__  = t_matmul
        torch.Tensor.__rmatmul__ = t_rmatmul
        torch.Tensor.view        = t_view
        torch.Tensor.reshape     = t_reshape
        torch.Tensor.permute     = t_permute
        torch.flatten            = t_flatten
        torch.transpose          = t_transpose
        torch.cat                = t_cat
        torch.stack              = t_stack
        F.relu                   = t_relu
        if self._orig['gelu'] is not None:   F.gelu = t_gelu
        if self._orig['silu'] is not None:   F.silu = t_silu
        F.sigmoid                = t_sigmoid
        F.softmax                = t_softmax
        F.interpolate            = t_interpolate
        F.linear                 = t_linear
        F.avg_pool2d             = t_avgpool
        F.max_pool2d             = t_maxpool

        self._patched = True

    def _unpatch(self):
        if not self._patched: return
        for k,v in self._orig.items():
            if k in ('gelu','silu') and v is None: continue
        torch.Tensor.__add__     = self._orig['add']
        torch.Tensor.__radd__    = self._orig['radd']
        torch.Tensor.__iadd__    = self._orig['iadd']
        torch.Tensor.__sub__     = self._orig['sub']
        torch.Tensor.__rsub__    = self._orig['rsub']
        torch.Tensor.__isub__    = self._orig['isub']
        torch.Tensor.__mul__     = self._orig['mul']
        torch.Tensor.__rmul__    = self._orig['rmul']
        torch.Tensor.__imul__    = self._orig['imul']
        torch.Tensor.__truediv__ = self._orig['truediv']
        torch.Tensor.__rtruediv__= self._orig['rtruediv']
        torch.Tensor.__itruediv__= self._orig['itruediv']
        torch.Tensor.__matmul__  = self._orig['matmul']
        torch.Tensor.__rmatmul__ = self._orig['rmatmul']
        torch.Tensor.view        = self._orig['view']
        torch.Tensor.reshape     = self._orig['reshape']
        torch.Tensor.permute     = self._orig['permute']
        torch.flatten            = self._orig['flatten']
        torch.transpose          = self._orig['transpose']
        torch.cat                = self._orig['cat']
        torch.stack              = self._orig['stack']
        F.relu                   = self._orig['relu']
        if self._orig['gelu'] is not None:   F.gelu = self._orig['gelu']
        if self._orig['silu'] is not None:   F.silu = self._orig['silu']
        F.sigmoid                = self._orig['sigmoid']
        F.softmax                = self._orig['softmax']
        F.interpolate            = self._orig['interpolate']
        F.linear                 = self._orig['linear']
        F.avg_pool2d             = self._orig['avg_pool2d']
        F.max_pool2d             = self._orig['max_pool2d']
        self._patched = False

    #==============================================================
    # Print
    #==============================================================
    def print_graph(self, limit: int = 60):
        print(f"\nDependency Graph: {len(self.nodes)} nodes")
        print("="*80)
        for i, n in enumerate(self.nodes[:limit]):
            typ = n['type']
            if n['conv_type']: typ = f"{typ} ({n['conv_type']})"
            if n['is_operation']: typ = f"OP: {typ}"
            ins = [self.nodes[j]['name'] for j in list(n['inputs'])[:3]]
            outs = [self.nodes[j]['name'] for j in list(n['outputs'])[:3]]
            print(f"{i:3d}. {n['name']:40} | {typ}")
            if ins:  print(f"      ← {ins}")
            if outs: print(f"      → {outs}")
        if len(self.nodes) > limit: print(f"... and {len(self.nodes)-limit} more")
        print("="*80)

#----------------------------------------------------------------------
def build_depgraph(model, example_input: torch.Tensor):               #
    model = model.to(example_input.device).eval()                     #
    return DependencyGraph().build(model, example_input)              #
#----------------------------------------------------------------------