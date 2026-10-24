"""Rewrite every single-step ONNX GRU node (seq_len 1, batch 1, linear_before_reset=1) as MatMul/Sigmoid/Tanh ops,
so onnxruntime's quantize_dynamic can turn the recurrent weights into INT8 MatMulInteger. External idea 2026-10-24.
    python scratch/gru_unroll_onnx.py SRC_DIR OUT_DIR [--int8]
"""
import os, shutil, sys, numpy as np, onnx
from onnx import helper as H, numpy_helper as nh
src, out = sys.argv[1], sys.argv[2]; int8 = '--int8' in sys.argv
m = onnx.load(f'{src}/model.onnx'); g = m.graph; init = {t.name: t for t in g.initializer}
new_nodes, new_init = [], []
def const(name, arr): new_init.append(nh.from_array(np.ascontiguousarray(arr), name)); return name
for n in g.node:
    if n.op_type != 'GRU':
        new_nodes.append(n); continue
    at = {a.name: H.get_attribute_value(a) for a in n.attribute}
    assert at.get('linear_before_reset') == 1 and at.get('direction', b'forward') in (b'forward', 'forward'), at
    Hd = at['hidden_size']; X, W, R, B, seqlen, h0 = (list(n.input) + [''] * 6)[:6]
    assert seqlen == '' and h0, n.input
    Wa, Ra, Ba = nh.to_array(init[W])[0], nh.to_array(init[R])[0], nh.to_array(init[B])[0]   # (3H,in) (3H,H) (6H) gates z,r,h
    p = n.name.replace('/', '_')
    Wt, Rt = const(p + '_Wt', Wa.T.astype(np.float32)), const(p + '_Rt', Ra.T.astype(np.float32))
    Wb, Rb = const(p + '_Wb', Ba[:3 * Hd].astype(np.float32)), const(p + '_Rb', Ba[3 * Hd:].astype(np.float32))
    s2 = const(p + '_s2', np.array([1, -1], np.int64)); sY = const(p + '_sY', np.array([1, 1, 1, Hd], np.int64)); sH = const(p + '_sH', np.array([1, 1, Hd], np.int64))
    sp = const(p + '_split', np.array([Hd, Hd, Hd], np.int64))
    t = lambda k: f'{p}_{k}'
    new_nodes += [
        H.make_node('Reshape', [X, s2], [t('x')]), H.make_node('Reshape', [h0, s2], [t('h')]),
        H.make_node('MatMul', [t('x'), Wt], [t('gi0')]), H.make_node('Add', [t('gi0'), Wb], [t('gi')]),
        H.make_node('MatMul', [t('h'), Rt], [t('gh0')]), H.make_node('Add', [t('gh0'), Rb], [t('gh')]),
        H.make_node('Split', [t('gi'), sp], [t('iz'), t('ir'), t('ih')], axis=1),
        H.make_node('Split', [t('gh'), sp], [t('hz'), t('hr'), t('hh')], axis=1),
        H.make_node('Add', [t('iz'), t('hz')], [t('z0')]), H.make_node('Sigmoid', [t('z0')], [t('z')]),
        H.make_node('Add', [t('ir'), t('hr')], [t('r0')]), H.make_node('Sigmoid', [t('r0')], [t('r')]),
        H.make_node('Mul', [t('r'), t('hh')], [t('rh')]), H.make_node('Add', [t('ih'), t('rh')], [t('n0')]), H.make_node('Tanh', [t('n0')], [t('n')]),
        H.make_node('Sub', [t('h'), t('n')], [t('hn')]), H.make_node('Mul', [t('z'), t('hn')], [t('zhn')]), H.make_node('Add', [t('n'), t('zhn')], [t('hnew')]),
    ]
    outs = list(n.output) + [''] * 2
    if outs[0]: new_nodes.append(H.make_node('Reshape', [t('hnew'), sY], [outs[0]]))
    if outs[1]: new_nodes.append(H.make_node('Reshape', [t('hnew'), sH], [outs[1]]))
used = {i for nd in new_nodes for i in nd.input}
keep = [t for t in g.initializer if t.name in used]
del g.node[:]; g.node.extend(new_nodes); del g.initializer[:]; g.initializer.extend(keep + new_init)
onnx.checker.check_model(m)
shutil.copytree(src, out, dirs_exist_ok=True)
onnx.save(m, f'{out}/model.onnx')
if int8:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(f'{out}/model.onnx', f'{out}/model_q.onnx', weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'])
    os.replace(f'{out}/model_q.onnx', f'{out}/model.onnx')
print('wrote', out, os.path.getsize(f'{out}/model.onnx'), 'bytes')
