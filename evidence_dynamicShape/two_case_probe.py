import importlib.util, sys, time
import paddle

def load(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

paddle.set_device("gpu")
paddle.base.core._set_prim_all_enabled(True)
for i, p in enumerate(sys.argv[1:]):
    m = load(p, f"c{i}")
    paddle.seed(123)
    net = m.LayerCase(); net.eval()
    net = paddle.jit.to_static(net, backend="CINN", full_graph=True,
                               input_spec=list(m.create_inputspec()))
    x = list(m.create_tensor_inputs())
    t0 = time.time(); net(*x); paddle.device.synchronize()
    print(f"CASE {p.split('/')[-2]}/{p.split('/')[-1]} shape={list(x[0].shape)} "
          f"{(time.time()-t0)*1000:9.2f} ms", flush=True)
print("DONE", flush=True)
