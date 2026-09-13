import os
import shutil
import numpy as np
import onnx
import onnxsim

def optimize():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models/4_model/onnx_models/no_batch_size'))
    backup_dir = os.path.join(models_dir, 'backup_original')
    os.makedirs(backup_dir, exist_ok=True)

    targets = ['dms_feature_extractor1.onnx', 'dms_tcn_classifier1.onnx']

    for name in targets:
        model_path = os.path.join(models_dir, name)
        bak_path = os.path.join(backup_dir, name)

        if not os.path.exists(bak_path):
            shutil.copy2(model_path, bak_path)
            print(f'[BACKUP] {name} -> {backup_dir}')

        source_path = bak_path if os.path.exists(bak_path) else model_path
        model = onnx.load(source_path)
        nodes_before = len(model.graph.node)
        size_before = os.path.getsize(source_path)

        print(f'\n[OPTIMIZE] Dang toi uu hoa {name} bang onnx-simplifier...')
        model_sim, check = onnxsim.simplify(model)
        assert check, f'Toi uu hoa that bai cho {name}'

        nodes_after = len(model_sim.graph.node)
        onnx.save(model_sim, model_path)
        size_after = os.path.getsize(model_path)

        print(f'[DONE] Hoan tat {name}:')
        print(f'     Nodes : {nodes_before} -> {nodes_after} (delta: {nodes_after - nodes_before} nodes)')
        print(f'     Size  : {size_before/(1024*1024):.2f}MB -> {size_after/(1024*1024):.2f}MB')

if __name__ == '__main__':
    optimize()
