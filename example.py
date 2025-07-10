
# from ultralytics import YOLO
# # Önceden eğitilmiş ağırlıklarla modeli yükle
# model = YOLO("yolov8s-PSA.yaml")
# model.train()

import yaml
from graphviz import Digraph
# YAML dosyasını yükle
with open('ultralytics/cfg/models/v8/yolov8-PSA.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
layers = cfg['backbone'] + cfg['head']

dot = Digraph(format='png')
dot.attr(rankdir='TB')

for i, layer in enumerate(layers):
    from_, repeat, module, args = layer
    label = f"{i}: {module}"
    dot.node(str(i), label)

    # Eğer from_ bir listeyse her birini bağla
    if isinstance(from_, list):
        for idx in from_:
            src = i + idx if idx < 0 else idx
            dot.edge(str(src), str(i))
    else:
        src = i + from_ if from_ < 0 else from_
        dot.edge(str(src), str(i))

dot.render('yolo_structure_fixed_ELAP3head', view=True)