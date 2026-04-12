import torch

ck = torch.load('models/best_model.pth', map_location='cpu')
keys = list(ck['model_state_dict'].keys())
print('Num keys:', len(keys))
print('Config backbone:', ck['config']['model']['backbone'])
print('Backbone guardado:', ck.get('backbone', 'NO GUARDADO'))
print('Primeras keys:', keys[:3])
print('Ultimas keys:', keys[-3:])