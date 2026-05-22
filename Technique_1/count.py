from model import PTModel as Model
from model_3D import PTModel as Model3D


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


model_1 = Model()      # depth network
model_2 = Model3D()    # residual / black-box network

t1, tr1 = count_parameters(model_1)
t2, tr2 = count_parameters(model_2)

print(f"Model 1 (depth)    : {t1:,} total | {tr1:,} trainable")
print(f"Model 2 (residual) : {t2:,} total | {tr2:,} trainable")
print(f"Technique 1 TOTAL  : {t1 + t2:,} total | {tr1 + tr2:,} trainable")