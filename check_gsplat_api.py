import gsplat
from gsplat.rendering import rasterization, fully_fused_projection
import inspect

print("gsplat version:", gsplat.__version__)
print()
print("=== rasterization ===")
print(inspect.signature(rasterization))
print()
print("=== fully_fused_projection ===")
print(inspect.signature(fully_fused_projection))
