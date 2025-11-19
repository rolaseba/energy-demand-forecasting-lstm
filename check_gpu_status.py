import torch
import tensorflow as tf

print("="*60)
print("🔍 GPU STATUS CHECK".center(60))
print("="*60)

# ---- PyTorch ----
print("\n🧠 PyTorch:")
try:
    torch_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if torch_available else "CPU"
    print(f"  • CUDA disponible: {torch_available}")
    print(f"  • Dispositivo en uso: {device_name}")
except Exception as e:
    print(f"  ⚠️ Error al verificar PyTorch: {e}")

# ---- TensorFlow ----
print("\n🤖 TensorFlow:")
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"  • GPU detectada: {gpus[0].name}")
    else:
        print("  • No se detectó GPU (usando CPU)")
except Exception as e:
    print(f"  ⚠️ Error al verificar TensorFlow: {e}")

print("\n✅ Verificación completa.")
print("="*60)