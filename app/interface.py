import torch
import gradio as gr

def verificar_cuda():
    disponible = torch.cuda.is_available()
    nombre = torch.cuda.get_device_name(0) if disponible else "No hay GPU disponible"
    return f"CUDA disponible: {disponible}\nGPU: {nombre}"

iface = gr.Interface(fn=verificar_cuda, inputs=[], outputs="text")
if __name__ == "__main__":
    import os
    print(" Si no se abre automáticamente, ve a: http://localhost:7860")
    iface.launch(server_name="0.0.0.0", server_port=7860, show_error=True)