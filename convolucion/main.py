import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import fftconvolve


def recorte_centrado(img, alto, ancho):
    H, W = img.shape[:2]
    y0 = (H - alto) // 2
    x0 = (W - ancho) // 2
    return img[y0:y0 + alto, x0:x0 + ancho]


def normalizar(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    return arr.astype(np.uint8)


def procesar(path1, path2, salida):
    a = np.array(Image.open(path1).convert("RGB"), dtype=float)
    b = np.array(Image.open(path2).convert("RGB"), dtype=float)
    print(f"Dims originales: a={a.shape}, b={b.shape}")

    alto = min(a.shape[0], b.shape[0])
    ancho = min(a.shape[1], b.shape[1])
    a = recorte_centrado(a, alto, ancho)
    b = recorte_centrado(b, alto, ancho)
    print(f"Procesando {alto}x{ancho} con FFT...")

    conv = fftconvolve(a, b, mode="same", axes=(0, 1))
    # Correlación cruzada = convolución con el kernel flipeado en x e y
    corr = fftconvolve(a, np.flip(b, axis=(0, 1)), mode="same", axes=(0, 1))

    conv = normalizar(conv)
    corr = normalizar(corr)

    base, ext = salida.rsplit(".", 1)
    path_conv = f"{base}_convolucion.{ext}"
    path_corr = f"{base}_correlacion.{ext}"
    Image.fromarray(conv).save(path_conv)
    Image.fromarray(corr).save(path_corr)
    print(f"Guardados: {path_conv}, {path_corr}")

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    imagenes = [a.astype(np.uint8), b.astype(np.uint8), conv, corr]
    titulos = ["Imagen 1", "Imagen 2", "Convolución", "Correlación cruzada"]
    for ax, img, titulo in zip(axes, imagenes, titulos):
        ax.imshow(img)
        ax.set_title(titulo)
        ax.axis("off")

    plt.tight_layout()
    grafico = f"{base}_comparacion.png"
    plt.savefig(grafico, dpi=120, bbox_inches="tight")
    print(f"Gráfico guardado en {grafico}")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python main.py imagen1 imagen2 salida")
        sys.exit(1)
    procesar(sys.argv[1], sys.argv[2], sys.argv[3])