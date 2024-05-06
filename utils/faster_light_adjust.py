
import numpy as np
import cv2
import time


def adjust_brightness_with_for(image, alpha, beta):
    new_image = np.zeros_like(image)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(image.shape[2]):
                new_image[y, x, c] = np.clip(alpha * image[y, x, c] + beta, 0, 255)
                
    return new_image
def adjust_brightness_vectorized(image, alpha, beta):
    new_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return new_image
def adjust_brightness_opencv(image, alpha, beta):
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return new_image

# Carregar imagem
image = cv2.imread("LOL results/1.png")

# Medir tempo de execução usando loop `for`
start_time = time.time()
adjusted_image_for = adjust_brightness_with_for(image, 1.0, -140)
end_time = time.time()
print(f"Tempo usando loop `for`: {end_time - start_time:.4f} segundos")

# Medir tempo de execução usando operações vetoriais
start_time = time.time()
adjusted_image_vectorized = adjust_brightness_vectorized(image, 1.0, -140)
end_time = time.time()
print(f"Tempo usando operações vetoriais: {end_time - start_time:.4f} segundos")

# Medir tempo de execução usando cv.convertScaleAbs()
start_time = time.time()
adjusted_image_opencv = adjust_brightness_opencv(image, 1.0, -140)
end_time = time.time()
print(f"Tempo usando cv.convertScaleAbs(): {end_time - start_time:.4f} segundos")