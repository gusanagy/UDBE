import os
import random
import math

def split_dataset(dataset_size, proportions, dataset_path):
    # Garantindo que a soma das proporções seja 1.0
    assert sum(proportions) == 1.0, "As proporções devem somar 1.0"
    
    # Calculando o número de imagens para cada conjunto
    train_size = math.floor(dataset_size * proportions[0])
    test_size = math.floor(dataset_size * proportions[1])
    val_size = dataset_size - train_size - test_size
    
    # Listando todas as imagens no diretório
    all_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    
    # Garantindo que o número de imagens no diretório seja igual ao tamanho do dataset
    assert len(all_images) == dataset_size, "O número de imagens no diretório não corresponde ao tamanho do dataset fornecido"
    
    # Embaralhando as imagens para garantir aleatoriedade
    random.shuffle(all_images)
    
    # Dividindo as imagens de acordo com os tamanhos calculados
    train_images = all_images[:train_size]
    test_images = all_images[train_size:train_size + test_size]
    val_images = all_images[train_size + test_size:]
    
    return {
        "train_size": train_size,
        "test_size": test_size,
        "val_size": val_size,
        "train_images": train_images,
        "test_images": test_images,
        "val_images": val_images
    }

# Exemplo de uso
dataset_size = 1000  # Tamanho do dataset
proportions = (0.7, 0.2, 0.1)  # Proporções para treino, teste e validação
dataset_path = "caminho/para/a/pasta/do/dataset"  # Caminho para a pasta do dataset

result = split_dataset(dataset_size, proportions, dataset_path)

print(f"Imagens para treino: {result['train_size']}")
print(f"Imagens para teste: {result['test_size']}")
print(f"Imagens para validação: {result['val_size']}")

print("Endereços das imagens para treino:")
for img in result['train_images']:
    print(img)

print("Endereços das imagens para teste:")
for img in result['test_images']:
    print(img)

print("Endereços das imagens para validação:")
for img in result['val_images']:
    print(img)
