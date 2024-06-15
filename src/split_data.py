import os
import random
import math

def split_dataset(dataset_size, proportions, dataset_path):
    # Garantindo que a soma das proporções seja 1.0
    assert round(sum(proportions)) == 1.0, "As proporções devem somar 1.0"
    
    # Calculando o número de imagens para cada conjunto
    train_size = math.floor(dataset_size * proportions[0])
    test_size = math.floor(dataset_size * proportions[1])
    val_size = dataset_size - train_size - test_size
    
    # Listando todas as imagens no diretório
    all_images = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
    
    print(len(all_images))
    print(dataset_size),
    # Garantindo que o número de imagens no diretório seja igual ao tamanho do dataset
    # assert len(all_images) == dataset_size, "O número de imagens no diretório não corresponde ao tamanho do dataset fornecido"
    
    # # Embaralhando as imagens para garantir aleatoriedade
    # random.shuffle(all_images)
    
    # # Dividindo as imagens de acordo com os tamanhos calculados
    # train_images = all_images[:train_size]
    # test_images = all_images[train_size:train_size + test_size]
    # val_images = all_images[train_size + test_size:]
    
    # return {
    #     "train_size": train_size,
    #     "test_size": test_size,
    #     "val_size": val_size,
    #     "train_images": train_images,
    #     "test_images": test_images,
    #     "val_images": val_images
    # }

# Exemplo de uso

