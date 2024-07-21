import os
import random
import math
import cv2
import numpy as np
import glob


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
def check_alpha_channel(image_path):
    # Carregar imagem com OpenCV
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print("Antes de se transformar: ",image.shape, image_path)
    # Verificar o número de canais
    num_channels = image.shape[2]
    if num_channels < 4:
        # Converter de BGR para RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Adicionar um canal alfa (definindo um valor fixo para alpha)
        rgba_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 4), dtype=np.uint8)
        rgba_image[:, :, :3] = rgb_image  # Copiar RGB para os primeiros 3 canais
        rgba_image[:, :, 3] = 255  # Definir canal alfa como 255 (totalmente opaco)
        print("transformada :", rgba_image.shape)

        return rgba_image
    else:
        print("NAo transformada: ", image.shape)
        return image



def load_image_paths(dataset_path, dataset="all",task="train",split=False):
    """
    dataset_path: endereço do dataset raiz
    dataset: "all", "UIEB", "RUIE", "SUIM"
    task: "train", "val"

    """
    image_paths = []
    if dataset == "all":
        # Constrói os padrões de caminho para os arquivos .jpg e .png dentro das pastas train e train/images
        pattern1_jpg = os.path.join(dataset_path, "*", f"{task}", "*.jpg")
        pattern2_jpg = os.path.join(dataset_path, "*", f"{task}", "images", "*.jpg")
        pattern1_png = os.path.join(dataset_path, "*", f"{task}", "*.png")
        pattern2_png = os.path.join(dataset_path, "*", f"{task}", "images", "*.png")
        pattern3_jpg = os.path.join(dataset_path, "*", "*",f"{task}", "*.jpg")

        
        # Encontra todos os arquivos .jpg e .png correspondentes aos padrões
        image_paths.extend(glob.glob(pattern1_jpg))
        image_paths.extend(glob.glob(pattern2_jpg))
        image_paths.extend(glob.glob(pattern1_png))
        image_paths.extend(glob.glob(pattern2_png))
        image_paths.extend(glob.glob(pattern3_jpg))
    elif dataset == "SUIM":
         pattern2_jpg = os.path.join(dataset_path, "*", f"{task}", "images", "*.jpg")
         image_paths.extend(glob.glob(pattern2_jpg))
    elif dataset == "UIEB":
        pattern1_png = os.path.join(dataset_path, "*", f"{task}", "*.png")
        image_paths.extend(glob.glob(pattern1_png))
    elif dataset == "RUIE":
        pattern3_jpg = os.path.join(dataset_path, "*", "*",f"{task}", "*.jpg")
        image_paths.extend(glob.glob(pattern3_jpg))
    
    # Embaralha os caminhos das imagens
    random.shuffle(image_paths)
    if split == True:
        # Divide os dados em 80% para treino e 20% para teste
        split_index = int(len(image_paths) * 0.8)
        train_paths = image_paths[:split_index]
        test_paths = image_paths[split_index:]
        
        return train_paths, test_paths
    else:
        return image_paths
