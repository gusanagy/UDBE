import wandb
import argparse
from src import train
from src.train import train, Test, Inference

# Função para mesclar valores do modelConfig no parser
def update_config_with_model_config(parser, model_config):
    """
    Atualiza os argumentos do parser com os valores do dicionário model_config.
    Caso o argumento já esteja no parser, ele será sobrescrito.
    """
    for key, value in model_config.items():
        # Verifica se o argumento já existe no parser
        if not any(action.dest == key for action in parser._actions):
            parser.add_argument(f'--{key}', type=type(value), default=value)
    return parser

# Inicializa classes e configura o parser
if __name__ == "__main__":
    # Configurações básicas do modelo
    modelConfig = {
        "DDP": False,
        #"state": "inference", # or eval
        "epoch": 1000,
        "batch_size": 16,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 5e-5,
        "multiplier": 2.0,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.0,
        "device": "cuda",  # MODIFICADO
        "device_list": [0, 1],
        "ddim": True,
        "unconditional_guidance_scale": 1,
        "ddim_step": 100,
    }

    # Configurações de argparse
    parser = argparse.ArgumentParser(description="Pipeline de Treinamento/Inferência")
    parser.add_argument('--dataset', type=str, default="all")
    parser.add_argument('--model', type=str, default="standart")#mask is the second option
    parser.add_argument('--dataset_path', type=str, default="./dataset/UDWdata/")
    parser.add_argument('--state', type=str, default="train")  #or eval
    parser.add_argument('--pretrained_path', type=str, default=None)  #or eval
    parser.add_argument('--inference_image', type=str, default="data/UDWdata/UIEB/val/205_img_.png")  #or eval
    parser.add_argument('--output_path', type=str, default="./output/")  #or eval
    parser.add_argument('--wandb', type=bool, default=False)  #or False
    parser.add_argument('--wandb_name', type=str, default="GLDiffusion")
    parser.add_argument('--epoch', type=int, default=1000)

    # Adiciona os valores do modelConfig ao parser
    parser = update_config_with_model_config(parser, modelConfig)

    # Converte os argumentos do parser para um namespace
    config = parser.parse_args()

    # Inicialização opcional do wandb
    if config.wandb:
        wandb.init(
            project=config.wandb_name,
            config=vars(config),
            name=f"{config.state}_{config.wandb_name}_{config.dataset}",
            tags=[config.state, config.dataset],
            group="Branch glown_diffusion_test",
            job_type="test"
        )

    # Debugging: Imprime as configurações carregadas
    print(config)

    print(config.epoch)

    if config.state == 'val':
        Test(config, config.epoch)
    elif config.state == 'train':
        train(config)
    elif config.state == 'inference':
        Inference(config, config.epoch)
    else:
        print("Invalid state")

    # Finaliza wandb, se estiver ativo
    if config.wandb:
        wandb.finish()

    

