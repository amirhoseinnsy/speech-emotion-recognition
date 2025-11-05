import torch, yaml, os
from scripts.train import train, set_seed
from data.data_loader import get_dataloader_cremad
from models.model import HubertEmotionClassifier, MelCNNEmotionClassifier
from scripts.evaluate import evaluate_model

def build_model(cfg):
    ft = cfg["data"]["feature_type"]
    if ft == "hubert":
        return HubertEmotionClassifier(
            num_classes=cfg["model"]["num_classes"],
            pretrained_model=cfg["model"]["pretrained"],
            dropout=cfg["model"]["dropout"],
            mlp_layers=cfg["model"]["hubert_mlp_layers"]
        )
    elif ft == "mel":
        return MelCNNEmotionClassifier(
            num_classes=cfg["model"]["num_classes"],
            n_mels=cfg["data"]["n_mels"],
            dropout=cfg["model"]["dropout"],
            conv_layers=cfg["model"]["mel_conv_layers"]
        )
    else:
        raise ValueError(f"Unknown feature_type: {ft}")
    
def load_and_evaluate(cfg, model_type, checkpoint_path, test_loader, device, class_names):
    if model_type == "hubert":
        model = HubertEmotionClassifier(
            num_classes=cfg["model"]["num_classes"],
            pretrained_model=cfg["model"]["pretrained"],
            dropout=cfg["model"]["dropout"],
            mlp_layers=cfg["model"]["hubert_mlp_layers"]
        ).to(device)
    elif model_type == "mel":
        model = MelCNNEmotionClassifier(
            num_classes=cfg["model"]["num_classes"],
            n_mels=cfg["data"]["n_mels"],
            dropout=cfg["model"]["dropout"],
            conv_layers=cfg["model"]["mel_conv_layers"]
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ Loaded {model_type} model from {checkpoint_path}")

    # Evaluate
    report, cm = evaluate_model(model, test_loader, device, class_names)
    return report, cm

def train_():
    cfg = yaml.safe_load(open("config/config.yaml", "r"))
    set_seed(cfg["experiment"]["seed"])
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloader_cremad(
        root=cfg["data"]["root"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["experiment"]["seed"],
        feature_type=cfg["data"]["feature_type"],
        max_speakers=cfg["data"]["max_speakers"],
        sample_rate=cfg["data"]["sample_rate"],
        seconds=cfg["data"]["seconds"],
        n_mels=cfg["data"]["n_mels"],
    )

    model = build_model(cfg).to(device)

    opt_name = cfg["training"]["optimizer"].lower()
    Optim = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[opt_name]
    optimizer = Optim(model.parameters(), lr=float(cfg["training"]["lr"]),
                      weight_decay=float(cfg["training"]["weight_decay"]))

    train(
        model, train_loader, val_loader, device, optimizer,
        epochs=cfg["training"]["epochs"],
        save_epoch=cfg["training"]["save_epoch"],
        patience=cfg["training"]["patience"],
        log_file=cfg["training"]["log_file"],
    )




if __name__ == "__main__":
    train_()

    # class_names = ["Angry", "Happy", "Neutral", "Sad"]
    # cfg = yaml.safe_load(open("config/config.yaml", "r"))
    # set_seed(cfg["experiment"]["seed"])
    # device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # train_loader, val_loader, test_loader = get_dataloader_cremad(
    #     root=cfg["data"]["root"],
    #     batch_size=cfg["training"]["batch_size"],
    #     num_workers=cfg["data"]["num_workers"],
    #     seed=cfg["experiment"]["seed"],
    #     feature_type=cfg["data"]["feature_type"],
    #     max_speakers=cfg["data"]["max_speakers"],
    #     sample_rate=cfg["data"]["sample_rate"],
    #     seconds=cfg["data"]["seconds"],
    #     n_mels=cfg["data"]["n_mels"],
    # )


    # Evaluate saved HuBERT
    # hubert_report, hubert_cm = load_and_evaluate(
    #     cfg,
    #     model_type="hubert",
    #     checkpoint_path="models/saved_models/hubert/Final.pth",
    #     test_loader=test_loader,
    #     device=device,
    #     class_names=class_names
    # )

    # mel_report, mel_cm = load_and_evaluate(
    #     cfg,
    #     model_type="mel",
    #     checkpoint_path="models/saved_models/mel/Final.pth",
    #     test_loader=test_loader,
    #     device=device,
    #     class_names=class_names
    # )


