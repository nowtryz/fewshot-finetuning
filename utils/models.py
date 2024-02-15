import torch

from pretrain.models.model_pretrain import SwinUNETR


def load_weights(model, weights_path, classifier=False):
    print('Use pretrained weights')

    model_dict = torch.load(weights_path)
    if 'state_dict' in list(model_dict.keys()):
        state_dict = model_dict["state_dict"]
    else:
        state_dict = model_dict
    # fix potential differences in state dict keys from pre-training to fine-tuning
    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)
    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    store_dict = {}
    for key in state_dict.keys():
        if 'classifier' not in key:
            store_dict[key] = state_dict[key]
        else:
            if classifier and 'classifier' in key:
                store_dict[key] = state_dict[key]

    model.load_state_dict(store_dict, strict=False)

    return model

def default_model(args, num_classes):
    return SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z), in_channels=1, out_channels=num_classes,
                        feature_size=48, drop_rate=0.0, attn_drop_rate=0.0, dropout_path_rate=0.0,
                        use_checkpoint=False)
