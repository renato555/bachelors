import tensorflow as tf
import tensorflow.keras as keras

models = ["effnetB0", "effnetB1", "effnetB2", "effnetB3", "effnetv2B0",
    "effnetv2B1", "effnetv2B2", "effnetv2B3", "densenet121", "densenet169",
    "resnet101v2", "resnet152v2" , "mobilenetv2"]

def get_backbone(model_name):
    if model_name == "effnetB0":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.efficientnet.preprocess_input(x)),
                        keras.applications.efficientnet.EfficientNetB0(weights="imagenet", include_top=False)])
    elif model_name == "effnetB1":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.efficientnet.preprocess_input(x)),
                        keras.applications.efficientnet.EfficientNetB1(weights="imagenet", include_top=False)])
    elif model_name == "effnetB2":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.efficientnet.preprocess_input(x)),
                        keras.applications.efficientnet.EfficientNetB2(weights="imagenet", include_top=False)])
    elif model_name == "effnetB3":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.efficientnet.preprocess_input(x)),
                        keras.applications.efficientnet.EfficientNetB3(weights="imagenet", include_top=False)])
    elif model_name == "effnetv2B0":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.efficientnet_v2.preprocess_input(x)),
                        keras.applications.efficientnet_v2.EfficientNetV2B0(weights="imagenet", include_top=False)])
    elif model_name == "effnetv2B1":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.efficientnet_v2.preprocess_input(x)),
                        keras.applications.efficientnet_v2.EfficientNetV2B1(weights="imagenet", include_top=False)])
    elif model_name == "effnetv2B2":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.efficientnet_v2.preprocess_input(x)),
                        keras.applications.efficientnet_v2.EfficientNetV2B2(weights="imagenet", include_top=False)])
    elif model_name == "effnetv2B3":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.efficientnet_v2.preprocess_input(x)),
                        keras.applications.efficientnet_v2.EfficientNetV2B3(weights="imagenet", include_top=False)])
    elif model_name == "densenet121":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.densenet.preprocess_input(x)),
                        keras.applications.densenet.DenseNet121(weights="imagenet", include_top=False)])
    elif model_name == "densenet169":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.densenet.preprocess_input(x)),
                        keras.applications.densenet.DenseNet169(weights="imagenet", include_top=False)])
    elif model_name == "resnet101v2":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.resnet_v2.preprocess_input(x)),
                        keras.applications.resnet_v2.ResNet101V2(weights="imagenet", include_top=False)])
    elif model_name == "resnet152v2":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.resnet_v2.preprocess_input(x)),
                        keras.applications.resnet_v2.ResNet152V2(weights="imagenet", include_top=False)])
    elif model_name == "mobilenetv2":
        return keras.models.Sequential([
                        keras.layers.Lambda( lambda x: keras.applications.mobilenet_v2.preprocess_input(x)),
                        keras.applications.mobilenet_v2.MobileNetV2(weights="imagenet", include_top=False)])

def get_image_size(model_name):
    if model_name == "effnetB0":
        return 224
    elif model_name == "effnetB1":
        return 240
    elif model_name == "effnetB2":
        return 260
    elif model_name == "effnetB3":
        return 300
    elif model_name == "effnetv2B0":
        return 224
    elif model_name == "effnetv2B1":
        return 240
    elif model_name == "effnetv2B2":
        return 260
    elif model_name == "effnetv2B3":
        return 300
    elif model_name == "densenet121":
        return 224
    elif model_name == "densenet169":
        return 224
    elif model_name == "resnet101v2":
        return 224
    elif model_name == "resnet152v2":
        return 224
    elif model_name == "mobilenetv2":
        return 224

def get_supported_backbones():
    return models

def get_classification_head(N_CLASSES, dropout_rate=0.7):
    head = keras.models.Sequential([
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=dropout_rate),
        keras.layers.Dense(N_CLASSES, activation="softmax")
    ])

    return head

def get_classification_model(N_CLASSES, backbone_name, head_dropout_rate=0.7):
    if backbone_name not in models:
        raise ValueError("this model name is not supported")

    backbone = get_backbone(backbone_name)
    for layer in backbone.layers[1].layers:
        layer.trainable = False
    head = get_classification_head(N_CLASSES, head_dropout_rate)

    model = keras.models.Sequential([backbone, head])

    return model