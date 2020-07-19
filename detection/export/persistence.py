from os import path

from detection.utils.filesystem import ensure_directory_exists


def trained_model_exists(model, dataset, models_dir):
    return path.isfile(get_model_fullname(model, dataset, models_dir))


def get_model_fullname(model, dataset, models_dir):
    return path.join(models_dir, f'{dataset.name()}_{model.name()}.gz')


def save_model(model, dataset, models_dir):
    ensure_directory_exists(models_dir)
    model.save(get_model_fullname(model, dataset, models_dir))


def load_model(model, dataset, models_dir):
    try:
        return model.load(get_model_fullname(model, dataset, models_dir))
    except Exception as e:
        print(f'[ERROR] Cannot load trained model. Original exception: {e}.')
        return model
