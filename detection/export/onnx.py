from tqdm import tqdm

from detection.utils.filesystem import ensure_directory_exists


def export_onnx_models(models, output_dir):
    print('Exporting ONNX models...')
    ensure_directory_exists(output_dir)

    for model in tqdm(models):
        model.export(output_dir)

    print(f'Exported ONNX models to {output_dir}.')
