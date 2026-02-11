import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

MERGED_DATASET = {
    "annotation_path": "/data1/hbshim/finetune/download/100k_FT_data/merged_dataset.json",
    "data_path": "",
}

TRAIN_DATASET = {
    "annotation_path": "/data1/hbshim/finetune/download/100k_FT_data/train.json",
    "data_path": "",
}

EVAL_DATASET = {
    "annotation_path": "/data1/hbshim/finetune/download/100k_FT_data/eval.json",
    "data_path": "",
}

AI2D = {
    "annotation_path": "/data1/hbshim/finetune/download/100k_FT_data/ai2d_transformed_2187.json",
    "data_path": "",
}

LLaVA665K = {
    "annotation_path": "/data1/hbshim/finetune/download/100k_FT_data/llava665k_transformed_88076.json",
    "data_path": "",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "merged_dataset": MERGED_DATASET,
    "train_dataset": TRAIN_DATASET,
    "eval_dataset": EVAL_DATASET,
    "ai2d": AI2D,
    "llava665k": LLaVA665K,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["train_dataset", "eval_dataset", "ai2d"]  # Example dataset names with sampling rates
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
