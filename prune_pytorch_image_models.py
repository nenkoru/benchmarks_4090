image_models = ["pytorch-image-models_float16.txt", "pytorch-image-models_float32.txt"]

for model in image_models:
    txt = open(f"raw_data/{model}", mode="r")
    filtered_warnings_error = list(filter(lambda x: "warning" not in x.lower() and "error" not in x.lower(), txt))
    filtered = []

    for i, x in enumerate(filtered_warnings_error):
        if x.startswith("Running train benchmark") and filtered_warnings_error[i+1].startswith("Train"):
            filtered.append(x)
        elif x.startswith("Running inference benchmark") and filtered_warnings_error[i+1].startswith("Infer"):
            filtered.append(x)
        elif x.startswith("Train benchmark"):
            filtered.append(x)
        elif x.startswith("Inference benchmark"):
            filtered.append(x)

    with open(f"pruned_data/{model.split('/')[-1]}", mode="w") as f:
        for x in filtered:
            f.write(x)

