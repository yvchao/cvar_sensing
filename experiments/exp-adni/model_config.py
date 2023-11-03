# structure of mlp networks
net_config = {
    "hidden_size": 10,
    "num_layer": 3,
    "resolution": 0.1,
}

# training options
train_config = {
    "batch_size": 200,
    "epochs": 200,
    "lr": 0.1,
}

# sensing options
sim_config = {
    "cost": [1.0, 1.0, 0.5, 0.5],
    "lambda": 400.0,
    "null_visit": 10.0,
    "min_dt": 0.5,
    "max_dt": 1.5,
}

# adversarial sampling options
sampler_config = {
    "interval": 10,
    "alpha": 0.1,
    "batch_size": 200,
}
