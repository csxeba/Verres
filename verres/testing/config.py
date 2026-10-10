import verres as V


def test_config_parsing_smokes():
    config = V.Config.from_paths("config/exp_od_baseline.yml", "config/class_mappings/cocodoom.yml")
