import verres as V


def test_config_parsing_smokes():
    config = V.Config.from_path("config/exp_od_baseline.yml")
