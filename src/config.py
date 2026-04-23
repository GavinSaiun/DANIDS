from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATHS = {
    "NF-UNSW-NB15-v3": BASE_DIR / "Datasets" / "NF-UNSW-NB15-v3" / "f7546561558c07c5_NFV3DATA-A11964_A11964" / "data" / "NF-UNSW-NB15-v3.csv",
    "NF-ToN-IoT-v3": BASE_DIR / "Datasets" / "NF-ToN-IoT-v3" / "02934b58528a226b_NFV3DATA-A11964_A11964" / "data" / "NF-ToN-IoT-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": BASE_DIR / "Datasets" / "NF-CSE-CIC-IDS2018-v3" / "f78acbaa2afe1595_NFV3DATA-A11964_A11964" / "data" / "NF-CICIDS2018-v3.csv",
}

ARTIFACTS_DIR = BASE_DIR / "artifacts"
STAGE1_DIR = ARTIFACTS_DIR / "stage1"
PAIR_DIR = ARTIFACTS_DIR / "pairs"

for d in [ARTIFACTS_DIR, STAGE1_DIR, PAIR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DROP_COLUMNS = [
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "Label",
    "Attack",
]

LABEL_COLUMN = "Label"
ATTACK_COLUMN = "Attack"

RANDOM_STATE = 42