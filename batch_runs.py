import subprocess
import sys
from src import CONFIGDIR
from src.config_dataclass import delete_config_file_contents, update_config_file

SCRIPT = "main.py"

CONFIG_FILE = CONFIGDIR.joinpath("config_overwrite.toml")

BATCH_CONFIGS = [
        {
            'run': {'name': 'Test_3'},
            'train_test': {'target': ['DNI', 'DHI']}
        },
        {
            'run': {'name': 'Test_4'},
            'train_test': {'target': ['GHI', 'DHI', 'DNI']}
        },
    ]


if __name__ == "__main__":
    for idx, config in enumerate(BATCH_CONFIGS):
        print(f"Running configuration {idx + 1}")

        delete_config_file_contents(CONFIG_FILE)

        update_config_file(CONFIG_FILE, config)

        subprocess.run([sys.executable, SCRIPT])

        print(f"Finished running configuration {idx + 1}\n")
