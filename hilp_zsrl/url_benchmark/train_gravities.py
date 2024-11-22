import subprocess
import re
import shutil
from pathlib import Path

def backup_original_xml(xml_path: Path) -> Path:
    """Creates a backup of the original XML file."""
    backup_path = xml_path.with_suffix(".backup.xml")
    shutil.copy(xml_path, backup_path)
    return backup_path

def restore_original_xml(xml_path: Path, backup_path: Path):
    """Restores the original XML file from the backup."""
    shutil.copy(backup_path, xml_path)
    backup_path.unlink()  # Delete the backup after restoration

def update_gravity_in_xml(xml_path: Path, new_gravity: tuple):
    """
    Updates the gravity setting in a MuJoCo XML file.
    
    Parameters:
    - xml_path (Path): Path to the XML file.
    - new_gravity (tuple): New gravity vector as (x, y, z).
    """
    with open(xml_path, "r") as file:
        xml_content = file.read()
    
    # Create a string for the new gravity values
    new_gravity_str = f'gravity="{new_gravity[0]} {new_gravity[1]} {new_gravity[2]}"'
    
    # Replace the gravity setting using regex
    modified_xml = re.sub(r'gravity="[^"]*"', new_gravity_str, xml_content)
    
    # Save the updated XML
    with open(xml_path, "w") as file:
        file.write(modified_xml)

def run_training_command(gravity_name: str):
    """
    Runs the training command for a given gravity.
    """
    command = [
        "PYTHONPATH=.",
        "python",
        "url_benchmark/train_offline.py",
        "run_group=EXP",
        "device=cuda",
        "agent=sf",
        "agent.feature_learner=hilp",
        "p_randomgoal=0.375",
        "agent.hilp_expectile=0.5",
        "agent.hilp_discount=0.96",
        "agent.q_loss=False",
        "seed=0",
        "task=walker_run",
        "expl_agent=rnd",
        f"load_replay_buffer=PATH_TO_DATASET/datasets/walker/rnd/replay.pt",
        "replay_buffer_episodes=5000"
    ]
    print(f"Running command for {gravity_name} gravity:\n{' '.join(command)}")
    subprocess.run(" ".join(command), shell=True)

def train_with_gravity(xml_path: Path, gravities: dict):
    """
    Updates the XML with different gravity values and runs the training command for each.
    
    Parameters:
    - xml_path (Path): Path to the MuJoCo XML file.
    - gravities (dict): Dictionary of gravity settings (name -> gravity vector).
    """
    # Back up the original XML
    backup_path = backup_original_xml(xml_path)

    try:
        for gravity_name, gravity_value in gravities.items():
            print(f"\nTraining model with {gravity_name} gravity: {gravity_value}")

            # Update the gravity in the XML file
            update_gravity_in_xml(xml_path, gravity_value)

            # Run the training command
            run_training_command(gravity_name)

        print("\nAll training runs completed.")
        
    finally:
        # Restore the original XML file
        restore_original_xml(xml_path, backup_path)
        print("Original XML file has been restored.")

# Example usage
xml_path = Path("PATH_TO_YOUR_XML_FILE/walker.xml")
gravities = {
    "Mars": (0, 0, -3.71),
    "Jupiter": (0, 0, -24.79),
    "Moon": (0, 0, -1.62)
}

train_with_gravity(xml_path, gravities)
