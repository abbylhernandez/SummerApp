from pathlib import Path

if __package__:
    from .TrainClass import trainclass
else:
    # Support running this file as: python3 ThirdPhase/main.py
    from TrainClass import trainclass


def main():
    trainer = trainclass()
    trainer.data_set_location = Path(
        "/Users/abbyhernandez/Desktop/NEES/SummerApp/FirstPhase/trial_logs/"
        "Validation4Data-2026-07-28 10-07-28 AM/ResultClipSizeUp700"
    )
    # trainer.split_training_testing(type="alternate", percentage=80)
    # trainer.set_data_info()
    # trainer.train_model()
    # trainer.test_model()
    trainer.run_all_analyses()
    print("LDA analysis complete.")


if __name__ == "__main__":
    main()
