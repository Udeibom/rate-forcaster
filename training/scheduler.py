import time
from datetime import datetime, timezone

from training.retrain import retrain_if_needed

CHECK_INTERVAL_SECONDS = 60 * 60 * 24  # once per day


def run_scheduler():
    print("Retraining scheduler started")

    try:
        while True:
            print(
                f"\n🔍 Checking retraining conditions at "
                f"{datetime.now(timezone.utc).isoformat()}"
            )

            retrain_if_needed()

            print(f"⏳ Sleeping for {CHECK_INTERVAL_SECONDS} seconds")
            time.sleep(CHECK_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\n Scheduler stopped by user (Ctrl+C)")


if __name__ == "__main__":
    run_scheduler()
