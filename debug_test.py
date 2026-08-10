import shutil, subprocess
import test_seed8_regression
def run():
    print("Testing seed 3...")
    results = test_seed8_regression._run_seed(seed=3, games=2)
    deaths = test_seed8_regression._our_death_events(results)
    for d in deaths:
        print("Death:", d)
run()
