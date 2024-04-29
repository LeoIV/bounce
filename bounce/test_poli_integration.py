def test_albuterol_similarity_in_poli():
    from pathlib import Path
    import json
    from poli.objective_repository import AlbuterolSimilarityBlackBox
    from bounce.benchmarks import PoliBenchmark
    from bounce.bounce import Bounce

    TEST_FILES_DIR = Path(__file__).parent.resolve() / "test_files"
    with open(TEST_FILES_DIR / "zinc250k_metadata.json") as fp_metadata:
        metadata = json.load(fp_metadata)

    with open(TEST_FILES_DIR / "zinc250k_alphabet_stoi.json") as fp_alphabet:
        alphabet = json.load(fp_alphabet)

    benchmark = PoliBenchmark(
        f=AlbuterolSimilarityBlackBox(string_representation="SELFIES"),
        noise_std=0.0,
        sequence_length=metadata["max_sequence_length"],
        alphabet=list(alphabet.keys()),
    )
    # Parameters taken from default.gin
    bounce = Bounce(
        benchmark=benchmark,
        number_initial_points=2,
        initial_target_dimensionality=2,
        number_new_bins_on_split=2,
        maximum_number_evaluations=3,
        batch_size=1,
        results_dir=".",
    )

    bounce.run()


def test_toy_continuous_problem_from_poli():
    from poli.objective_repository import ToyContinuousBlackBox
    from bounce.benchmarks import PoliBenchmark
    from bounce.bounce import Bounce

    benchmark = PoliBenchmark(
        f=ToyContinuousBlackBox(function_name="ackley_function_01", n_dimensions=100),
        noise_std=0.0,
    )
    # Parameters taken from default.gin
    bounce = Bounce(
        benchmark=benchmark,
        number_initial_points=2,
        initial_target_dimensionality=2,
        number_new_bins_on_split=2,
        maximum_number_evaluations=4,
        batch_size=1,
        results_dir=".",
    )
    bounce.run()
