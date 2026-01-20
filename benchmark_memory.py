import main
import perplexity


def profile_batch() -> None:
    print("Loading test data...")
    data = main.load_test_data()
    texts = main.get_test_texts(data)

    print("Loading model...")
    context = main.load_model()

    print("Running perplexity calculations...")
    main.run_batch_perplexity(
        context, texts, perplexity.calculate_perplexity_onnxruntime_genai
    )

    print("Done.")


if __name__ == "__main__":
    profile_batch()
