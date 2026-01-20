import main
import perplexity


def profile_batch() -> None:
    data = main.load_test_data()
    texts = main.get_test_texts(data)

    context = main.load_model()

    main.run_batch_perplexity(
        context,
        texts,
        perplexity.calculate_perplexity_onnxruntime_genai,
    )


if __name__ == "__main__":
    profile_batch()
