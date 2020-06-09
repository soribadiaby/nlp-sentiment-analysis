from sentiment_analysis.models import get_baseline_model

def test_get_baseline_models():
    n_dim = 10
    model = get_baseline_model(n_dim)
    assert model.input_shape[1] == n_dim
    