"""Guard against held-out labels or images entering learned preprocessing."""
import numpy as np
from kinship.algorithms import kinver


def test_fisher_and_pca_are_fitted_inside_each_training_fold(monkeypatch):
    rng = np.random.default_rng(7)
    features = rng.normal(size=(30, 70))
    # Distinct people per pair make held-out rows unambiguous.
    idxa = np.arange(0, 30, 2)
    idxb = idxa + 1
    folds = np.repeat([1, 2, 3], 5)
    labels = np.tile([0, 1, 0, 1, 1], 3)
    monkeypatch.setattr(kinver, '_load_feature_matrix', lambda *args: {
        'ux': features, 'idxa': idxa, 'idxb': idxb,
        'fold': folds, 'matches': labels,
    })
    fisher_inputs, pca_inputs = [], []
    def fisher(x, y, fraction):
        fisher_inputs.append((x.copy(), y.copy()))
        return np.arange(50)
    monkeypatch.setattr(kinver, '_top_fisher_indices', fisher)
    real_pca = kinver.PCA
    class ObservedPCA(real_pca):
        def fit(self, x, y=None):
            pca_inputs.append(x.copy())
            return super().fit(x, y)
    monkeypatch.setattr(kinver, 'PCA', ObservedPCA)
    kinver.run_kinver('fs', use_vggface=True, use_vggf=False,
                     use_mnrml=False, use_feature_selection=True)
    normalized = kinver._normalize_rows(features)
    assert len(fisher_inputs) == len(pca_inputs) == 3
    for index, fold in enumerate([1, 2, 3]):
        train = folds != fold
        np.testing.assert_allclose(fisher_inputs[index][0],
            np.abs(normalized[idxa[train]] - normalized[idxb[train]]))
        np.testing.assert_array_equal(fisher_inputs[index][1], labels[train])
        expected = np.vstack([normalized[idxa[train], :50], normalized[idxb[train], :50]])
        np.testing.assert_allclose(pca_inputs[index], expected)
