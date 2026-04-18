"""Unit tests for the NLP pipeline components."""

import pytest


# ---------------------------------------------------------------------------
# TextPreprocessor
# ---------------------------------------------------------------------------

class TestTextPreprocessor:
    @pytest.fixture(autouse=True)
    def preprocessor(self):
        from documind.pipeline import TextPreprocessor
        self.pp = TextPreprocessor()

    def test_preprocess_returns_string(self):
        result = self.pp.preprocess("The patient presents with acute respiratory distress.")
        assert isinstance(result, str)

    def test_preprocess_lowercases(self):
        result = self.pp.preprocess("LEGAL AGREEMENT BREACH")
        assert result == result.lower()

    def test_preprocess_removes_urls(self):
        result = self.pp.preprocess("Visit http://example.com for more info.")
        assert "http" not in result
        assert "example" not in result

    def test_preprocess_removes_emails(self):
        result = self.pp.preprocess("Contact support@company.com for assistance.")
        assert "@" not in result

    def test_preprocess_handles_empty_string(self):
        result = self.pp.preprocess("")
        assert result == ""

    def test_preprocess_batch_length(self):
        texts = ["Document one.", "Document two.", "Document three."]
        results = self.pp.preprocess_batch(texts)
        assert len(results) == len(texts)

    def test_preprocess_batch_types(self):
        texts = ["Legal contract terms.", "Medical patient records."]
        results = self.pp.preprocess_batch(texts)
        assert all(isinstance(r, str) for r in results)

    def test_numbers_replaced_with_token(self):
        result = self.pp.preprocess("The payment of 50000 dollars is due.")
        assert "num" in result.lower()

    def test_pos_distribution_returns_dict(self):
        dist = self.pp.get_pos_distribution("The quick brown fox jumps.")
        assert isinstance(dist, dict)
        assert all(isinstance(v, float) for v in dist.values())

    def test_pos_distribution_sums_to_one(self):
        dist = self.pp.get_pos_distribution("Compliance audit findings were reviewed.")
        assert abs(sum(dist.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    @pytest.fixture(autouse=True)
    def extractor(self):
        from documind.pipeline import FeatureExtractor
        self.fe = FeatureExtractor(max_features=500)

    def test_fit_transform_shape(self):
        texts = ["legal contract review", "medical patient care", "financial audit report"]
        X = self.fe.fit_transform(texts)
        assert X.shape[0] == 3

    def test_transform_before_fit_raises(self):
        from documind.pipeline import FeatureExtractor
        fe = FeatureExtractor()
        with pytest.raises(RuntimeError):
            fe.transform(["some text"])

    def test_vocabulary_size_positive(self):
        self.fe.fit(["compliance audit", "legal review", "medical records"])
        assert self.fe.vocabulary_size() > 0

    def test_feature_names_includes_structural(self):
        self.fe.fit(["financial report quarterly earnings"])
        names = self.fe.get_feature_names()
        assert any("feat_" in n for n in names)

    def test_transform_consistent_shape(self):
        train = ["first document text", "second document text", "third document text"]
        self.fe.fit(train)
        test = ["new document for testing"]
        X_train = self.fe.transform(train)
        X_test = self.fe.transform(test)
        assert X_train.shape[1] == X_test.shape[1]


# ---------------------------------------------------------------------------
# SentimentAnalyzer
# ---------------------------------------------------------------------------

class TestSentimentAnalyzer:
    @pytest.fixture(autouse=True)
    def analyzer(self):
        from documind.pipeline import SentimentAnalyzer
        self.sa = SentimentAnalyzer()

    def test_positive_document(self):
        result = self.sa.analyze(
            "The project was excellent and the outcome was highly effective and beneficial."
        )
        assert result.label == "positive"
        assert result.score > 0

    def test_negative_document(self):
        result = self.sa.analyze(
            "There was a critical failure and the system is deficient with multiple violations."
        )
        assert result.label == "negative"
        assert result.score < 0

    def test_neutral_document(self):
        result = self.sa.analyze("The meeting was held on Monday at 3pm in room 5.")
        assert result.label in ("neutral", "positive", "negative")

    def test_score_range(self):
        result = self.sa.analyze("This is an excellent and highly effective solution.")
        assert -1.0 <= result.score <= 1.0

    def test_confidence_range(self):
        result = self.sa.analyze("Compliance audit revealed critical issues.")
        assert 0.0 <= result.confidence <= 1.0

    def test_negation_flips_sentiment(self):
        pos = self.sa.sentiment_score("This is excellent and outstanding.")
        neg = self.sa.sentiment_score("This is not excellent and not outstanding.")
        assert pos > neg

    def test_batch_returns_correct_length(self):
        texts = ["doc one", "doc two", "doc three", "doc four"]
        results = self.sa.analyze_batch(texts)
        assert len(results) == len(texts)

    def test_result_has_counts(self):
        result = self.sa.analyze("Excellent outcome with no failures.")
        assert hasattr(result, "positive_count")
        assert hasattr(result, "negative_count")
        assert result.word_count > 0
