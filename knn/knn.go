package knn

import "github.com/shubhamraj216/go-tf-idf/similarityMetrics"

func NewNearestNeighbors(k int, similarityMetric similarityMetrics.SIMILARITY_CHECK_METRIC) *NearestNeighbors {
	return &NearestNeighbors{
		K:                k,
		SimilarityMetric: similarityMetric,
	}
}

func (nbrs *NearestNeighbors) Fit(data [][]float64) {
	nbrs.Instances = data
}

func (nbrs *NearestNeighbors) KNeighbors(query []float64) ([]int, []float64) {
	confidenceScores := make([]similarityMetrics.Confidence, len(nbrs.Instances))

	var confidenceMetric similarityMetrics.ConfidenceScore

	switch nbrs.SimilarityMetric {
	case similarityMetrics.COSINE_SIMILARITY:
		confidenceMetric = similarityMetrics.CosineConfidence{}
	case similarityMetrics.EUCLIDEAN_DISTANCE:
		confidenceMetric = similarityMetrics.EuclideanConfidence{}
	case similarityMetrics.MANHATTAN_DISTANCE:
		confidenceMetric = similarityMetrics.ManhattanConfidence{}
	case similarityMetrics.MINKOWSKI_DISTANCE:
		confidenceMetric = similarityMetrics.MinkowskiConfidence{}
	case similarityMetrics.PEARSON_CORRELATION:
		confidenceMetric = similarityMetrics.PearsonConfidence{}
	default:
		confidenceMetric = similarityMetrics.CosineConfidence{}
	}

	confidenceScores = confidenceMetric.GetConfidenceScore(query, nbrs.Instances)

	topKConfidence := make([]float64, nbrs.K)
	topKIndices := make([]int, nbrs.K)

	for i, confidence := range confidenceScores {
		if i >= nbrs.K {
			break
		}
		topKConfidence[i] = confidence.ConfidenceScore
		topKIndices[i] = confidence.Index
	}

	return topKIndices, topKConfidence
}
