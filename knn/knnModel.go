package knn

import "github.com/shubham216/go_tf_idf/similarityMetrics"

type NearestNeighbors struct {
	K                int
	Instances        [][]float64
	SimilarityMetric similarityMetrics.SIMILARITY_CHECK_METRIC
}

type Item struct {
	index    int
	distance float64
}
