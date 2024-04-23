package knn

import "github.com/shubhamraj216/go-tf-idf/similarityMetrics"

type NearestNeighbors struct {
	K                int
	Instances        [][]float64
	SimilarityMetric similarityMetrics.SIMILARITY_CHECK_METRIC
}

type Item struct {
	index    int
	distance float64
}
